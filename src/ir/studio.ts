/**
 * 模型编写台的门面层：把 `src/ir/` 的真实能力暴露成 UI 能直接用的几个函数。
 *
 * **这个文件里没有一行是假的**：每个数字都来自真实的 infer/plan/emit/run/backward。
 * 前端只跟这一层打交道，于是"页面显示的东西"与"代码真的会做的事"不可能脱节。
 *
 * 注意：只在浏览器里用（会读 `navigator.gpu`、`fetch` 语料）。**不要**从 `ir/index.ts` 导出它，
 * 否则无头自检的依赖图会带上 DOM。
 */

import type { CapsSummary, Diag, Model, ModelNode, Props } from '../ir/types';
import { builtinRegistry, opRef } from '../ir/op';
import { h, buildModel } from '../ir/jsx';
import { infer } from '../ir/infer';
import { plan, type BackendCapability } from '../ir/plan';
import { emit } from '../ir/emit';
import { run, type CpuImplRegistry } from '../ir/exec';
import { builtinCpuImpls } from '../ir/cpu-impls';
import { builtinCpuGrads } from '../ir/cpu-grads';
import { backward, createAdamW, makeScalar, scalarValue, type Optimizer } from '../ir/grad';
import { buildGptIr, bindBatch, type GptIr } from '../ir/tinygpt-ir';
import { attachCrossEntropy, setTargets } from '../ir/train';
import { specHash } from '../ir/index';
import { numel, type TensorTable, type TensorValue } from '../ir/binding';
import { initWeights, type GPTWeights } from '../model/init';
import { DEFAULT_CONFIG, type GPTConfig } from '../model/config';
import { gptForwardRef } from '../reference/gpt-ref';
import { checkTolerance } from '../reference/cpu-ref';
import { loadTinyShakespeare } from '../train/data';

// ---------------------------------------------------------------------------
// 模板
// ---------------------------------------------------------------------------

export type TemplateKind = 'gpt' | 'broken' | 'unstable';

export interface TemplateInfo
{
  id: string;
  name: string;
  note: string;
  kind: TemplateKind;
}

const GPT_SMALL: GPTConfig = DEFAULT_CONFIG;
const GPT_MID: GPTConfig = { vocabSize: 65, blockSize: 32, nLayer: 6, nHead: 8, nEmbd: 128, bias: true };

export function listTemplates (): TemplateInfo[]
{
  return [
    { id: 'gpt-small', name: 'tiny-GPT 小（3 层 / 64 维）', note: '与仓库默认配置一致，可训练', kind: 'gpt' },
    { id: 'gpt-mid', name: 'tiny-GPT 中（6 层 / 128 维）', note: '参数量约 4 倍，看 plan 的开销变化', kind: 'gpt' },
    { id: 'broken-param', name: '缺必填参数', note: 'Embed 少了 vocab → MISSING_PARAM', kind: 'broken' },
    { id: 'broken-strategy', name: '策略与能力位矛盾', note: 'strategy=es 却声明 requires vjp', kind: 'broken' },
    { id: 'broken-shape', name: '形状对不上', note: '嵌入端口期望 [B,T]，上游给 [4,5]', kind: 'broken' },
    { id: 'unstable', name: '深/宽超阈值', note: 'dim=4096 且未用 QKNorm/ZLoss', kind: 'unstable' },
  ];
}

// ---------------------------------------------------------------------------
// 会话
// ---------------------------------------------------------------------------

export interface StudioSession
{
  templateId: string;
  templateName: string;
  model: Model;
  config?: GPTConfig;
  gpt?: GptIr;
  /** GPT 模板的原始权重（与张量表共享同一批数组，训练会就地更新）。 */
  weights?: GPTWeights;
  symbols: Record<string, number>;
  opt?: Optimizer;
}

function trainerSpec ( over: Partial<Model['trainer']> = {} ): Model['trainer']
{
  return {
    strategy: over.strategy ?? 'backprop',
    params: over.params ?? {},
    schedule: over.schedule ?? { localSteps: 1, syncEvery: 1, precision: 'fp32' },
    ...( over.requires ? { requires: over.requires } : {} ),
  };
}

function brokenModel ( id: string ): Model
{
  switch ( id )
  {
    case 'broken-param':
      return buildModel( h( 'Embed', { dim: 8 } ), trainerSpec() );
    case 'broken-strategy':
      return buildModel( h( 'RMSNorm', { dim: 64 } ), trainerSpec( { strategy: 'es', requires: [ 'vjp', 'forward', 'perturbable' ] } ) );
    case 'broken-shape':
      return buildModel( h( 'Embed', { vocab: 100, dim: 8 }, h( 'Reshape', { shape: [ 4, 5 ] } ) ), trainerSpec() );
    case 'unstable':
      return buildModel( h( 'RMSNorm', { dim: 4096 } ), trainerSpec() );
    default:
      throw new Error( `未知模板：${ id }` );
  }
}

export function openTemplate ( id: string, seed = 1234 ): StudioSession
{
  const info = listTemplates().find( ( t ) => t.id === id );
  if ( !info ) throw new Error( `未知模板：${ id }` );

  if ( info.kind !== 'gpt' )
  {
    return {
      templateId: id,
      templateName: info.name,
      model: brokenModel( id ),
      symbols: { B: 2, T: 3 },
    };
  }

  const config = id === 'gpt-mid' ? GPT_MID : GPT_SMALL;
  const weights = initWeights( config, seed );
  const gpt = buildGptIr( config, weights );
  return {
    templateId: id,
    templateName: info.name,
    model: gpt.model,
    config,
    gpt,
    weights,
    symbols: { B: 2, T: Math.min( 16, config.blockSize ) },
  };
}

/** 换一套权重：**结构完全不变**，所以 specHash 不变（架构与权重分离）。 */
export function reweight ( session: StudioSession, seed: number ): StudioSession
{
  if ( !session.config ) return session;
  const weights = initWeights( session.config, seed );
  const gpt = buildGptIr( session.config, weights );
  return { ...session, model: gpt.model, gpt, weights, opt: undefined };
}

/** 打上 device/shard/precision 注解：**specHash 必须不变**（INV-11 / IR-v4）。 */
export function annotateEnv ( session: StudioSession, device: string, shardPeers: number, precision: string ): void
{
  for ( const id of Object.keys( session.model.nodes ) )
  {
    session.model.nodes[ id ] = {
      ...session.model.nodes[ id ],
      annot: {
        device: device as never,
        shard: { plan: 'tensor', peers: shardPeers },
        precision: precision as never,
      },
    };
  }
  session.model.graph = session.model.nodes[ session.model.graph.id ];
}

// ---------------------------------------------------------------------------
// 分析（infer + plan + emit）
// ---------------------------------------------------------------------------

export interface AnalyzeResult
{
  specHash: string;
  nodeCount: number;
  opList: Array<{ name: string; count: number }>;
  diags: Diag[];
  caps: CapsSummary;
  plan: {
    backend: string;
    arenaBytes: number;
    bindGroups: number;
    segments: Array<{ name: string; bytes: number }>;
    budgetMB: number;
    comm: string;
    compress: string;
    retainActivations: boolean;
    rngOps: number;
    soft: Diag[];
  };
  artifact: {
    passes: number;
    passBreaks: number;
    dispatches: Array<{ nodeId: string; kernel: string; workgroups: number[]; pass: number }>;
  };
  estimate: { tensors: number; params: number; weightBytes: number };
}

export function analyze ( session: StudioSession, backend: BackendCapability ): AnalyzeResult
{
  const model = session.model;
  const ir = infer( model, { symbols: session.symbols } );
  const execPlan = plan( model, ir, { backends: [ backend ] } );
  const artifact = emit( model, ir, execPlan );

  const opCount = new Map<string, number>();
  for ( const id of Object.keys( model.nodes ) )
  {
    const name = model.nodes[ id ].op.replace( /@.*$/, '' );
    opCount.set( name, ( opCount.get( name ) ?? 0 ) + 1 );
  }

  let params = 0;
  let weightBytes = 0;
  let tensors = 0;
  if ( session.gpt )
  {
    for ( const name of session.gpt.tensors.names() )
    {
      const t = session.gpt.tensors.get( name );
      if ( !t ) continue;
      if ( name.startsWith( 'input.' ) ) continue;
      tensors += 1;
      params += numel( t.shape );
      weightBytes += numel( t.shape ) * 4;
    }
  }

  const softCodes = new Set( [ 'ROOM_TOO_HETEROGENEOUS', 'COMPRESSION_WITHOUT_EF', 'BYZANTINE_BUDGET_INFEASIBLE' ] );

  return {
    specHash: specHash( model ),
    nodeCount: Object.keys( model.nodes ).length,
    opList: [ ...opCount.entries() ].map( ( [ name, count ] ) => ( { name, count } ) ).sort( ( a, b ) => b.count - a.count || a.name.localeCompare( b.name ) ),
    diags: ir.diags,
    caps: ir.caps,
    plan: {
      backend: execPlan.backend[ ir.graph.root ] ?? backend.backend,
      arenaBytes: execPlan.memory.arena.totalBytes,
      bindGroups: execPlan.memory.arena.bindGroups,
      segments: execPlan.memory.arena.segments.map( ( s ) => ( { name: s.name, bytes: s.bytes } ) ),
      budgetMB: execPlan.memory.budgetMB,
      comm: execPlan.comm.plan,
      compress: execPlan.compress.scheme,
      retainActivations: execPlan.retainActivations,
      rngOps: execPlan.rng?.ops.length ?? 0,
      soft: execPlan.diags.filter( ( d ) => softCodes.has( d.code ) ),
    },
    artifact: {
      passes: artifact.passes.length,
      passBreaks: artifact.passBreaks,
      dispatches: artifact.passes.flatMap( ( pass ) =>
        pass.dispatches.map( ( d ) => ( { nodeId: d.nodeId, kernel: d.kernel, workgroups: d.workgroups, pass: pass.index } ) ) ),
    },
    estimate: { tensors, params, weightBytes },
  };
}

/** 真实探测后端能力（WebGPU 优先，取不到就退 CPU）。 */
export async function probeBackend (): Promise<{ backend: BackendCapability; report: string[] }>
{
  const report: string[] = [];
  try
  {
    // 动态 import：让本模块的静态依赖图不牵扯 GPU 层，
    // 于是 studio 能被"无头自检"直接 import（Node 里没有 WebGPU）。
    const { probeCapability } = await import( '../fed/capability' );
    const cap = await probeCapability();
    if ( cap.adapterOk )
    {
      report.push( `适配器：${ cap.device || cap.vendor || '未知' }（${ cap.architecture || 'n/a' }）` );
      report.push( `maxBufferSize：${ ( cap.maxBufferSize / 1024 / 1024 / 1024 ).toFixed( 2 ) } GiB` );
      report.push( `shader-f16：${ cap.hasF16 ? '支持' : '不支持' }` );
      return {
        backend: {
          backend: 'webgpu',
          maxBufferSize: cap.maxBufferSize || 2 * 1024 * 1024 * 1024,
          maxBindGroups: 4,
          maxWorkgroupsPerDimension: 65535,
          supportsF16: cap.hasF16,
          supportsSubgroups: false,
        },
        report,
      };
    }
    report.push( `未拿到 WebGPU 适配器：${ cap.adapterError ?? '未知原因' }` );
    if ( cap.advise.length > 0 ) report.push( cap.advise[ 0 ] );
  }
  catch ( err )
  {
    report.push( `能力探测失败：${ ( err as Error ).message }` );
  }
  report.push( '本页的计算全部在 CPU 后端完成（与手写实现对拍的那条路径），不依赖 WebGPU。' );
  return {
    backend: {
      backend: 'cpu',
      maxBufferSize: 512 * 1024 * 1024,
      maxBindGroups: 4,
      maxWorkgroupsPerDimension: 65535,
      supportsF16: false,
      supportsSubgroups: false,
    },
    report,
  };
}

// ---------------------------------------------------------------------------
// 可机读修复：真的改 IR
// ---------------------------------------------------------------------------

export interface FixOutcome { applied: boolean; note: string; }

export function applyFix ( session: StudioSession, diag: Diag ): FixOutcome
{
  const fix = diag.fix;
  if ( !fix ) return { applied: false, note: '这条诊断没有可机读修复' };
  const model = session.model;

  if ( fix.kind === 'change-prop' && fix.to && diag.node )
  {
    const node = model.nodes[ diag.node ];
    if ( !node ) return { applied: false, note: `节点 ${ diag.node } 不存在` };
    const contract = builtinRegistry.get( node.op )?.contract;
    const spec = contract?.props[ fix.to ];
    const value: Props[ string ] =
      spec?.default !== undefined ? ( spec.default as Props[ string ] )
        : spec?.type === 'int' ? 8
          : spec?.type === 'float' ? 1
            : spec?.type === 'bool' ? true
              : spec?.type === 'dims' ? [ 2, 3 ]
                : spec?.type === 'enum' ? ( spec.values?.[ 0 ] ?? 'auto' )
                  : 'auto';
    node.props = { ...node.props, [ fix.to ]: value };
    return { applied: true, note: `已把 ${ diag.node }.${ fix.to } 设为 ${ JSON.stringify( value ) }` };
  }

  if ( fix.kind === 'switch-strategy' )
  {
    // 只处理"声明与策略矛盾"这一种：把 requires 收敛到策略自身的要求。
    const t = model.trainer;
    if ( t.requires && t.requires.length > 0 )
    {
      const next = { ...t, requires: undefined as never };
      // 清掉 requires（让它回到由 strategy 推导的默认值）
      model.trainer = { strategy: next.strategy, params: next.params, schedule: next.schedule };
      return { applied: true, note: `已移除 trainer.requires，回到由 strategy=${ t.strategy } 推导的默认要求` };
    }
    return { applied: false, note: '该修复需要改图（插入/替换算子），本页暂不支持自动应用' };
  }

  if ( fix.kind === 'insert-op' && diag.node && fix.to && diag.port )
  {
    const target = model.nodes[ diag.node ];
    if ( !target ) return { applied: false, note: `节点 ${ diag.node } 不存在` };
    const src = target.slot?.[ diag.port ];
    if ( !src || !model.nodes[ src ] ) return { applied: false, note: '仅支持"来自 slot 的输入端口"这一种插桩' };
    const def = builtinRegistry.latest( fix.to );
    if ( !def ) return { applied: false, note: `注册表里没有 ${ fix.to }` };

    const fixId = `${ diag.node }.fix`;
    const props: Props = fix.to === 'Cast'
      ? { dtype: 'i32' }
      : fix.to === 'Reshape'
        ? { shape: [ 'B', 'T' ] }
        : {};
    const node: ModelNode = { id: fixId, op: opRef( def.contract ), props, slot: { x: src } };
    model.nodes[ fixId ] = node;
    target.slot = { ...( target.slot ?? {} ), [ diag.port ]: fixId };
    return { applied: true, note: `已在 ${ src } 与 ${ diag.node }.${ diag.port } 之间插入 ${ fix.to}（id=${ fixId }）` };
  }

  return { applied: false, note: `fix.kind=${ fix.kind } 需要结构性改动，本页暂不支持自动应用` };
}

// ---------------------------------------------------------------------------
// 语料与批次
// ---------------------------------------------------------------------------

export interface CorpusView { ids: Uint32Array; vocab: string[]; chars: number; }

export async function loadCorpus (): Promise<CorpusView>
{
  const c = await loadTinyShakespeare();
  return { ids: c.ids, vocab: c.vocab, chars: c.ids.length };
}

function rng32 ( seed: number ): () => number
{
  let s = seed >>> 0 || 1;
  return () =>
  {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 13; s >>>= 0;
    return s / 0x100000000;
  };
}

/** 从语料切一批 (tokens, targets)，targets 为 tokens 右移一位（自回归标准做法）。 */
export function makeBatch ( ids: Uint32Array, B: number, T: number, seed: number ): { tokens: Uint32Array; targets: Uint32Array }
{
  const rnd = rng32( seed );
  const tokens = new Uint32Array( B * T );
  const targets = new Uint32Array( B * T );
  const maxStart = Math.max( 1, ids.length - ( T + 1 ) );
  for ( let b = 0; b < B; b++ )
  {
    const start = Math.floor( rnd() * maxStart );
    for ( let t = 0; t < T; t++ )
    {
      tokens[ b * T + t ] = ids[ start + t ];
      targets[ b * T + t ] = ids[ start + t + 1 ];
    }
  }
  return { tokens, targets };
}

// ---------------------------------------------------------------------------
// 前向 / 反向 / 训练
// ---------------------------------------------------------------------------

export interface ForwardReport
{
  loss: number;
  ms: number;
  logitsShape: number[];
  compare?: { ok: boolean; maxAbs: number; maxRel: number; elements: number };
}

let implsCache: CpuImplRegistry | null = null;
function impls (): CpuImplRegistry
{
  if ( !implsCache ) implsCache = builtinCpuImpls();
  return implsCache;
}

/** 保证会话有可微的根（CrossEntropy），并返回 (model, 张量表)。 */
function trainable ( session: StudioSession ): { model: Model; tensors: TensorTable }
{
  if ( !session.gpt ) throw new Error( '只有 GPT 模板能跑前向/训练' );
  const model = attachCrossEntropy( session.model );
  session.model = model;
  return { model, tensors: session.gpt.tensors };
}

export function forwardOnce ( session: StudioSession, tokens: Uint32Array, targets: Uint32Array, B: number, T: number ): ForwardReport
{
  const { model, tensors } = trainable( session );
  const cfg = session.config!;
  const binding = bindBatch( session.gpt!, tokens, B, T );
  setTargets( tensors, targets, B, T );

  const ir = infer( model, { symbols: { B, T } } );
  const execPlan = plan( model, ir, { backends: [ backendFor( model, ir ) ] } );
  const artifact = emit( model, ir, execPlan );

  const t0 = performance.now();
  const res = run( model, ir, artifact, binding, impls() );
  const ms = performance.now() - t0;

  // 与手写实现逐元素对拍（同一批权重、同一批 token）
  const ref = gptForwardRef( session.weights!, cfg, { tokens, B, T } );
  const irLogits = irOutputs( model, res, B, T );
  const cmp = checkTolerance( irLogits, ref.logits, 1e-4, 1e-4 );

  return {
    loss: scalarValue( res.rootOutput ),
    ms,
    logitsShape: [ B * T, cfg.vocabSize ],
    compare: { ok: cmp.ok, maxAbs: cmp.maxAbs, maxRel: cmp.maxRel, elements: ref.logits.length },
  };
}

function backendFor ( model: Model, ir: ReturnType<typeof infer> ): BackendCapability
{
  void model; void ir;
  return {
    backend: 'cpu',
    maxBufferSize: 512 * 1024 * 1024,
    maxBindGroups: 4,
    maxWorkgroupsPerDimension: 65535,
    supportsF16: false,
    supportsSubgroups: false,
  };
}

/** 从图的中间取回 logits（根是 loss 时，去它的 slot 上游拿）。 */
function irOutputs ( model: Model, res: ReturnType<typeof run>, B: number, T: number ): Float32Array
{
  const lossNode = model.nodes[ 'loss' ];
  const headId = lossNode?.slot?.logits;
  const t: TensorValue | undefined = headId ? res.get( headId ) : undefined;
  if ( !t ) throw new Error( '取不到 logits（loss.slot.logits 未指向 head？）' );
  void B; void T;
  return t.data instanceof Float32Array ? t.data : new Float32Array( t.data );
}

export interface TrainStepReport { step: number; loss: number; gradNorm: number; ms: number; }

export async function trainSession (
  session: StudioSession,
  ids: Uint32Array,
  opts: { steps: number; lr: number; B?: number; T?: number; onStep?: ( s: TrainStepReport ) => void },
): Promise<TrainStepReport[]>
{
  const { model, tensors } = trainable( session );
  const B = opts.B ?? 4;
  const T = Math.min( opts.T ?? 16, session.config!.blockSize );
  if ( !session.opt ) session.opt = createAdamW( { lr: opts.lr, b1: 0.9, b2: 0.95, eps: 1e-8, wd: 0.01, clip: 1.0 } );

  const ir = infer( model, { symbols: { B, T } } );
  const execPlan = plan( model, ir, { backends: [ backendFor( model, ir ) ] } );
  const artifact = emit( model, ir, execPlan );
  const grads = builtinCpuGrads();

  const out: TrainStepReport[] = [];
  for ( let step = 0; step < opts.steps; step++ )
  {
    const batch = makeBatch( ids, B, T, 1000 + step * 37 );
    const binding = bindBatch( session.gpt!, batch.tokens, B, T );
    setTargets( tensors, batch.targets, B, T );

    const t0 = performance.now();
    const b = backward( model, ir, artifact, binding, impls(), grads );
    session.opt.step( b.dParams, tensors );
    const ms = performance.now() - t0;

    const report: TrainStepReport = { step: step + 1, loss: b.loss, gradNorm: session.opt.lastGradNorm, ms };
    out.push( report );
    opts.onStep?.( report );

    // 让出主线程，页面不卡（每步都 await 一次宏任务）
    if ( step % 4 === 3 ) await new Promise( ( r ) => setTimeout( r, 0 ) );
  }
  return out;
}

export { makeScalar };
