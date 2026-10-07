/**
 * plan()：唯一的调度决策点（design/06-执行与调度引擎.md §2）。
 *
 * 输入 (IR, BackendCapability, ClusterCapability, Budget) → 输出 ExecutionPlan：
 * 每节点的后端/策略/角色/分片/精度/融合/RNG/通信/压缩/信任预算/pass 划分。
 *
 * 两条硬性设计：
 *   1. **策略感知**（06 §4 / ENG-V4）：读 `trainer.strategy`，反传才保留激活 + 排反向节点；
 *      前向-only 策略（ES/SPSA/…）retainActivations=false 且 backwardNodes=[]。
 *   2. **纯静态、可复现**：禁止 `Math.random()` / `Date.now()`；所有启发式都是
 *      确定性的整数/浮点运算，同一输入必得同一计划。
 *
 * 硬约束不满足时产 error Diag 但**尽力给合法计划**（不抛异常）；软判据产 warn + fix。
 */

import type {
  AutoFix,
  Cap,
  Diag,
  DiagCode,
  Dim,
  InferResult,
  Model,
  NodeId,
  Precision,
  StrategyId,
  TypedNode,
} from './types';
import { MAX_BIND_GROUPS, MAX_BUFFER_SIZE, STRATEGY_REQUIRES } from './types';
import type { OpContract, OpRegistry } from './op';
import { builtinRegistry } from './op';
import { ensureBuiltinOps } from './ops';
import type { ExprEnv } from './expr';
import { propNumber } from './expr';

// ---------------------------------------------------------------------------
// 公开类型契约（06 §2）
// ---------------------------------------------------------------------------

export type BackendId = 'webgpu' | 'cpu' | 'webnn' | 'wasm';

export interface BackendCapability
{
  backend: BackendId;
  maxBufferSize: number;            // 字节
  maxBindGroups: number;            // 本项目实测 = 4
  maxWorkgroupsPerDimension: number;// 65535
  supportsF16: boolean;
  supportsSubgroups: boolean;
}

export interface ClusterPeer { id: string; backend: BackendId; caps: Cap[]; strategy?: StrategyId; }
export interface ClusterCapability { peers: ClusterPeer[]; }

export interface Budget
{
  memMB?: number;
  flops?: number;
  /** 拜占庭预算 f/n 与房间规模（ADR-029）。 */
  byzantineF?: number;
  roomSize?: number;
}

export interface PlanEnv
{
  backends: BackendCapability[];
  cluster?: ClusterCapability;
  budget?: Budget;
  registry?: OpRegistry;
}

export interface ArenaSegment { name: string; kind: 'weight' | 'activation' | 'kv' | 'uniform'; bytes: number; }
export interface ArenaLayout { segments: ArenaSegment[]; totalBytes: number; bindGroups: number; }

export interface DispatchItem { nodeId: NodeId; kernel: string; workgroups: number[]; }
export interface PassPlan { index: number; dispatches: DispatchItem[]; }

export interface RNGPlan { ops: Array<{ nodeId: NodeId; counterScope: string; from: number; to: number }>; }
export interface CommPlan { plan: 'full-delta' | 'sparse' | 'quant'; syncEvery: number; maxFrameBytes: number; }
export interface CompressPlan { scheme: 'none' | 'topk' | 'sign' | 'quant'; errorFeedback: boolean; }
export interface TrustPlan { byzantineBudget: number; minRoomSize: number; expectedBias: number; }
export interface BackendRule { backend: BackendId; rule: string; }

export interface ExecutionPlan
{
  backend: Record<NodeId, BackendId>;
  strategy: Record<NodeId, StrategyId>;   // 每节点策略（降级阶梯落点）
  role: Record<NodeId, string>;
  shard: { plan: 'layer' | 'tensor' | 'expert' | 'auto'; peers: number };
  precision: { global: Precision };
  fusion: Array<NodeId[]>;
  memory: { arena: ArenaLayout; activationsBytes: number; budgetMB: number };
  rng: RNGPlan | null;
  comm: CommPlan;
  compress: CompressPlan;
  recompute: NodeId[];
  trust: TrustPlan;
  backendRules: BackendRule[];
  /** ENG-V4：前向-only 策略必须为 false，且 backwardNodes 为空。 */
  retainActivations: boolean;
  /** ENG-V4：反向节点集合（前向-only 策略下为空）。 */
  backwardNodes: NodeId[];
  passes: PassPlan[];
  diags: Diag[];
}

// ---------------------------------------------------------------------------
// 常量与确定性启发式参数
// ---------------------------------------------------------------------------

const KIB = 1024;
const MIB = 1024 * 1024;
const MAX_DISPATCH_DIM = 65535;
/** 一个 workgroup 处理 64 元素（启发式，稳定即可，不要求物理精确）。 */
const ELEMS_PER_WORKGROUP = 64;
/** 收敛邻域目标：把 κ* 压到 0.1 以下所需的房间规模阈值（docs/评审记录第四轮推导 15）。 */
const KAPPA_TARGET = 0.1;
/** uniform 段固定预留 64 KiB（小且稳定）。 */
const UNIFORM_BYTES = 64 * KIB;

/** 无后端能力信息时的兜底（WebGPU 主编译目标，06 §6）。 */
const DEFAULT_BACKEND: BackendCapability = {
  backend: 'webgpu',
  maxBufferSize: MAX_BUFFER_SIZE,
  maxBindGroups: MAX_BIND_GROUPS,
  maxWorkgroupsPerDimension: MAX_DISPATCH_DIM,
  supportsF16: true,
  supportsSubgroups: false,
};

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------

function mkDiag (
  level: Diag['level'],
  code: DiagCode,
  message: string,
  messageEn: string,
  fix?: AutoFix,
  node?: NodeId,
): Diag
{
  const d: Diag = { level, code, message, messageEn };
  if ( fix ) d.fix = fix;
  if ( node ) d.node = node;
  return d;
}

/** 拓扑序；缺 `order` 时退化为 id 字典序（保证确定性）。 */
function topoOrder ( ir: InferResult ): NodeId[]
{
  const order = ir.graph?.order;
  if ( order && order.length > 0 ) return order;
  return Object.keys( ir.graph?.nodes ?? {} ).sort();
}

/** 契约的 arena 区段读写（缺省为空）。 */
function segsOf ( c: OpContract | undefined ): { reads: string[]; writes: string[] }
{
  const wr = c?.writesReads;
  return { reads: wr?.reads ?? [], writes: wr?.writes ?? [] };
}

/** 由输出 shape 启发式算 workgroups（符号维视作 1；可复现）。 */
function workgroupsOf ( typed: TypedNode | undefined ): number[]
{
  const outs = typed?.outputs ?? {};
  const first = Object.values( outs )[ 0 ] ?? null;
  const shape = ( first as { shape?: Dim[] } | null )?.shape;
  let elems = 1;
  if ( shape )
  {
    for ( const d of shape )
      if ( typeof d === 'number' && d > 0 ) elems *= d;
  }
  elems = Math.max( 1, Math.floor( elems ) );
  let x = Math.ceil( elems / ELEMS_PER_WORKGROUP );
  const wg = [ 1, 1, 1 ];
  let i = 0;
  while ( x > MAX_DISPATCH_DIM && i < 2 )
  {
    wg[ i ] = MAX_DISPATCH_DIM;
    x = Math.ceil( x / MAX_DISPATCH_DIM );
    i++;
  }
  wg[ i ] = Math.min( MAX_DISPATCH_DIM, Math.max( 1, x ) );
  return wg;
}

/** 一个节点的 dispatch：kernel 取 impl.entry，缺省回退 op 名。 */
function makeDispatch ( id: NodeId, model: Model, registry: OpRegistry, ir: InferResult ): DispatchItem
{
  const node = model.nodes?.[ id ];
  const contract = node ? registry.get( node.op )?.contract : undefined;
  const kernel = contract?.impl?.entry ?? contract?.op ?? node?.op ?? id;
  return { nodeId: id, kernel, workgroups: workgroupsOf( ir.graph?.nodes?.[ id ] ) };
}

/** 参数量粗估：取几个常见维度 prop 的乘积（确定性，不要求物理精确）。 */
function estimateParams ( model: Model, order: NodeId[] ): number
{
  const env: ExprEnv = { symbols: {} };
  let total = 0;
  for ( const id of order )
  {
    const node = model.nodes?.[ id ];
    if ( !node ) continue;
    let p = 1;
    let used = false;
    for ( const k of [ 'dim', 'vocab', 'hidden', 'outChannels', 'heads', 'patch' ] )
    {
      const v = propNumber( node.props, k, env );
      if ( typeof v === 'number' && v > 1 ) { p *= v; used = true; }
    }
    if ( used ) total += p;
  }
  return total;
}

/** 统计引用某 arena 区段的节点数。 */
function countSegRefs ( order: NodeId[], model: Model, registry: OpRegistry, seg: string ): number
{
  let n = 0;
  for ( const id of order )
  {
    const node = model.nodes?.[ id ];
    if ( !node ) continue;
    const c = registry.get( node.op )?.contract;
    const { reads, writes } = segsOf( c );
    if ( reads.includes( seg ) || writes.includes( seg ) ) n++;
  }
  return n;
}

/**
 * arena 布局：固定 4 段 weight/activation/kv/uniform（06 §3，≤ MAX_BIND_GROUPS）。
 * 字节数用确定性启发式：权重段按参数量×4B，激活/状态段按节点数×1MiB 池。
 */
function buildArena ( order: NodeId[], model: Model, registry: OpRegistry ): ArenaLayout
{
  const params = estimateParams( model, order );
  const weightReaders = countSegRefs( order, model, registry, 'weight' );
  const stateNodes = countSegRefs( order, model, registry, 'state' ) + countSegRefs( order, model, registry, 'kv' );

  const weightBytes = Math.max( params * 4, weightReaders * MIB );
  const activationBytes = order.length * MIB;
  const kvBytes = stateNodes * MIB;

  const segments: ArenaSegment[] = [
    { name: 'weight', kind: 'weight', bytes: weightBytes },
    { name: 'activation', kind: 'activation', bytes: activationBytes },
    { name: 'kv', kind: 'kv', bytes: kvBytes },
    { name: 'uniform', kind: 'uniform', bytes: UNIFORM_BYTES },
  ];
  const totalBytes = segments.reduce( ( s, seg ) => s + seg.bytes, 0 );
  return { segments, totalBytes, bindGroups: Math.min( MAX_BIND_GROUPS, segments.length ) };
}

/** 挑一个满足约束的后端能力（默认取第一个支持需求的）。 */
function pickBackend (
  pool: BackendCapability[],
  arenaBytes: number,
  needF16: boolean,
): BackendCapability
{
  const cands = pool.length > 0 ? pool : [ DEFAULT_BACKEND ];
  const ok = cands.find( ( b ) =>
    b.maxBindGroups >= MAX_BIND_GROUPS &&
    b.maxBufferSize >= arenaBytes &&
    ( !needF16 || b.supportsF16 ) );
  return ok ?? cands[ 0 ];
}

/**
 * 自动插 passBreak（06 §3 / ENG-V2）：
 * 维护"当前 pass 已读/已写区段集合"；新 dispatch 与当前 pass 有任何
 * WAW / WAR / RAW 冲突则开新 pass。**读集合里该节点之前从未写过的区段视为外部
 * 依赖，不触发 break**。
 */
function partitionPasses (
  order: NodeId[],
  model: Model,
  registry: OpRegistry,
  ir: InferResult,
): { passes: PassPlan[]; passBreaks: number }
{
  const passes: PassPlan[] = [];
  let dispatches: DispatchItem[] = [];
  let curReads = new Set<string>();
  let curWrites = new Set<string>();
  const everWritten = new Set<string>();
  let passBreaks = 0;

  const flush = (): void =>
  {
    if ( dispatches.length === 0 ) return;
    passes.push( { index: passes.length, dispatches } );
    dispatches = [];
    curReads = new Set<string>();
    curWrites = new Set<string>();
  };

  for ( const id of order )
  {
    const node = model.nodes?.[ id ];
    const contract = node ? registry.get( node.op )?.contract : undefined;
    const { reads, writes } = segsOf( contract );
    // 只保留"之前被写过"的读：其余是外部依赖（如预加载权重），不参与冲突判定。
    const effReads = reads.filter( ( s ) => everWritten.has( s ) );

    const conflict =
      writes.some( ( s ) => curWrites.has( s ) ) ||   // WAW
      writes.some( ( s ) => curReads.has( s ) ) ||    // WAR
      effReads.some( ( s ) => curWrites.has( s ) );   // RAW

    if ( dispatches.length > 0 && conflict )
    {
      flush();
      passBreaks++;
    }

    dispatches.push( makeDispatch( id, model, registry, ir ) );
    for ( const s of effReads ) curReads.add( s );
    for ( const s of writes ) curWrites.add( s );
    for ( const s of writes ) everWritten.add( s );
  }
  flush();
  return { passes, passBreaks };
}

/** 读所有 RNG 契约节点，按拓扑序分配**互不重叠**的 counter 段（ENG-V5 静态前提）。 */
function buildRng ( order: NodeId[], model: Model, registry: OpRegistry, ir: InferResult ): RNGPlan | null
{
  const ops: RNGPlan['ops'] = [];
  let cursor = 0;
  for ( const id of order )
  {
    const node = model.nodes?.[ id ];
    const contract = node ? registry.get( node.op )?.contract : undefined;
    if ( !contract?.rng ) continue;
    const block = Math.max( 1, firstOutputElems( ir.graph?.nodes?.[ id ] ) );
    ops.push( { nodeId: id, counterScope: contract.rng.counterScope, from: cursor, to: cursor + block } );
    cursor += block;
  }
  return ops.length > 0 ? { ops } : null;
}

/** 第一个输出张量的元素数（符号维视作 1，确定性）。 */
function firstOutputElems ( typed: TypedNode | undefined ): number
{
  const outs = typed?.outputs ?? {};
  const first = Object.values( outs )[ 0 ] ?? null;
  const shape = ( first as { shape?: Dim[] } | null )?.shape;
  if ( !shape ) return 1;
  let elems = 1;
  for ( const d of shape )
    if ( typeof d === 'number' && d > 0 ) elems *= d;
  return Math.max( 1, Math.floor( elems ) );
}

/** 压缩计划：从 `trainer.params` 读（可被测试驱动），缺省 scheme='none'。 */
function readCompress ( model: Model ): CompressPlan
{
  const params = model.trainer?.params ?? {};
  const raw =
    params[ 'compress' ] ?? params[ 'compression' ] ?? params[ 'scheme' ] ??
    params[ 'compressScheme' ] ?? params[ 'compressionScheme' ];
  const scheme: CompressPlan['scheme'] =
    raw === 'topk' || raw === 'sign' || raw === 'quant' || raw === 'none' ? raw : 'none';
  const efRaw = params[ 'errorFeedback' ] ?? params[ 'ef' ] ?? params[ 'useErrorFeedback' ];
  const biased = scheme === 'topk' || scheme === 'sign';
  // ADR-032：有偏压缩默认必须带 EF；显式给 false 才算"未开 EF"（供 COMPRESSION_WITHOUT_EF 检出）。
  const errorFeedback = typeof efRaw === 'boolean' ? efRaw : biased;
  return { scheme, errorFeedback };
}

function jaccard ( a: Set<string>, b: Set<string> ): number
{
  if ( a.size === 0 && b.size === 0 ) return 1;
  let inter = 0;
  for ( const x of a ) if ( b.has( x ) ) inter++;
  const union = a.size + b.size - inter;
  return union === 0 ? 1 : inter / union;
}

// ---------------------------------------------------------------------------
// plan()
// ---------------------------------------------------------------------------

export function plan ( model: Model, ir: InferResult, env: PlanEnv ): ExecutionPlan
{
  ensureBuiltinOps();
  const registry = env.registry ?? builtinRegistry;
  const order = topoOrder( ir );
  const diags: Diag[] = [];

  const strategy: StrategyId = model.trainer?.strategy ?? 'backprop';
  const isBackprop = strategy === 'backprop';

  // ---- 策略感知（ENG-V4） -------------------------------------------------
  const retainActivations = isBackprop;
  const backwardNodes: NodeId[] = isBackprop
    ? order.filter( ( id ) =>
    {
      const caps = ir.graph?.nodes?.[ id ]?.caps ?? [];
      return caps.includes( 'vjp' ) || caps.includes( 'localGrad' );
    } )
    : [];

  // ---- 精度 ---------------------------------------------------------------
  const precisionGlobal: Precision = model.trainer?.schedule?.precision ?? 'fp32';
  const needF16 = precisionGlobal === 'fp16';

  // ---- arena 与后端选择 ---------------------------------------------------
  const arena = buildArena( order, model, registry );
  const chosen = pickBackend( env.backends ?? [], arena.totalBytes, needF16 );

  // 硬约束①：arena.totalBytes ≤ 后端 maxBufferSize（超了报错但继续给计划）。
  if ( arena.totalBytes > chosen.maxBufferSize )
  {
    diags.push( mkDiag( 'error', 'BUDGET_EXCEEDED',
      `arena 需 ${ arena.totalBytes } B，超过后端 ${ chosen.backend } 的 maxBufferSize ${ chosen.maxBufferSize } B`,
      `arena needs ${ arena.totalBytes } B > maxBufferSize ${ chosen.maxBufferSize } B on ${ chosen.backend }` ) );
  }

  // 硬约束②：caps 相容 —— 策略要求的能力位若在 ir.caps.missing 中则冲突。
  const required: Cap[] = model.trainer?.requires ?? STRATEGY_REQUIRES[ strategy ] ?? [];
  const missing = ir.caps?.missing ?? [];
  const conflictCaps = required.filter( ( c ) => missing.includes( c ) );
  if ( conflictCaps.length > 0 )
  {
    diags.push( mkDiag( 'error', 'CAPABILITY_MISSING',
      `策略 ${ strategy } 需要能力位 ${ conflictCaps.join( ', ' ) }，但图内缺失`,
      `strategy ${ strategy } requires ${ conflictCaps.join( ', ' ) } but graph is missing them`,
      { kind: 'switch-strategy', detail: `补上缺失算子的 ${ conflictCaps.join( '/' ) } 实现，或降级到不需要该能力位的策略` } ) );
  }

  // ---- 显存预算 -----------------------------------------------------------
  const activationsBytes = retainActivations ? order.length * MIB : 0;
  const budgetMB = env.budget?.memMB ?? Math.floor( chosen.maxBufferSize / MIB );
  const budgetBytes = budgetMB * MIB;
  if ( arena.totalBytes + activationsBytes > budgetBytes )
  {
    diags.push( mkDiag( 'error', 'BUDGET_EXCEEDED',
      `arena(${ arena.totalBytes } B) + 激活(${ activationsBytes } B) 超过预算 ${ budgetMB } MiB`,
      `arena (${ arena.totalBytes } B) + activations (${ activationsBytes } B) exceed budget ${ budgetMB } MiB`,
      { kind: 'add-annot', detail: '对部分节点开启 recompute 或降低分片/精度以降显存' } ) );
  }

  // ---- 压缩 / 通信 --------------------------------------------------------
  const compress = readCompress( model );
  const syncEvery = model.trainer?.schedule?.syncEvery ?? 1;
  const commPlan: CommPlan['plan'] =
    strategy === 'es' || strategy === 'spsa' ? 'sparse'
      : compress.scheme === 'quant' ? 'quant' : 'full-delta';
  const frameBytes = Math.min( Math.max( MIB, estimateParams( model, order ) * 4 ), chosen.maxBufferSize );
  const comm: CommPlan = { plan: commPlan, syncEvery, maxFrameBytes: frameBytes };

  // 软判据①：有偏压缩却没开 EF（ADR-032）。
  if ( ( compress.scheme === 'topk' || compress.scheme === 'sign' ) && !compress.errorFeedback )
  {
    diags.push( mkDiag( 'warn', 'COMPRESSION_WITHOUT_EF',
      `压缩方案 ${ compress.scheme } 是有偏压缩，但 errorFeedback=false：偏置不随步数衰减`,
      `biased compressor ${ compress.scheme } without errorFeedback: bias will not decay`,
      { kind: 'change-prop', to: 'errorFeedback', detail: '把压缩配置的 errorFeedback 置 true（ADR-032），并把 EF 残差纳入 Snapshot' } ) );
  }

  // ---- 拜占庭预算与房间规模（ADR-029） -----------------------------------
  const trust: TrustPlan = { byzantineBudget: 0, minRoomSize: 0, expectedBias: 0 };
  const f = env.budget?.byzantineF;
  const n = env.budget?.roomSize;
  if ( typeof f === 'number' && typeof n === 'number' && n > 0 )
  {
    trust.byzantineBudget = f / n;
    trust.minRoomSize = f > 0 ? Math.ceil( 2 * f + f / KAPPA_TARGET ) : 0;
    const denom = n - 2 * f;
    trust.expectedBias = denom > 0 ? f / denom : Number.POSITIVE_INFINITY;
    // 软判据②：n ≤ 2f 或所需房间规模 > 实际 ⇒ 不可行（ADR-029）。
    if ( denom <= 0 || trust.minRoomSize > n )
    {
      diags.push( mkDiag( 'warn', 'BYZANTINE_BUDGET_INFEASIBLE',
        `拜占庭预算 f=${ f }, n=${ n }：κ*=f/(n−2f) 邻域不可行（需房间规模 ≥ ${ trust.minRoomSize }）`,
        `byzantine budget f=${ f }, n=${ n }: κ*=f/(n−2f) neighbourhood infeasible (need n ≥ ${ trust.minRoomSize })`,
        { kind: 'resize-probe-pool', detail: `扩大房间规模至 n ≥ ${ trust.minRoomSize }，或下调可容忍的 f` } ) );
    }
  }

  // ---- 房间异构度（ADR-033，软判据③） -----------------------------------
  const peers = env.cluster?.peers ?? [];
  const stratSet = new Set( peers.map( ( p ) => p.strategy ?? '' ) );
  const backendSet = new Set( peers.map( ( p ) => p.backend ) );
  let minJac = 1;
  for ( let i = 0; i < peers.length; i++ )
  {
    for ( let j = i + 1; j < peers.length; j++ )
      minJac = Math.min( minJac, jaccard( new Set( peers[ i ].caps ), new Set( peers[ j ].caps ) ) );
  }
  const heterogeneous =
    peers.length >= 2 &&
    ( ( stratSet.size > 1 && backendSet.size > 1 ) || minJac < 0.5 );
  if ( heterogeneous )
  {
    diags.push( mkDiag( 'warn', 'ROOM_TOO_HETEROGENEOUS',
      `房间异构度过高：策略集=${ stratSet.size }、后端集=${ backendSet.size }、caps 最小杰卡德相似度=${ minJac.toFixed( 3 ) }`,
      `room too heterogeneous: strategies=${ stratSet.size }, backends=${ backendSet.size }, min caps Jaccard=${ minJac.toFixed( 3 ) }`,
      { kind: 'change-shard', detail: '三类建议：① 缩分片（减小 peer 分片粒度）② 加共享池（提高样本/探针共享）③ 先聚类再在簇内鲁棒聚合（ADR-035）' } ) );
  }

  // ---- 逐节点映射 ---------------------------------------------------------
  const backendMap: Record<NodeId, BackendId> = {};
  const strategyMap: Record<NodeId, StrategyId> = {};
  const roleMap: Record<NodeId, string> = {};
  const rootId = ir.graph?.root ?? order[ 0 ];
  for ( const id of order )
  {
    backendMap[ id ] = chosen.backend;
    strategyMap[ id ] = strategy;
    roleMap[ id ] = id === rootId ? 'trainer' : 'shard';
  }
  if ( rootId && roleMap[ rootId ] === undefined ) roleMap[ rootId ] = 'trainer';

  // ---- 后端规则（06 §6.1 Mesh Rule） -------------------------------------
  const backendRules: BackendRule[] = [
    {
      backend: chosen.backend,
      rule: `内核映射：优先 impl.entry，缺省回退 op 名（f16=${ chosen.supportsF16 }, subgroups=${ chosen.supportsSubgroups }）`,
    },
  ];
  // 集群多后端不一致：按本轮约定**不加诊断**，只在 backendRules 记录（无可复用且语义贴切的 DiagCode）。
  for ( const b of backendSet )
    backendRules.push( { backend: b, rule: 'peer 路由：该 backend 的 peer 只执行其被分配的分片' } );

  // ---- pass 划分 / 融合 / RNG --------------------------------------------
  const { passes } = partitionPasses( order, model, registry, ir );
  const fusion = passes.map( ( p ) => p.dispatches.map( ( d ) => d.nodeId ) );
  const rng = buildRng( order, model, registry, ir );
  const recompute = order.filter( ( id ) => model.nodes?.[ id ]?.annot?.recompute === true );
  const shardPlan = model.graph?.annot?.shard?.plan ?? 'auto';
  const peerCount = env.cluster?.peers?.length ?? model.graph?.annot?.shard?.peers ?? 1;

  return {
    backend: backendMap,
    strategy: strategyMap,
    role: roleMap,
    shard: { plan: shardPlan, peers: peerCount },
    precision: { global: precisionGlobal },
    fusion,
    memory: { arena, activationsBytes, budgetMB },
    rng,
    comm,
    compress,
    recompute,
    trust,
    backendRules,
    retainActivations,
    backwardNodes,
    passes,
    diags,
  };
}
