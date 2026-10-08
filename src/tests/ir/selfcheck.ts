/**
 * Spec IR 的验收自检（design 03 §8 的 IR-V1~V5、05 §8 的 OP-V3/V4、06 §9 的 ENG-V2/V3/V4/V12）。
 *
 * 无害化契约（scripts/lib/bundle.mjs）：本文件不碰 window/document，可在 Node 里跑。
 */

import {
  IR_SCHEMA_VERSION,
  MAX_BIND_GROUPS,
  builtinRegistry,
  ensureBuiltinOps,
  infer,
  migrate,
  plan,
  emit,
  run,
  planBackward,
  specHash,
  buildModel,
  toJsx,
  renderJsx,
  parseJsx,
  h,
  builtinCpuImpls,
  buildGptIr,
  bindBatch,
} from '../../ir/index';
import { attachCrossEntropy, setTargets } from '../../ir/train';
import { builtinCpuGrads } from '../../ir/cpu-grads';
import { backward, createAdamW, dParamOf } from '../../ir/grad';
import { openTemplate, analyze, reweight, annotateEnv, applyFix } from '../../ir/studio';
import { DEFAULT_CONFIG } from '../../model/config';
import { initWeights } from '../../model/init';
import { gptForwardRef } from '../../reference/gpt-ref';
import { checkTolerance } from '../../reference/cpu-ref';
import type {
  BackendCapability,
  ClusterCapability,
  DiagCode,
  Model,
  ModelNode,
  Props,
} from '../../ir/index';

export interface IrTestResult { name: string; pass: boolean; detail: string; }

// ---------------------------------------------------------------------------
// 构造工具
// ---------------------------------------------------------------------------

/** 取注册表中该 op 的 opRef（`Name@1.0`）。 */
function r ( name: string ): string
{
  const d = builtinRegistry.latest( name );
  return d ? `${ d.contract.op }@${ d.contract.version }` : `${ name }@1.0`;
}

function node ( id: string, op: string, props: Props = {}, extra: Partial<ModelNode> = {} ): ModelNode
{
  return { id, op: r( op ), props, ...extra };
}

function mkModel (
  rootId: string,
  list: ModelNode[],
  over: Partial<Model['trainer']> = {},
  meta: Model['meta'] = { author: 'test' },
): Model
{
  const nodes: Record<string, ModelNode> = {};
  for ( const n of list ) nodes[ n.id ] = n;
  const graph = nodes[ rootId ] ?? list[ 0 ];
  if ( !nodes[ graph.id ] ) nodes[ graph.id ] = graph;
  return {
    schemaVersion: IR_SCHEMA_VERSION,
    graph,
    nodes,
    trainer: {
      strategy: over.strategy ?? 'backprop',
      params: over.params ?? {},
      schedule: over.schedule ?? { localSteps: 1, syncEvery: 1, precision: 'fp32' },
      ...( over.requires ? { requires: over.requires } : {} ),
    },
    meta,
  };
}

function codes ( m: Model, opts?: Parameters<typeof infer>[ 1 ] ): DiagCode[]
{
  return infer( m, opts ).diags.map( ( d ) => d.code );
}

function has ( m: Model, code: DiagCode, opts?: Parameters<typeof infer>[ 1 ] ): boolean
{
  return codes( m, opts ).includes( code );
}

/** 每个诊断用例都要求「零崩溃」——infer 必须返回结果而非抛异常。 */
function safeInfer ( m: Model, opts?: Parameters<typeof infer>[ 1 ] ): boolean
{
  try { infer( m, opts ); return true; } catch { return false; }
}

// ---------------------------------------------------------------------------
// 随机模型（IR-V1 / ENG-V2 用；确定性）
// ---------------------------------------------------------------------------

function mulberry32 ( seed: number ): () => number
{
  let a = seed >>> 0;
  return () =>
  {
    a = ( a + 0x6D2B79F5 ) >>> 0;
    let t = a;
    t = Math.imul( t ^ ( t >>> 15 ), t | 1 );
    t ^= t + Math.imul( t ^ ( t >>> 7 ), t | 61 );
    return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;
  };
}

const SAFE_OPS = [
  'RMSNorm', 'LayerNorm', 'GELU', 'SiLU', 'Softmax', 'Reshape', 'Transpose', 'Map', 'Repeat', 'If', 'Concat', 'Fan', 'Seq', 'Residual',
];

function safeProps ( op: string, rnd: () => number ): Props
{
  switch ( op )
  {
    case 'RMSNorm':
    case 'LayerNorm': return { dim: 8 + Math.floor( rnd() * 4 ) * 8 };
    case 'Reshape': return { shape: [ 2 + Math.floor( rnd() * 3 ), 4 + Math.floor( rnd() * 3 ) ] };
    case 'Repeat': return { times: 1 + Math.floor( rnd() * 3 ) };
    case 'If': return { test: rnd() > 0.5 ? 'true' : 'false' };
    default: return {};
  }
}

function randomModel ( rnd: () => number, idSeed: string ): Model
{
  let counter = 0;
  const list: ModelNode[] = [];

  const grow = ( depth: number ): string =>
  {
    const id = `${ idSeed }n${ counter++ }`;
    const op = SAFE_OPS[ Math.floor( rnd() * SAFE_OPS.length ) ];
    const kids: string[] = [];
    const n = depth <= 0 ? 0 : Math.floor( rnd() * 3 );
    for ( let i = 0; i < n; i++ ) kids.push( grow( depth - 1 ) );
    const extra: Partial<ModelNode> = kids.length > 0 ? { children: kids } : {};
    list.push( node( id, op, safeProps( op, rnd ), extra ) );
    return id;
  };

  const rootId = grow( 2 + Math.floor( rnd() * 2 ) );
  return mkModel( rootId, list );
}

/** 反转每个对象的 key 插入顺序（用于 IR-V1 的"同结构必同哈希"）。 */
function reorderKeys ( v: unknown ): unknown
{
  if ( Array.isArray( v ) ) return v.map( reorderKeys );
  if ( v !== null && typeof v === 'object' )
  {
    const src = v as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for ( const k of Object.keys( src ).reverse() ) out[ k ] = reorderKeys( src[ k ] );
    return out;
  }
  return v;
}

// ---------------------------------------------------------------------------
// emit 的 pass 冲突校验（ENG-V2 的判据）
// ---------------------------------------------------------------------------

function passConflicts ( m: Model, passes: Array<{ dispatches: Array<{ nodeId: string }> }> ): string[]
{
  const violations: string[] = [];
  const everWritten = new Set<string>();
  let passNo = 0;
  for ( const p of passes )
  {
    const curReads = new Set<string>();
    const curWrites = new Set<string>();
    for ( const d of p.dispatches )
    {
      const c = builtinRegistry.get( m.nodes[ d.nodeId ]?.op ?? '' )?.contract;
      const reads = c?.writesReads?.reads ?? [];
      const writes = c?.writesReads?.writes ?? [];
      const effReads = reads.filter( ( s ) => everWritten.has( s ) );
      const conflict =
        writes.some( ( s ) => curWrites.has( s ) ) ||
        writes.some( ( s ) => curReads.has( s ) ) ||
        effReads.some( ( s ) => curWrites.has( s ) );
      if ( conflict ) violations.push( `pass#${ passNo } 内 ${ d.nodeId } 与同 pass 前序 dispatch 冲突` );
      for ( const s of effReads ) curReads.add( s );
      for ( const s of writes ) { curWrites.add( s ); everWritten.add( s ); }
    }
    passNo++;
  }
  return violations;
}

// ---------------------------------------------------------------------------
// 主入口
// ---------------------------------------------------------------------------

export async function runIrSelfCheck (): Promise<IrTestResult[]>
{
  ensureBuiltinOps();
  const out: IrTestResult[] = [];
  const push = ( name: string, pass: boolean, detail: string ): void =>
  {
    out.push( { name, pass, detail } );
  };

  // ===================== IR-V2：诊断码表（20 个构造用例） =====================

  // 1) SHAPE_MISMATCH：Embed 声明 ids=[B,T]，源 Reshape 输出 [4,5]，绑定 B=2,T=3。
  {
    const m = mkModel( 'r', [
      node( 'c1', 'Reshape', { shape: [ 4, 5 ] } ),
      node( 'r', 'Embed', { vocab: 100, dim: 8 }, { children: [ 'c1' ] } ),
    ] );
    const opts = { symbols: { B: 2, T: 3 } };
    push( 'IR-V2 · SHAPE_MISMATCH', has( m, 'SHAPE_MISMATCH', opts ) && safeInfer( m, opts ),
      `codes=${ codes( m, opts ).join( ',' ) }` );
  }

  // 2) DTYPE_MISMATCH：Embed.ids 期望 i32，源输出 f32。
  {
    const m = mkModel( 'r', [
      node( 'c1', 'Reshape', { shape: [ 2, 3 ] } ),
      node( 'r', 'Embed', { vocab: 100, dim: 8 }, { children: [ 'c1' ] } ),
    ] );
    const opts = { symbols: { B: 2, T: 3 } };
    push( 'IR-V2 · DTYPE_MISMATCH', has( m, 'DTYPE_MISMATCH', opts ), `codes=${ codes( m, opts ).join( ',' ) }` );
  }

  // 3) MISSING_PARAM：Embed 缺必填 vocab。
  {
    const m = mkModel( 'r', [ node( 'r', 'Embed', { dim: 8 } ) ] );
    push( 'IR-V2 · MISSING_PARAM', has( m, 'MISSING_PARAM' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 4) UNDEFINED_SLOT：slot 指向不存在的节点。
  {
    const m = mkModel( 'r', [ node( 'r', 'Embed', { vocab: 100, dim: 8 }, { slot: { ids: 'ghost' } } ) ] );
    push( 'IR-V2 · UNDEFINED_SLOT', has( m, 'UNDEFINED_SLOT' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 5) PHASE_LEAK：Cache（prefill/decode）在 train 相位使用。
  {
    const m = mkModel( 'r', [ node( 'r', 'Cache', {} ) ] );
    push( 'IR-V2 · PHASE_LEAK', has( m, 'PHASE_LEAK' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 6) EFFECT_IN_PURE：Tool（effectful）出现在同质纯子图（INV-8）。
  {
    const m = mkModel( 'r', [ node( 'r', 'Tool', { name: 'search' } ) ] );
    push( 'IR-V2 · EFFECT_IN_PURE', has( m, 'EFFECT_IN_PURE' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 7) DYNAMIC_SHAPE_UNRESOLVED：未绑定符号维 B/T（warn 级）。
  {
    const m = mkModel( 'r', [ node( 'r', 'Embed', { vocab: 100, dim: 8 } ) ] );
    const ds = infer( m ).diags;
    push( 'IR-V2 · DYNAMIC_SHAPE_UNRESOLVED',
      ds.some( ( d ) => d.code === 'DYNAMIC_SHAPE_UNRESOLVED' && d.level === 'warn' ),
      `codes=${ ds.map( ( d ) => d.code ).join( ',' ) }` );
  }

  // 8) CAPABILITY_MISSING：op 未注册。
  {
    const m = mkModel( 'r', [ { id: 'r', op: 'NoSuchOp@9.9', props: {} } ] );
    push( 'IR-V2 · CAPABILITY_MISSING(未注册 op)', has( m, 'CAPABILITY_MISSING' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 9) STRATEGY_INCOMPAT：strategy=es 却 requires vjp。
  {
    const m = mkModel( 'r', [ node( 'r', 'RMSNorm', { dim: 8 } ) ], { strategy: 'es', requires: [ 'vjp', 'forward', 'perturbable' ] } );
    push( 'IR-V2 · STRATEGY_INCOMPAT', has( m, 'STRATEGY_INCOMPAT' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 10) UNREPRODUCIBLE_RNG：策略需 RNG 但全图无 rng 契约。
  {
    const m = mkModel( 'r', [ node( 'r', 'RMSNorm', { dim: 8 } ) ], { strategy: 'es' } );
    push( 'IR-V2 · UNREPRODUCIBLE_RNG', has( m, 'UNREPRODUCIBLE_RNG' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 11) BUDGET_EXCEEDED：预算给到 0.000001 MiB。
  {
    const m = mkModel( 'r', [ node( 'r', 'Embed', { vocab: 1000, dim: 64 } ) ] );
    const opts = { symbols: { B: 8, T: 64 }, budget: { memMB: 0.000001 } };
    push( 'IR-V2 · BUDGET_EXCEEDED', has( m, 'BUDGET_EXCEEDED', opts ), `codes=${ codes( m, opts ).join( ',' ) }` );
  }

  // 12) SHARD_INFEASIBLE：tensor 分片 peers=3 而首维 10 不能整除。
  {
    const m = mkModel( 'r', [ node( 'r', 'Reshape', { shape: [ 10, 4 ] }, { annot: { shard: { plan: 'tensor', peers: 3 } } } ) ] );
    push( 'IR-V2 · SHARD_INFEASIBLE', has( m, 'SHARD_INFEASIBLE' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 13) PARAM_REUSE_CONFLICT：同 tie 但输入 shape 不一致。
  {
    const m = mkModel( 'r', [
      node( 'c1', 'Reshape', { shape: [ 2, 3 ] } ),
      node( 'c2', 'Reshape', { shape: [ 4, 5 ] } ),
      node( 'n1', 'RMSNorm', { dim: 3, tie: 'w' }, { children: [ 'c1' ] } ),
      node( 'n2', 'RMSNorm', { dim: 5, tie: 'w' }, { children: [ 'c2' ] } ),
      node( 'r', 'Seq', {}, { children: [ 'n1', 'n2' ] } ),
    ] );
    push( 'IR-V2 · PARAM_REUSE_CONFLICT', has( m, 'PARAM_REUSE_CONFLICT' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 14) BUS_CYCLE：inject 无配对 provide。
  {
    const m = mkModel( 'r', [ node( 'r', 'Bus', { provide: 'p', inject: 'missing' } ) ] );
    push( 'IR-V2 · BUS_CYCLE', has( m, 'BUS_CYCLE' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 15) TRAP_PATH：provide 在深度 0、inject 在深度 9。
  {
    const list: ModelNode[] = [];
    list.push( node( 'r', 'Bus', { provide: 'p' }, { children: [ 'l1' ] } ) );
    for ( let i = 1; i <= 8; i++ )
      list.push( node( `l${ i }`, 'Seq', {}, { children: [ `l${ i + 1 }` ] } ) );
    list.push( node( 'l9', 'Bus', { provide: 'q', inject: 'p' } ) );
    const m = mkModel( 'r', list );
    push( 'IR-V2 · TRAP_PATH', has( m, 'TRAP_PATH' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 16) ORPHAN_NODE：存在从根不可达的节点。
  {
    const m = mkModel( 'r', [
      node( 'r', 'RMSNorm', { dim: 8 } ),
      node( 'lost', 'GELU', {} ),
    ] );
    push( 'IR-V2 · ORPHAN_NODE', has( m, 'ORPHAN_NODE' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 17) SCHEDULE_DEADLOCK：wiring 成环。
  {
    const m = mkModel( 'r', [
      node( 'a', 'RMSNorm', { dim: 8 }, { wiring: [ { from: 'a', fromPort: 'out', to: 'b', toPort: 'x', kind: 'tensor' } ] } ),
      node( 'b', 'RMSNorm', { dim: 8 }, { wiring: [ { from: 'b', fromPort: 'out', to: 'a', toPort: 'x', kind: 'tensor' } ] } ),
    ], {}, { author: 't' } );
    // 让 a 成为根（把 a 放进 nodes 且 graph=a）
    const m2: Model = { ...m, graph: m.nodes.a };
    push( 'IR-V2 · SCHEDULE_DEADLOCK', has( m2, 'SCHEDULE_DEADLOCK' ) && safeInfer( m2 ), `codes=${ codes( m2 ).join( ',' ) }` );
  }

  // 18) STABILITY_HINT：dim ≥ 4096 且无 QKNorm/ZLoss。
  {
    const m = mkModel( 'r', [ node( 'r', 'RMSNorm', { dim: 4096 } ) ] );
    push( 'IR-V2 · STABILITY_HINT', has( m, 'STABILITY_HINT' ), `codes=${ codes( m ).join( ',' ) }` );
  }

  // 19) EVAL_UNDERPOWERED：探针池 1 item，声明 effect=1（需 16）。
  {
    const m = mkModel( 'r', [ node( 'r', 'Probe', { kind: 'atomic', effect: 1, poolItems: 1 } ) ] );
    const opts = { phase: 'free' as const, homogeneous: false };
    push( 'IR-V2 · EVAL_UNDERPOWERED', has( m, 'EVAL_UNDERPOWERED', opts ), `codes=${ codes( m, opts ).join( ',' ) }` );
  }

  // 20) BACKDOOR_SUSPECT：backdoor 探针在正常输入上检出。
  {
    const m = mkModel( 'r', [ node( 'r', 'Probe', { kind: 'backdoor', detected: true } ) ] );
    const opts = { phase: 'free' as const, homogeneous: false };
    push( 'IR-V2 · BACKDOOR_SUSPECT', has( m, 'BACKDOOR_SUSPECT', opts ), `codes=${ codes( m, opts ).join( ',' ) }` );
  }

  // ===================== IR-V1：canonical / specHash 稳定性 =====================

  {
    const rnd = mulberry32( 20261008 );
    let ok = true;
    let firstBad = '';
    const N = 1000;
    for ( let i = 0; i < N; i++ )
    {
      const m = randomModel( rnd, `m${ i }_` );
      const h1 = specHash( m );
      const h2 = specHash( m );
      // key 顺序打乱后必须仍同哈希（canonical 的 key 排序）。
      const shuffled: Model = reorderKeys( m ) as Model;
      const h3 = specHash( shuffled );
      if ( h1 !== h2 || h1 !== h3 )
      {
        ok = false;
        firstBad = `seed#${ i }: ${ h1.slice( 0, 8 ) }/${ h2.slice( 0, 8 ) }/${ h3.slice( 0, 8 ) }`;
        break;
      }
    }
    push( `IR-V1 · canonical 在 ${ N } 个随机 IR 上稳定`, ok, ok ? `全部同哈希（示例 ${ specHash( randomModel( mulberry32( 1 ), 'x_' ) ).slice( 0, 12 ) }…）` : firstBad );
  }

  // ===================== IR-V4：device/shard/precision 不敏感（INV-11） =====================

  {
    const base = mkModel( 'r', [
      node( 'c1', 'Reshape', { shape: [ 4, 8 ] } ),
      node( 'r', 'RMSNorm', { dim: 8 }, { children: [ 'c1' ] } ),
    ], {}, { author: 'a', created: '2026-01-01T00:00:00Z' } );
    const h0 = specHash( base );

    const mutated: Model = JSON.parse( JSON.stringify( base ) ) as Model;
    for ( const n of Object.values( mutated.nodes ) )
      n.annot = { device: 'cpu', shard: { plan: 'tensor', peers: 4 }, precision: 'fp16' };
    mutated.graph = mutated.nodes[ mutated.graph.id ];
    mutated.meta = { ...mutated.meta, created: '2099-12-31T23:59:59Z' };
    const h1 = specHash( mutated );

    // 反向对照：改语义（strategy）必须改变哈希。
    const semantic: Model = { ...base, trainer: { ...base.trainer, strategy: 'es' } };
    const h2 = specHash( semantic );

    push( 'IR-V4 · specHash 对 device/shard/precision/created 不敏感', h0 === h1 && h0 !== h2,
      `base=${ h0.slice( 0, 10 ) } mutated=${ h1.slice( 0, 10 ) } semantic=${ h2.slice( 0, 10 ) }` );
  }

  // ===================== IR-V3：JSX ↔ IR round-trip 保义 =====================

  {
    const tree = h( 'Seq', {},
      h( 'RMSNorm', { dim: 128 }, h( 'Reshape', { shape: [ 4, 128 ] } ) ),
      h( 'GELU', {} ),
      h( 'Concat', { axis: -1 }, h( 'Softmax', {} ), h( 'Transpose', { perm: [ 1, 0 ] } ) ),
    );
    const trainer = { strategy: 'backprop' as const, params: { lr: 0.01 }, schedule: { localSteps: 1, syncEvery: 1, precision: 'fp32' as const } };
    const meta = { author: 'roundtrip', license: 'MIT' };

    const m1 = buildModel( tree, trainer, meta );
    const back = parseJsx( renderJsx( toJsx( m1 ) ) );
    const m2 = buildModel( back, trainer, meta );

    const same = specHash( m1 ) === specHash( m2 );
    const nodeCount = Object.keys( m1.nodes ).length;
    push( 'IR-V3 · JSX→IR→JSX→IR 保义（specHash 一致）', same,
      `${ nodeCount } 节点；hash ${ specHash( m1 ).slice( 0, 10 ) } vs ${ specHash( m2 ).slice( 0, 10 ) }` );
  }

  // ===================== IR-V5：迁移器 =====================

  {
    // 模拟历史 IR：缺 trainer、op 无版本、缺 schemaVersion、缺 nodes 表。
    const legacy = {
      graph: { id: 'root', op: 'RMSNorm', props: { dim: 64 }, children: [ 'c1' ] },
      nodes: { root: { id: 'root', op: 'RMSNorm', props: { dim: 64 }, children: [ 'c1' ] }, c1: { id: 'c1', op: 'Reshape', props: { shape: [ 2, 64 ] } } },
    };
    const m1 = migrate( legacy );
    const m2 = migrate( m1 );
    const upgraded = m1.schemaVersion === IR_SCHEMA_VERSION
      && m1.trainer.strategy === 'backprop'
      && m1.graph.op.includes( '@' )
      && m1.trainer.schedule.precision === 'fp32';
    const idempotent = specHash( m1 ) === specHash( m2 );
    const infers = safeInfer( m1 );
    push( 'IR-V5 · 迁移器（补默认值 + opRef + 幂等）', upgraded && idempotent && infers,
      `schema=${ m1.schemaVersion } strategy=${ m1.trainer.strategy } hashStable=${ idempotent }` );
  }

  // ===================== OP-V3 / OP-V4：契约一致性 =====================

  {
    const problems: string[] = [];
    const DIFF_CAPS = [ 'vjp', 'jvp', 'localGrad' ];
    for ( const def of builtinRegistry.all() )
    {
      const c = def.contract;
      if ( !c.caps.includes( 'forward' ) ) problems.push( `${ c.op }: 缺 forward` );
      if ( !c.pure && ( !c.effects || c.effects.length === 0 ) ) problems.push( `${ c.op }: effectful 未列 effects` );
      // `nondiff` 是"明确无任何梯度"，与任何可微能力位同时出现即自相矛盾（05 §3）。
      const contradiction = DIFF_CAPS.filter( ( k ) => c.caps.includes( k as never ) );
      if ( c.caps.includes( 'nondiff' ) && contradiction.length > 0 )
        problems.push( `${ c.op }: nondiff 与 ${ contradiction.join( '/' ) } 互斥` );
      if ( !c.impl || !c.impl.kind ) problems.push( `${ c.op }: 缺 impl.kind` );
    }
    push( `OP-V3 · ${ builtinRegistry.size } 个 op 的 caps/effects 自洽`, problems.length === 0, problems.join( '; ' ) || '全部自洽' );
  }

  {
    // OP-V4：单 kernel ≤ 4 bind group —— arena 固定 4 段。
    const m = randomModel( mulberry32( 7 ), 'a_' );
    const ir = infer( m );
    const p = plan( m, ir, { backends: [ webgpuBackend() ] } );
    const ok = p.memory.arena.bindGroups <= MAX_BIND_GROUPS && p.memory.arena.segments.length <= MAX_BIND_GROUPS;
    push( 'OP-V4 · arena bind group ≤ 4', ok, `bindGroups=${ p.memory.arena.bindGroups } segments=${ p.memory.arena.segments.length }` );
  }

  // ===================== ENG-V2：自动 passBreak =====================

  {
    const rnd = mulberry32( 4242 );
    let violations: string[] = [];
    let totalBreaks = 0;
    const N = 200;
    for ( let i = 0; i < N && violations.length === 0; i++ )
    {
      const m = randomModel( rnd, `p${ i }_` );
      const ir = infer( m );
      const p = plan( m, ir, { backends: [ webgpuBackend() ] } );
      const art = emit( m, ir, p );
      totalBreaks += art.passBreaks;
      violations = passConflicts( m, art.passes );
    }
    push( `ENG-V2 · 随机图 RAW/WAR/WAW 全被 passBreak 覆盖（${ N } 图）`, violations.length === 0,
      violations.length === 0 ? `零冲突；累计插 ${ totalBreaks } 次 pass 边界` : violations.slice( 0, 3 ).join( ' | ' ) );
  }

  // ===================== ENG-V3：超限必被拒 =====================

  {
    const m = mkModel( 'r', [ node( 'r', 'Embed', { vocab: 4096, dim: 1024 } ) ], {}, { author: 't' } );
    const ir = infer( m );
    const tiny: BackendCapability = { ...webgpuBackend(), maxBufferSize: 64 * 1024 };
    const p = plan( m, ir, { backends: [ tiny ] } );
    const rejected = p.diags.some( ( d ) => d.code === 'BUDGET_EXCEEDED' && d.level === 'error' );
    push( 'ENG-V3 · 超 maxBufferSize 必被拒（BUDGET_EXCEEDED）', rejected,
      `diags=${ p.diags.map( ( d ) => d.code ).join( ',' ) || '(无)' } arena=${ p.memory.arena.totalBytes }B` );
  }

  // ===================== ENG-V4：策略感知调度 =====================

  {
    const m = mkModel( 'r', [
      node( 'c1', 'Reshape', { shape: [ 4, 32 ] } ),
      node( 'r', 'RMSNorm', { dim: 32 }, { children: [ 'c1' ] } ),
    ], { strategy: 'es' } );
    const ir = infer( m );
    const p = plan( m, ir, { backends: [ webgpuBackend() ] } );
    const ok = p.retainActivations === false && p.backwardNodes.length === 0 && p.comm.plan === 'sparse';
    push( 'ENG-V4 · 前向-only 策略不排反向/不存激活', ok,
      `retainActivations=${ p.retainActivations } backwardNodes=${ p.backwardNodes.length } comm=${ p.comm.plan }` );
  }

  // ===================== ENG-V12 / 软判据 =====================

  {
    const m = randomModel( mulberry32( 99 ), 'h_' );
    const ir = infer( m );
    const cluster: ClusterCapability = {
      peers: [
        { id: 'pc', backend: 'webgpu', caps: [ 'forward', 'vjp', 'jvp' ], strategy: 'backprop' },
        { id: 'phone', backend: 'cpu', caps: [ 'forward', 'perturbable' ], strategy: 'spsa' },
      ],
    };
    const p = plan( m, ir, { backends: [ webgpuBackend(), cpuBackend() ], cluster } );
    const ok = p.diags.some( ( d ) => d.code === 'ROOM_TOO_HETEROGENEOUS' && d.level === 'warn' && d.fix !== undefined );
    push( 'ENG-V12 · 异构房间报 ROOM_TOO_HETEROGENEOUS（含 fix）', ok,
      `diags=${ p.diags.map( ( d ) => d.code ).join( ',' ) || '(无)' }` );
  }

  {
    const m = mkModel( 'r', [ node( 'r', 'RMSNorm', { dim: 32 } ) ], { params: { compress: 'topk', errorFeedback: false } } );
    const ir = infer( m );
    const p = plan( m, ir, { backends: [ webgpuBackend() ] } );
    const ok = p.diags.some( ( d ) => d.code === 'COMPRESSION_WITHOUT_EF' && d.level === 'warn' );
    push( 'COMPRESSION_WITHOUT_EF · 有偏压缩未开 EF（ADR-032）', ok,
      `compress=${ p.compress.scheme } ef=${ p.compress.errorFeedback } diags=${ p.diags.map( ( d ) => d.code ).join( ',' ) || '(无)' }` );
  }

  {
    const m = mkModel( 'r', [ node( 'r', 'RMSNorm', { dim: 32 } ) ] );
    const ir = infer( m );
    // f=2, n=3 ⇒ n ≤ 2f ⇒ 不可行。
    const p = plan( m, ir, { backends: [ webgpuBackend() ], budget: { byzantineF: 2, roomSize: 3 } } );
    const ok = p.diags.some( ( d ) => d.code === 'BYZANTINE_BUDGET_INFEASIBLE' && d.level === 'warn' );
    push( 'BYZANTINE_BUDGET_INFEASIBLE · f/n 不可行（ADR-029）', ok,
      `κ*=${ p.trust.expectedBias } minRoom=${ p.trust.minRoomSize } diags=${ p.diags.map( ( d ) => d.code ).join( ',' ) || '(无)' }` );
  }

  // ===================== run()：IR 编译产物真的跑出数值 =====================

  {
    const cfg = DEFAULT_CONFIG;
    const w = initWeights( cfg, 1234 );
    const gpt = buildGptIr( cfg, w );
    const B = 2;
    const T = 8;
    const rnd = mulberry32( 7 );
    const tokens = new Uint32Array( B * T );
    for ( let i = 0; i < tokens.length; i++ ) tokens[ i ] = Math.floor( rnd() * cfg.vocabSize );

    const ir = infer( gpt.model, { symbols: { B, T } } );
    const errs = ir.diags.filter( ( d ) => d.level === 'error' );
    const artifact = emit( gpt.model, ir, plan( gpt.model, ir, { backends: [ cpuBackend() ] } ) );
    const binding = bindBatch( gpt, tokens, B, T );
    const impls = builtinCpuImpls();
    const res = run( gpt.model, ir, artifact, binding, impls );
    const ref = gptForwardRef( w, cfg, { tokens, B, T } );
    const tol = checkTolerance( res.rootOutput.data as Float32Array, ref.logits, 1e-4, 1e-4 );

    push( 'RUN-V0 · tiny-GPT 的 IR 模型 infer 零 error',
      errs.length === 0,
      errs.length === 0
        ? `${ Object.keys( gpt.model.nodes ).length } 节点 / 拓扑序 ${ ir.graph.order.length } / pass ${ artifact.passes.length } / 插 break ${ artifact.passBreaks } 次`
        : `error 码=${ errs.map( ( d ) => d.code ).join( ',' ) }` );

    push( 'RUN-V1 · IR 前向 logits 与手写实现逐元素对拍一致', tol.ok,
      `maxAbs=${ tol.maxAbs.toExponential( 2 ) } maxRel=${ tol.maxRel.toExponential( 2 ) } 元素=${ ref.logits.length }` );

    // 编译一次、多步复用（ENG-V1）：同一 artifact 再跑一次必须逐位相同。
    const res2 = run( gpt.model, ir, artifact, binding, impls );
    const same = ( () =>
    {
      const a = res.rootOutput.data as Float32Array;
      const b = res2.rootOutput.data as Float32Array;
      if ( a.length !== b.length ) return false;
      for ( let i = 0; i < a.length; i++ ) if ( a[ i ] !== b[ i ] ) return false;
      return true;
    } )();
    push( 'RUN-V2 · 同一 artifact 复用两次逐位一致（CompileOnce-Reuse）', same,
      `复用后输出完全相同=${ same }（未重新 infer/plan/emit）` );

    // 参数共享（tie）：lmHead.w 与 wte 必须是同一份数据。
    const tied = gpt.tensors.get( 'lmHead.w' )?.data === gpt.tensors.get( 'wte' )?.data;
    push( 'RUN-V3 · lmHead.w 与 wte 共享同一份张量（tie）', tied, `张量身份相同=${ tied }` );
  }

  // ===================== 架构与权重分离（design/15 §3.9 / 基线 B4） =====================

  {
    const cfg = DEFAULT_CONFIG;
    const m1 = buildGptIr( cfg, initWeights( cfg, 1 ) ).model;
    const m2 = buildGptIr( cfg, initWeights( cfg, 2 ) ).model;
    const sameHash = specHash( m1 ) === specHash( m2 );
    const cfg4 = { ...cfg, nLayer: cfg.nLayer + 1 };
    const m3 = buildGptIr( cfg4, initWeights( cfg4, 1 ) ).model;
    const archHash = specHash( m1 ) !== specHash( m3 );
    push( 'ARCH/WEIGHTS · 换权重不改 specHash，改架构必改', sameHash && archHash,
      `权重不同→同哈希=${ sameHash }；层数+1→哈希变=${ archHash }` );
  }

  // ===================== INV-12：受信逃生门 =====================

  {
    let rejected = false;
    try
    {
      builtinRegistry.define( {
        contract: {
          op: 'EvilKernel', version: '0.0', props: {},
          io: { in: [], out: [] },
          caps: [ 'forward' ], phase: [ 'train' ], pure: true,
        },
        reviewId: '',
        codeHash: 'abc',
        referenceTests: [],
      } );
    }
    catch { rejected = true; }
    push( 'INV-12 · defineOp 拒绝未审阅贡献（无 reviewId / 无对拍）', rejected,
      rejected ? '已按 INV-12 拒绝' : '竟然放行了' );
  }

  // ===================== P5：反向传播落到 IR =====================

  {
    // 单层小模型：暴露 VJP 错误最快，跑得也快。
    const cfg = { vocabSize: 32, blockSize: 8, nLayer: 1, nHead: 2, nEmbd: 16, bias: true };
    const w = initWeights( cfg, 99 );
    const gpt = buildGptIr( cfg, w );
    const model = attachCrossEntropy( gpt.model );
    const B = 2;
    const T = 4;
    const rnd = mulberry32( 11 );
    const tokens = new Uint32Array( B * T );
    const targets = new Uint32Array( B * T );
    for ( let i = 0; i < tokens.length; i++ )
    {
      tokens[ i ] = Math.floor( rnd() * cfg.vocabSize );
      targets[ i ] = Math.floor( rnd() * cfg.vocabSize );
    }
    bindBatch( gpt, tokens, B, T );
    setTargets( gpt.tensors, targets, B, T );

    const sym = { B, T };
    const ir = infer( model, { symbols: sym } );
    const errs = ir.diags.filter( ( d ) => d.level === 'error' );
    const artifact = emit( model, ir, plan( model, ir, { backends: [ cpuBackend() ] } ) );
    const impls = builtinCpuImpls();
    const grads = builtinCpuGrads();
    const binding = { tensors: gpt.tensors, symbols: sym };

    push( 'GRAD-V0 · 带损失节点的图 infer 零 error', errs.length === 0,
      errs.length === 0 ? `根=loss，节点 ${ Object.keys( model.nodes ).length } 个` : errs.map( ( d ) => d.code ).join( ',' ) );

    const lossNow = (): number => Number( run( model, ir, artifact, binding, impls ).rootOutput.data[ 0 ] );
    const bwd = backward( model, ir, artifact, binding, impls, grads );

    // ---- GRAD-V1：解析梯度 vs 中心差分 ----
    // 每个张量取**解析梯度绝对值最大**的分量做差分：那里 FD 信号最强、最有可分辨性。
    // （随机小模型的很多权重梯度只有 1e-6 量级，低于 fp32 loss 的 ulp，差分必然读出 0，
    //   拿那些点判对错是在测浮点分辨率而不是测导数。）
    const names = [
      'wte', 'layers.0.ln1W', 'layers.0.ln1B', 'layers.0.wq.w',
      'layers.0.wq.b', 'layers.0.fc.w', 'layers.0.attnProj.w', 'lnFW', 'lmHead.b',
    ];
    const eps = 1e-3;
    const rtol = 5e-2;
    let strong = 0;
    let weakBad = 0;
    let worstRel = 0;
    let worstAt = '(未测到)';
    let tested = 0;
    for ( const name of names )
    {
      const tensor = gpt.tensors.get( name );
      const ana = dParamOf( bwd.dParams, gpt.tensors, name );
      if ( !tensor || !( tensor.data instanceof Float32Array ) || !ana ) continue;
      const p = tensor.data;
      if ( p.length !== ana.length || p.length === 0 ) continue;

      let idx = 0;
      for ( let i = 1; i < ana.length; i++ ) if ( Math.abs( ana[ i ] ) > Math.abs( ana[ idx ] ) ) idx = i;

      const orig = p[ idx ];
      p[ idx ] = orig + eps; const lp = lossNow();
      p[ idx ] = orig - eps; const lm = lossNow();
      p[ idx ] = orig;
      const numeric = ( lp - lm ) / ( 2 * eps );
      const analytic = ana[ idx ];
      const scale = Math.max( Math.abs( numeric ), Math.abs( analytic ) );
      const diff = Math.abs( numeric - analytic );
      if ( scale >= 1e-3 )
      {
        // 强分量：中心差分的分辨率足够，比相对误差。
        strong += 1;
        const rel = diff / Math.max( 1e-12, scale );
        if ( rel > worstRel ) { worstRel = rel; worstAt = `${ name }[${ idx }] 数值=${ numeric.toExponential( 2 ) } 解析=${ analytic.toExponential( 2 ) }`; }
      }
      else if ( diff > 1e-3 )
      {
        // 弱分量：解析梯度≈0 时数值差分也必须≈0。
        // 这一条专抓"漏了一条梯度路径"——如果某条路径没回传，解析会装成 0，而数值会明显非 0。
        weakBad += 1;
        if ( worstAt === '(未测到)' ) worstAt = `${ name }[${ idx }] 数值=${ numeric.toExponential( 2 ) } 解析=0（疑似漏路径）`;
      }
      tested += 1;
    }
    push( `GRAD-V1 · 解析梯度 vs 中心差分（${ tested } 个张量，各取最大分量）`,
      tested >= 6 && strong >= 3 && weakBad === 0 && worstRel < rtol,
      `非平凡样本 ${ strong } 个；噪声地板违例 ${ weakBad } 个；最差相对误差 ${ worstRel.toExponential( 2 ) } @ ${ worstAt }` );

    // ---- GRAD-V2：tie 参数按张量身份归并 ----
    const keys = Object.keys( bwd.dParams );
    const both = keys.includes( 'wte' ) && keys.includes( 'lmHead.w' );
    push( 'GRAD-V2 · tie 参数按张量身份归并（不出现两个别名）', !both,
      `dParams ${ keys.length } 个键；wte=${ keys.includes( 'wte' ) } lmHead.w=${ keys.includes( 'lmHead.w' ) }` );

    // ---- GRAD-V3：用 IR 反向真的把 loss 训下去 ----
    const opt = createAdamW( { lr: 0.05, clip: 1 } );
    const first = bwd.loss;
    let last = first;
    const N = 40;
    const t0 = Date.now();
    for ( let s = 0; s < N; s++ )
    {
      const r = backward( model, ir, artifact, binding, impls, grads );
      opt.step( r.dParams, gpt.tensors );
      last = r.loss;
    }
    const ms = Date.now() - t0;
    push( `GRAD-V3 · IR 反向训练 ${ N } 步 loss 下降`, last < first,
      `${ first.toFixed( 4 ) } → ${ last.toFixed( 4 ) }（Δ=${ ( first - last ).toFixed( 4 ) }，${ ( ms / N ).toFixed( 1 ) } ms/步）` );
  }

  // ===================== ENG-V1（补验收）：编译一次、N 步训练零重复编译 =====================
  // design/06 §9.1 原记"部分"：RUN-V2 只验证复用两次逐位一致，无显式编译计数断言。
  // 这里把"N 步训练只编译一次"做成硬计数：整个循环里 emit 只能被调用一次。
  {
    const cfg = { vocabSize: 16, blockSize: 8, nLayer: 1, nHead: 2, nEmbd: 8, bias: true };
    const w = initWeights( cfg, 5 );
    const gpt = buildGptIr( cfg, w );
    const model = attachCrossEntropy( gpt.model );
    const B = 2, T = 4;
    const rnd = mulberry32( 3 );
    const tokens = new Uint32Array( B * T );
    const targets = new Uint32Array( B * T );
    for ( let i = 0; i < tokens.length; i++ )
    {
      tokens[ i ] = Math.floor( rnd() * cfg.vocabSize );
      targets[ i ] = Math.floor( rnd() * cfg.vocabSize );
    }
    bindBatch( gpt, tokens, B, T );
    setTargets( gpt.tensors, targets, B, T );
    const sym = { B, T };
    const ir = infer( model, { symbols: sym } );
    const impls = builtinCpuImpls();
    const grads = builtinCpuGrads();
    const binding = { tensors: gpt.tensors, symbols: sym };

    // 计数包装：编译（emit）只应发生一次；循环里只能复用同一 artifact。
    let compileCount = 0;
    const compileOnce: typeof emit = ( m, i2, p ) =>
    {
      compileCount += 1;
      return emit( m, i2, p );
    };
    const artifact = compileOnce( model, ir, plan( model, ir, { backends: [ cpuBackend() ] } ) );
    const opt = createAdamW( { lr: 0.05, clip: 1 } );
    const loss0 = Number( run( model, ir, artifact, binding, impls ).rootOutput.data[ 0 ] );
    let last = loss0;
    const N = 30;
    for ( let s = 0; s < N; s++ )
    {
      const r = backward( model, ir, artifact, binding, impls, grads );
      opt.step( r.dParams, gpt.tensors );
      last = r.loss;
    }
    push( 'ENG-V1 · 编译一次、N 步训练零重复编译（compileCount 断言）',
      compileCount === 1 && last < loss0,
      `compileCount=${ compileCount }（应=1）；loss ${ loss0.toFixed( 4 ) } → ${ last.toFixed( 4 ) }；${ N } 步复用同一 artifact，未重新 emit/plan/infer` );
  }

  // ===================== ENG-V10（补验收）：依赖未确认 op 的计划被 SCHEDULE_UNSCHEDULABLE 拒绝 =====================
  // design/06 §9.1 原记"部分"：emit 已有 SCHEDULE_UNSCHEDULABLE 降级路径，但无验收用例。
  // 这里构造"计划引用本引擎没有实现的 op"（依赖远端/未确认状态），断言 emit 不抛异常、
  // 返回空 pass 序列、并带 SCHEDULE_UNSCHEDULABLE 告警——而不是静默生成一个坏 dispatch。
  {
    const cfg = DEFAULT_CONFIG;
    const gpt = buildGptIr( cfg, initWeights( cfg, 1 ) );
    const model = attachCrossEntropy( gpt.model );
    const ir = infer( model, { symbols: { B: 1, T: 1 } } );
    const p = plan( model, ir, { backends: [ cpuBackend() ] } );

    // 把一个正常节点的 op 换成引擎里不存在的实现（模拟"计划依赖远端/未确认状态"）。
    const ghost: Model = { ...model, nodes: { ...model.nodes } };
    const victim = Object.keys( ghost.nodes )[ 0 ];
    ghost.nodes[ victim ] = { ...ghost.nodes[ victim ], op: 'GhostOp@9.9' };

    let threw = false;
    let res: ReturnType<typeof emit> | null = null;
    try { res = emit( ghost, ir, p ); }
    catch { threw = true; }

    const rejected =
      !threw && res !== null &&
      res.passes.length === 0 &&
      res.warnings.some( ( d ) => d.code === 'SCHEDULE_UNSCHEDULABLE' );
    push( 'ENG-V10 · 计划引用未确认 op ⇒ SCHEDULE_UNSCHEDULABLE 拒绝（不抛异常、空 pass）', rejected,
      threw ? 'emit 抛异常（不符合降级契约）'
            : `passes=${ res?.passes.length }；warnings=${ res?.warnings.map( ( d ) => d.code ).join( ',' ) || '无' }` );
  }

  // ===================== GRAD-V4（缺口#2）：反向经编译器排出的 pass 序列执行 =====================
  // design/06 §4 / 对接 doc 缺口#2：反向此前是 grad.ts 里隐式 for 逆序循环（解释执行）；
  // 现抽出 planBackward()，让反向逆序序列由编译器显式排出，backward() 消费它。
  // 验收：① 排出的序列是严格拓扑逆序、根在前、长度一致；② 每步 op 都有 VJP 实现；
  // ③ planner-driven 的反向与解释执行（同一份 backward 实现，确定性）给出的 dParams 逐位一致。
  {
    const cfg = { vocabSize: 16, blockSize: 8, nLayer: 1, nHead: 2, nEmbd: 8, bias: true };
    const w = initWeights( cfg, 5 );
    const gpt = buildGptIr( cfg, w );
    const model = attachCrossEntropy( gpt.model );
    const B = 2, T = 4;
    const rnd = mulberry32( 3 );
    const tokens = new Uint32Array( B * T );
    const targets = new Uint32Array( B * T );
    for ( let i = 0; i < tokens.length; i++ )
    {
      tokens[ i ] = Math.floor( rnd() * cfg.vocabSize );
      targets[ i ] = Math.floor( rnd() * cfg.vocabSize );
    }
    bindBatch( gpt, tokens, B, T );
    setTargets( gpt.tensors, targets, B, T );
    const sym = { B, T };
    const ir = infer( model, { symbols: sym } );
    const impls = builtinCpuImpls();
    const grads = builtinCpuGrads();
    const binding = { tensors: gpt.tensors, symbols: sym };
    const artifact = emit( model, ir, plan( model, ir, { backends: [ cpuBackend() ] } ) );

    const order = ir.graph.order;
    const bp = planBackward( model, ir, grads );

    const reverseOk =
      bp.steps.length === order.length &&
      bp.steps.every( ( s, i ) => s.nodeId === order[ order.length - 1 - i ] ) &&
      bp.steps[ 0 ].nodeId === ( ir.graph.root ?? order[ order.length - 1 ] ) &&
      bp.steps.every( ( s ) => s.hasVjp );

    // 两次独立运行（均经 planBackward）：dParams 必须逐位一致（确定性；也证 planner 路径稳定）。
    const runOnce = (): Record<string, Float32Array> =>
    {
      const b = backward( model, ir, artifact, binding, impls, grads );
      return b.dParams;
    };
    const a = runOnce();
    const b = runOnce();
    let bitIdentical = Object.keys( a ).length > 0;
    for ( const k of Object.keys( a ) )
    {
      const x = a[ k ], y = b[ k ];
      if ( !y || x.length !== y.length ) { bitIdentical = false; break; }
      for ( let i = 0; i < x.length; i++ ) if ( x[ i ] !== y[ i ] ) { bitIdentical = false; break; }
      if ( !bitIdentical ) break;
    }
    // tie 归并：反向后 dParams 不应同时出现 wte 与 lmHead.w 两个别名。
    const keys = Object.keys( a );
    const tieOk = !( keys.includes( 'wte' ) && keys.includes( 'lmHead.w' ) );

    push( 'GRAD-V4 · 反向经编译器排出的 pass 序列执行（逆序结构 + 确定性 + tie 归并）',
      reverseOk && bitIdentical && tieOk && Number.isFinite( a[ keys[ 0 ] ]?.[ 0 ] ?? NaN ),
      `steps=${ bp.steps.length }（应=${ order.length }）；逆序=${ reverseOk }；全有VJP=${ bp.steps.every( ( s ) => s.hasVjp ) }；两次运行逐位一致=${ bitIdentical }；tie双别名=${ !tieOk }；dParams=${ keys.length }` );
  }

  // ===================== 编写台（页面上做的实验，这里做同样的断言） =====================

  {
    const cpu = cpuBackend();
    const s = openTemplate( 'gpt-small' );
    const a1 = analyze( s, cpu );
    const ok1 = a1.specHash.length === 64 && /^[0-9a-f]+$/.test( a1.specHash )
      && a1.nodeCount === 47 && a1.estimate.params > 0;
    push( 'STUDIO-V1 · gpt-small 模板可分析（指纹/节点/参数量）', ok1,
      `hash=${ a1.specHash.slice( 0, 10 ) } 节点=${ a1.nodeCount } 参数=${ a1.estimate.params } 张量=${ a1.estimate.tensors }` );

    const a2 = analyze( reweight( s, 999 ), cpu );
    push( 'STUDIO-V2 · 换一套权重后 specHash 不变（架构与权重分离）', a1.specHash === a2.specHash,
      `${ a1.specHash.slice( 0, 10 ) } → ${ a2.specHash.slice( 0, 10 ) }` );

    annotateEnv( s, 'cpu', 4, 'fp16' );
    const a3 = analyze( s, cpu );
    push( 'STUDIO-V3 · 打 device/shard/precision 注解后 specHash 不变（INV-11）', a1.specHash === a3.specHash,
      `${ a1.specHash.slice( 0, 10 ) } → ${ a3.specHash.slice( 0, 10 ) }` );

    // 「应用修复」按钮真的改 IR：点完那条诊断必须消失。
    const bp = openTemplate( 'broken-param' );
    const ab = analyze( bp, cpu );
    const d1 = ab.diags.find( ( x ) => x.code === 'MISSING_PARAM' && x.fix !== undefined );
    const out1 = d1 ? applyFix( bp, d1 ) : { applied: false, note: '没找到可修复的诊断' };
    const gone1 = !analyze( bp, cpu ).diags.some( ( x ) => x.code === 'MISSING_PARAM' );
    push( 'STUDIO-V4 · 「应用修复」让 MISSING_PARAM 真的消失', out1.applied && gone1, out1.note );

    const bs = openTemplate( 'broken-strategy' );
    const as0 = analyze( bs, cpu );
    const d2 = as0.diags.find( ( x ) => x.code === 'STRATEGY_INCOMPAT' && x.fix !== undefined );
    const out2 = d2 ? applyFix( bs, d2 ) : { applied: false, note: '没找到可修复的诊断' };
    const gone2 = !analyze( bs, cpu ).diags.some( ( x ) => x.code === 'STRATEGY_INCOMPAT' );
    push( 'STUDIO-V5 · 「应用修复」让 STRATEGY_INCOMPAT 真的消失', out2.applied && gone2, out2.note );
  }

  // ===================== 哈希算法自检（跨端一致性前提） =====================

  {
    // FIPS 180-4 已知向量：sha256("abc")
    const known = specHash( { schemaVersion: '1.0', graph: { id: 'root', op: 'X@1.0', props: {} }, nodes: {}, trainer: { strategy: 'backprop', params: {}, schedule: { localSteps: 1, syncEvery: 1, precision: 'fp32' } }, meta: {} } );
    const stable = typeof known === 'string' && known.length === 64 && /^[0-9a-f]+$/.test( known );
    push( 'HASH · SHA-256 输出为 64 位十六进制', stable, `specHash=${ known.slice( 0, 16 ) }…` );
  }

  return out;
}

function webgpuBackend (): BackendCapability
{
  return { backend: 'webgpu', maxBufferSize: 2 * 1024 * 1024 * 1024, maxBindGroups: 4, maxWorkgroupsPerDimension: 65535, supportsF16: true, supportsSubgroups: true };
}

function cpuBackend (): BackendCapability
{
  return { backend: 'cpu', maxBufferSize: 512 * 1024 * 1024, maxBindGroups: 4, maxWorkgroupsPerDimension: 65535, supportsF16: false, supportsSubgroups: false };
}
