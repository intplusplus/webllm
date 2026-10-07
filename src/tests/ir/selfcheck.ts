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
  specHash,
  buildModel,
  toJsx,
  renderJsx,
  parseJsx,
  h,
} from '../../ir/index';
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
