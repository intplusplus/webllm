/**
 * infer()：Spec IR 的静态类型分析与诊断（design/03-IR与类型系统.md §5）。
 *
 * 输入 Model → 输出 InferResult（TypedGraph + diags + caps 摘要）。它是**纯静态**的：
 *   - 只依赖注册表里的 op 契约与 shape 函数（不进 specHash，03 §2.1）；
 *   - 全程确定性（无时钟/随机源），同一模型必得同一诊断序列；
 *   - **绝不抛异常**：单节点推导失败要 catch 成诊断后继续跑完其余节点（03 §5 验收线）。
 *
 * 拓扑方向约定：数据从**子/`from`**流向**父/`to`**（03 §2「from 是子、to 是父」）。
 * 因此 `order` 是**依赖优先**序 —— 每个节点的输入来源（children / wiring.from）先于它
 * 被推导，这样解析输入类型时源节点的 `outputs` 一定已经就绪。
 */

import type {
  AutoFix,
  Cap,
  CapsSummary,
  Diag,
  DiagCode,
  DiagLevel,
  Dim,
  DType,
  Edge,
  Effects,
  InferResult,
  Model,
  ModelNode,
  NodeId,
  OpRef,
  Phase,
  PortType,
  StrategyId,
  TensorType,
  TypedGraph,
  TypedNode,
} from './types';
import {
  STRATEGIES_NEEDING_RNG,
  STRATEGY_REQUIRES,
  STABILITY_DEPTH_THRESHOLD,
  STABILITY_WIDTH_THRESHOLD,
  isTensorType,
} from './types';
import type { OpContract, OpDefinition, OpPortSpec, OpRegistry } from './op';
import { builtinRegistry } from './op';
import { ensureBuiltinOps } from './ops';
import type { ExprEnv } from './expr';
import { propBool, propNumber, propString } from './expr';

// ---------------------------------------------------------------------------
// 公开选项
// ---------------------------------------------------------------------------

export interface InferOptions
{
  registry?: OpRegistry;
  /** 分析的相位（默认 'train'），用于 PHASE_LEAK。 */
  phase?: Phase;
  /** 符号维绑定（{ B: 4, T: 128 }）；未绑定则报 DYNAMIC_SHAPE_UNRESOLVED。 */
  symbols?: Record<string, number | undefined>;
  /** plan() 给定的预算（用于 BUDGET_EXCEEDED）。 */
  budget?: { memMB?: number; flops?: number };
  /** 同质档（默认 true）：effectful 节点出现在纯子图里即 EFFECT_IN_PURE（INV-8）。 */
  homogeneous?: boolean;
}

// ---------------------------------------------------------------------------
// 常量：声明顺序即稳定排序依据（03 §3.2 / §2 的联合类型顺序）
// ---------------------------------------------------------------------------

const CAP_ORDER: Cap[] = [
  'forward', 'vjp', 'jvp', 'logprob', 'localGrad', 'perturbable', 'invertible', 'nondiff',
];
const PHASE_ORDER: Phase[] = [ 'train', 'prefill', 'decode', 'sample', 'rollout', 'nudge', 'free' ];

/** DType → 字节宽（预算粗估用；确定性，不依赖平台）。 */
const DTYPE_BYTES: Record<DType, number> = {
  f32: 4, f16: 2, bf16: 2, i32: 4, i64: 8, u32: 4,
};

const MIB = 1024 * 1024;
/** INV-21：原子能力差的统计功效系数（所需池 item 数 = ceil(16 / effect²)）。 */
const INV21_COEF = 16;
/** TRAP_PATH：Bus provide→inject 在 children 树上的最大可接受深度差。 */
const TRAP_PATH_DEPTH = 8;

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------

interface DiagInit
{
  level: DiagLevel;
  code: DiagCode;
  message: string;
  messageEn: string;
  node?: NodeId;
  port?: string;
  fix?: AutoFix;
}

function mkDiag ( init: DiagInit ): Diag
{
  const d: Diag = {
    level: init.level,
    code: init.code,
    message: init.message,
    messageEn: init.messageEn,
  };
  if ( init.node !== undefined ) d.node = init.node;
  if ( init.port !== undefined ) d.port = init.port;
  if ( init.fix !== undefined ) d.fix = init.fix;
  return d;
}

/** 从 `Name@version` 取回不带版本的 op 名。 */
function opNameOf ( ref: OpRef ): string
{
  const at = ref.lastIndexOf( '@' );
  return at >= 0 ? ref.slice( 0, at ) : ref;
}

/** 归一化后的端口：名字 + 可选的声明式 dtype/shape。 */
interface NormPort
{
  name: string;
  dtype?: DType;
  shape?: Dim[];
}

/**
 * 归一化 `io.in` / `io.out`：字符串端口用自身或按位名（`in0`/`out0`），
 * 无名对象端口同规则。
 */
function normPorts ( list: Array<OpPortSpec | string> | undefined, prefix: string ): NormPort[]
{
  const out: NormPort[] = [];
  ( list ?? [] ).forEach( ( s, i ) =>
  {
    if ( typeof s === 'string' )
    {
      out.push( { name: s.trim() !== '' ? s : `${ prefix }${ i }` } );
      return;
    }
    const name = s.name && s.name.trim() !== '' ? s.name : `${ prefix }${ i }`;
    const p: NormPort = { name };
    if ( s.dtype !== undefined ) p.dtype = s.dtype;
    if ( s.shape !== undefined ) p.shape = s.shape;
    out.push( p );
  } );
  return out;
}

/** 第 k 个输入端口是否允许"位置兜底"（与 exec() 的 `inPortPositional` 同语义，W3）。 */
function inPortPositionalFor ( contract: OpContract | undefined, k: number ): boolean
{
  const s = contract?.io?.in?.[ k ];
  if ( s && typeof s === 'object' ) return s.positional !== false;
  return true;
}

function toPhaseArray ( p: Phase | Phase[] | undefined ): Phase[]
{
  if ( p === undefined ) return [];
  return Array.isArray( p ) ? [ ...p ] : [ p ];
}

/**
 * 把 shape 解析为**具体数字**：任一维是未绑定符号则返回 undefined（用于 shape 比较）。
 * 只认 `opts.symbols` 里的绑定，避免把 `dim` 这类 prop 派生符号误判为已知。
 */
function concreteShape ( dims: Dim[] | undefined, symbols: Record<string, number | undefined> ): number[] | undefined
{
  if ( !dims ) return undefined;
  const out: number[] = [];
  for ( const d of dims )
  {
    if ( typeof d === 'number' ) { out.push( d ); continue; }
    const v = symbols[ d ];
    if ( v === undefined ) return undefined;
    out.push( v );
  }
  return out;
}

function shapeEq ( a: number[], b: number[] ): boolean
{
  if ( a.length !== b.length ) return false;
  for ( let i = 0; i < a.length; i++ ) if ( a[ i ] !== b[ i ] ) return false;
  return true;
}

/** 元素数粗估：符号维取绑定值，未绑定回退 1（确定性、可复现）。 */
function elemCount ( dims: Dim[], symbols: Record<string, number | undefined> ): number
{
  let p = 1;
  for ( const d of dims )
  {
    const n = typeof d === 'number' ? d : ( symbols[ d ] ?? 1 );
    p *= n;
  }
  return p;
}

function asTensorPort ( p: PortType | null | undefined ): TensorType | undefined
{
  if ( !p ) return undefined;
  return isTensorType( p ) ? p : undefined;
}

function emptyCaps (): CapsSummary
{
  return { all: [], missing: [], effectfulNodes: [], nondiffNodes: [], phases: [] };
}

// ---------------------------------------------------------------------------
// infer()
// ---------------------------------------------------------------------------

export function infer ( model: Model, opts: InferOptions = {} ): InferResult
{
  try
  {
    return inferGraph( model, opts );
  }
  catch ( err )
  {
    // 兜底：任何意外都降级为一条 warn，绝不把异常抛给调用方（03 §5 验收线）。
    const root = model?.graph?.id ?? 'root';
    return {
      graph: { root, order: [], nodes: {}, children: {}, incoming: {} },
      diags: [ mkDiag( {
        level: 'warn',
        code: 'DYNAMIC_SHAPE_UNRESOLVED',
        message: `infer 期间发生意外，已降级为空图：${ String( err ) }`,
        messageEn: `unexpected error during infer; degraded to empty graph: ${ String( err ) }`,
      } ) ],
      caps: emptyCaps(),
    };
  }
}

function inferGraph ( model: Model, opts: InferOptions ): InferResult
{
  ensureBuiltinOps();
  const registry: OpRegistry = opts.registry ?? builtinRegistry;
  const phaseSel: Phase = opts.phase ?? 'train';
  const symbols: Record<string, number | undefined> = opts.symbols ?? {};
  const homogeneous = opts.homogeneous !== false;
  const strategy: StrategyId = model.trainer?.strategy ?? 'backprop';

  const diags: Diag[] = [];
  const rootId: NodeId = model.graph?.id ?? 'root';

  // ---- 节点宇宙：model.nodes + 根（根可能只以 model.graph 形式给出） --------
  const table: Record<NodeId, ModelNode> = { ...( model.nodes ?? {} ) };
  if ( model.graph && !table[ rootId ] ) table[ rootId ] = model.graph;
  const ids = Object.keys( table );

  const envFor = ( node: ModelNode ): ExprEnv =>
    ( { symbols, props: node.props, meta: model.meta as unknown as Record<string, unknown> } );

  const contractOf = ( id: NodeId ): OpContract | undefined =>
    table[ id ] ? registry.get( table[ id ].op )?.contract : undefined;

  const sourceOut0 = ( srcId: NodeId | undefined ): string =>
  {
    if ( srcId === undefined || !table[ srcId ] ) return 'out';
    const c = contractOf( srcId );
    return c ? ( normPorts( c.io?.out, 'out' )[ 0 ]?.name ?? 'out' ) : 'out';
  };

  // ---- 收集全部 wiring 边，建立反向索引（to → Edge[]，去重） ---------------
  const incoming: Record<NodeId, Edge[]> = {};
  for ( const id of ids ) incoming[ id ] = [];
  const seenEdge = new Set<string>();
  const allEdges: Edge[] = [];
  for ( const id of ids )
    for ( const e of table[ id ].wiring ?? [] )
    {
      const key = `${ e.from }|${ e.fromPort }|${ e.to }|${ e.toPort }|${ e.kind }`;
      if ( seenEdge.has( key ) ) continue;
      seenEdge.add( key );
      allEdges.push( e );
    }
  for ( const e of allEdges )
  {
    if ( !incoming[ e.to ] ) incoming[ e.to ] = [];
    incoming[ e.to ].push( e );
  }

  // ---- 依赖关系：children ∪ wiring.from ∪ slot（from 是子、to 是父） -------
  // slot 也是命名输入来源，必须计入依赖，否则消费者会早于其插槽源被推导（输入成 null）。
  const upstreamIds = ( id: NodeId ): NodeId[] =>
  {
    const n = table[ id ];
    if ( !n ) return [];
    const out: NodeId[] = [];
    for ( const c of n.children ?? [] ) if ( table[ c ] ) out.push( c );
    for ( const e of incoming[ id ] ?? [] ) if ( table[ e.from ] ) out.push( e.from );
    if ( n.slot ) for ( const k of Object.keys( n.slot ) ) if ( table[ n.slot[ k ] ] ) out.push( n.slot[ k ] );
    return out;
  };

  const deps: Record<NodeId, Set<NodeId>> = {};
  for ( const id of ids ) deps[ id ] = new Set<NodeId>( upstreamIds( id ) );

  // ---- 拓扑排序（Kahn，依赖优先；发现环 ⇒ SCHEDULE_DEADLOCK） --------------
  const indeg: Record<NodeId, number> = {};
  const dependents: Record<NodeId, NodeId[]> = {};
  for ( const id of ids ) { indeg[ id ] = 0; dependents[ id ] = []; }
  for ( const id of ids )
    for ( const d of deps[ id ] )
    {
      indeg[ id ]++;
      dependents[ d ].push( id );
    }

  const order: NodeId[] = [];
  const ready = ids.filter( ( id ) => indeg[ id ] === 0 ).sort();
  while ( ready.length > 0 )
  {
    const id = ready.shift()!;
    order.push( id );
    const newly: NodeId[] = [];
    for ( const m of dependents[ id ] )
    {
      indeg[ m ]--;
      if ( indeg[ m ] === 0 ) newly.push( m );
    }
    if ( newly.length > 0 ) { ready.push( ...newly ); ready.sort(); }
  }

  if ( order.length < ids.length )
  {
    const stuck = ids.filter( ( id ) => !order.includes( id ) ).sort();
    diags.push( mkDiag( {
      level: 'error',
      code: 'SCHEDULE_DEADLOCK',
      node: stuck[ 0 ],
      message: `图中存在循环依赖，调度死锁：涉及节点 ${ stuck.join( ', ' ) }`,
      messageEn: `cyclic dependency detected; schedule deadlock among: ${ stuck.join( ', ' ) }`,
      fix: { kind: 're-wire', at: stuck[ 0 ], detail: '打断环：让某个节点不再（间接）依赖自身' },
    } ) );
    // 环上节点仍要参与类型推导，追加到末尾（保持确定性）。
    for ( const id of stuck ) order.push( id );
  }

  // ---- 可达性（从根沿 children + 反向边闭包）⇒ ORPHAN_NODE ----------------
  const reachable = new Set<NodeId>();
  {
    const stack: NodeId[] = [ rootId ];
    while ( stack.length > 0 )
    {
      const id = stack.pop()!;
      if ( reachable.has( id ) ) continue;
      reachable.add( id );
      for ( const u of upstreamIds( id ) ) stack.push( u );
    }
  }

  // ---- 逐节点类型推导 -----------------------------------------------------
  const typed: Record<NodeId, TypedNode> = {};
  const childMap: Record<NodeId, NodeId[]> = {};
  for ( const id of ids ) childMap[ id ] = [ ...( table[ id ].children ?? [] ) ];

  for ( const id of order )
  {
    const node = table[ id ];
    if ( !node ) continue;
    try
    {
      typed[ id ] = typeOne( id, node );
    }
    catch ( err )
    {
      // 单节点失败不阻塞其余节点：转成 warn，输出留空。
      diags.push( mkDiag( {
        level: 'warn',
        code: 'DYNAMIC_SHAPE_UNRESOLVED',
        node: id,
        message: `节点 ${ id } 类型推导失败：${ String( err ) }`,
        messageEn: `type inference failed for node ${ id }: ${ String( err ) }`,
      } ) );
      typed[ id ] = {
        node, op: node.op, inputs: {}, outputs: {},
        caps: [], effects: 'pure', pure: true, phase: [],
      };
    }
  }

  // ---- 节点级推导闭包 -----------------------------------------------------
  function typeOne ( id: NodeId, node: ModelNode ): TypedNode
  {
    const def: OpDefinition | undefined = registry.get( node.op );
    const contract: OpContract | undefined = def?.contract;
    const props = node.props ?? {};
    const caps: Cap[] = contract ? [ ...contract.caps ] : [];
    const effectful = contract ? contract.pure === false : false;
    const effects: Effects = effectful ? 'effectful' : 'pure';

    // (1) op 未注册 ⇒ CAPABILITY_MISSING（a）。
    if ( !contract )
    {
      diags.push( mkDiag( {
        level: 'error',
        code: 'CAPABILITY_MISSING',
        node: id,
        message: `op 未注册：${ node.op }（opRef 在注册表中查不到）`,
        messageEn: `op not registered: ${ node.op } (opRef not found in registry)`,
        fix: { kind: 'switch-strategy', at: id, detail: '改为注册表中的合法 opRef，或先注册该 op 契约' },
      } ) );
    }

    // (2) MISSING_PARAM：契约声明 required 而节点未给。
    if ( contract )
    {
      for ( const key of Object.keys( contract.props ) )
      {
        if ( contract.props[ key ].required === true && props[ key ] === undefined )
        {
          diags.push( mkDiag( {
            level: 'error',
            code: 'MISSING_PARAM',
            node: id,
            message: `节点 ${ id } 缺少必填参数 ${ key }`,
            messageEn: `node ${ id } missing required prop ${ key }`,
            fix: { kind: 'change-prop', to: key, at: id, detail: `为 ${ key } 补给一个合法值` },
          } ) );
        }
      }
    }

    // (3) 解析输入（slot > wiring > children 位置），并做边类型检查。
    const inPorts = contract ? normPorts( contract.io?.in, 'in' ) : [];
    const inputs: Record<string, PortType | null> = {};
    const children = node.children ?? [];

    const pushUndefinedSlot = ( port: string, srcId: NodeId ): void =>
    {
      diags.push( mkDiag( {
        level: 'error',
        code: 'UNDEFINED_SLOT',
        node: id,
        port,
        message: `输入端口 ${ port } 引用了不存在的节点 ${ srcId }`,
        messageEn: `input port ${ port } references undefined node ${ srcId }`,
        fix: { kind: 're-wire', at: id, detail: `把端口 ${ port } 接到 model.nodes 中已存在的节点` },
      } ) );
    };

    const resolveSource = (
      portName: string,
      srcId: NodeId | undefined,
      outPort: string,
    ): PortType | null =>
    {
      if ( srcId === undefined ) return null;
      if ( !table[ srcId ] ) { pushUndefinedSlot( portName, srcId ); return null; }
      const src = typed[ srcId ];
      return src ? ( src.outputs[ outPort ] ?? null ) : null;
    };

    for ( let k = 0; k < inPorts.length; k++ )
    {
      const port = inPorts[ k ];

      // a) 命名插槽
      if ( node.slot && Object.prototype.hasOwnProperty.call( node.slot, port.name ) )
      {
        const srcId = node.slot[ port.name ];
        inputs[ port.name ] = resolveSource( port.name, srcId, sourceOut0( srcId ) );
        checkEdgeTypes( id, port, srcId, inputs[ port.name ] );
        continue;
      }
      // b) wiring：toPort 匹配
      const edge = ( incoming[ id ] ?? [] ).find( ( e ) => e.toPort === port.name );
      if ( edge )
      {
        inputs[ port.name ] = resolveSource( port.name, edge.from, edge.fromPort );
        checkEdgeTypes( id, port, edge.from, inputs[ port.name ] );
        continue;
      }
      // c) children 位置对应：第 k 个子对第 k 个输入端口。
      //    但若该端口声明 `positional: false`（如 Residual.x），禁用兜底——缺显式来源时
      //    保持 null，由实现层清晰报错，而非静默复用 child（W3，须与 exec() 一致）。
      if ( k < children.length && inPortPositionalFor( contract, k ) )
      {
        const srcId = children[ k ];
        inputs[ port.name ] = resolveSource( port.name, srcId, sourceOut0( srcId ) );
        checkEdgeTypes( id, port, srcId, inputs[ port.name ] );
        continue;
      }
      inputs[ port.name ] = null;
    }

    // slot 里 io.in 未列出的命名端口也要进 inputs。
    if ( node.slot )
    {
      for ( const key of Object.keys( node.slot ) )
      {
        if ( Object.prototype.hasOwnProperty.call( inputs, key ) ) continue;
        const srcId = node.slot[ key ];
        inputs[ key ] = resolveSource( key, srcId, sourceOut0( srcId ) );
      }
    }

    // (4) 输出：shape 函数（缺省时按端口声明透传）+ 未绑定符号维检查。
    const outPorts = contract ? normPorts( contract.io?.out, 'out' ) : [];
    const outputs: Record<string, PortType | null> = {};
    for ( const p of outPorts ) outputs[ p.name ] = null;

    if ( def?.shape )
    {
      try
      {
        const r = def.shape( inputs, { props, symbols, nodeId: id } );
        for ( const key of Object.keys( r ) ) outputs[ key ] = r[ key ] ?? null;
      }
      catch ( err )
      {
        diags.push( mkDiag( {
          level: 'warn',
          code: 'DYNAMIC_SHAPE_UNRESOLVED',
          node: id,
          message: `节点 ${ id } 的 shape 函数求值失败：${ String( err ) }`,
          messageEn: `shape function of node ${ id } threw: ${ String( err ) }`,
        } ) );
      }
    }
    else
    {
      for ( const p of outPorts )
        if ( p.shape !== undefined ) outputs[ p.name ] = { shape: p.shape, dtype: p.dtype ?? 'f32' };
    }

    const unresolved: string[] = [];
    for ( const key of Object.keys( outputs ) )
    {
      const t = asTensorPort( outputs[ key ] );
      if ( !t ) continue;
      for ( const d of t.shape )
        if ( typeof d === 'string' && symbols[ d ] === undefined && !unresolved.includes( d ) )
          unresolved.push( d );
    }
    if ( unresolved.length > 0 )
    {
      diags.push( mkDiag( {
        level: 'warn',
        code: 'DYNAMIC_SHAPE_UNRESOLVED',
        node: id,
        message: `节点 ${ id } 的输出含未绑定符号维 ${ unresolved.join( ', ' ) }，退化为运行时检查`,
        messageEn: `node ${ id } output has unbound symbolic dims ${ unresolved.join( ', ' ) }; defer to runtime check`,
      } ) );
    }

    // (5) PHASE_LEAK：有效相位集合（annot.phase 优先，缺省用 contract.phase）不含当前相位。
    const declared = node.annot?.phase !== undefined
      ? toPhaseArray( node.annot.phase )
      : ( contract ? [ ...contract.phase ] : [] );
    if ( declared.length > 0 && !declared.includes( phaseSel ) )
    {
      diags.push( mkDiag( {
        level: 'error',
        code: 'PHASE_LEAK',
        node: id,
        message: `节点 ${ id } 在相位 ${ phaseSel } 使用，但其声明相位为 [${ declared.join( ', ' ) }]`,
        messageEn: `node ${ id } used in phase ${ phaseSel } but declared for [${ declared.join( ', ' ) }]`,
        fix: { kind: 'add-annot', at: id, detail: `在 annot.phase 中补上 ${ phaseSel }，或改用匹配相位的组件` },
      } ) );
    }

    // (6) EFFECT_IN_PURE：effectful 节点出现在同质纯子图（INV-8）。
    if ( effectful && homogeneous )
    {
      diags.push( mkDiag( {
        level: 'error',
        code: 'EFFECT_IN_PURE',
        node: id,
        message: `effectful 节点 ${ id } 出现在同质纯子图中（INV-8：禁入）`,
        messageEn: `effectful node ${ id } appears in a homogeneous pure subgraph (INV-8)`,
        fix: { kind: 'switch-strategy', at: id, detail: '把该节点移出纯子图，或切换到允许副作用的策略/子图' },
      } ) );
    }

    // (7) SHARD_INFEASIBLE：tensor 分片且首个具体数字维不能被 peers 整除。
    const shard = node.annot?.shard;
    if ( shard?.plan === 'tensor' )
    {
      const peers = shard.peers ?? 1;
      if ( peers > 1 )
      {
        const firstDim = firstConcreteDim( outputs );
        if ( firstDim !== undefined && firstDim % peers !== 0 )
        {
          diags.push( mkDiag( {
            level: 'error',
            code: 'SHARD_INFEASIBLE',
            node: id,
            message: `节点 ${ id } 声明 tensor 分片 peers=${ peers }，但首维 ${ firstDim } 不能被整除`,
            messageEn: `node ${ id } requests tensor shard with peers=${ peers }, but leading dim ${ firstDim } is not divisible`,
            fix: { kind: 'change-shard', at: id, detail: `改为 layer/auto 方案，或把 peers 调整到整除 ${ firstDim } 的值` },
          } ) );
        }
      }
    }

    // (8) CAPABILITY_MISSING（b）：策略要求的 caps 该节点不具备。
    if ( contract )
    {
      const required: Cap[] = [ ...new Set( model.trainer?.requires ?? STRATEGY_REQUIRES[ strategy ] ?? [] ) ];
      const lack = required.filter( ( c ) => !caps.includes( c ) );
      if ( lack.length > 0 )
      {
        diags.push( mkDiag( {
          level: 'error',
          code: 'CAPABILITY_MISSING',
          node: id,
          message: `策略 ${ strategy } 需要能力位 [${ lack.join( ', ' ) }]，但节点 ${ id } 不具备`,
          messageEn: `strategy ${ strategy } requires caps [${ lack.join( ', ' ) }] missing on node ${ id }`,
          fix: { kind: 'switch-strategy', at: id, detail: '换用具备该能力位的算子，或降级到不需要它的策略' },
        } ) );
      }
      // backprop 特例（03 §3.2）：非 effectful 的计算节点必须含 vjp 或 localGrad。
      if ( strategy === 'backprop' && !effectful && !( caps.includes( 'vjp' ) || caps.includes( 'localGrad' ) ) )
      {
        diags.push( mkDiag( {
          level: 'error',
          code: 'CAPABILITY_MISSING',
          node: id,
          message: `backprop 要求损失路径上每个非 effectful 节点含 vjp 或 localGrad，但节点 ${ id } 两者皆无`,
          messageEn: `backprop needs vjp or localGrad on every non-effectful node, but node ${ id } has neither`,
          fix: { kind: 'switch-strategy', at: id, detail: '换用可反传的算子，或改用不依赖反传的策略' },
        } ) );
      }
    }

    // (9) Probe：EVAL_UNDERPOWERED / BACKDOOR_SUSPECT。
    if ( opNameOf( node.op ) === 'Probe' )
    {
      const env = envFor( node );
      const kind = propString( props, 'kind', env, 'atomic' ) ?? 'atomic';
      if ( kind === 'atomic' )
      {
        const effect = propNumber( props, 'effect', env, 0 ) ?? 0;
        if ( effect > 0 )
        {
          const poolItems = propNumber( props, 'poolItems', env, 0 ) ?? 0;
          const need = Math.ceil( INV21_COEF / ( effect * effect ) );
          if ( poolItems < need )
          {
            diags.push( mkDiag( {
              level: 'warn',
              code: 'EVAL_UNDERPOWERED',
              node: id,
              message: `Probe 池 item 数 ${ poolItems } < 检出 effect=${ effect } 所需 ${ need }（INV-21）`,
              messageEn: `Probe pool ${ poolItems } items < required ${ need } for effect=${ effect } (INV-21)`,
              fix: { kind: 'resize-probe-pool', at: id, detail: `把 poolItems 扩到 ≥ ${ need }，或缩小声明的 effect` },
            } ) );
          }
        }
      }
      else if ( kind === 'backdoor' )
      {
        const detected = propBool( props, 'detected', env, false );
        if ( detected )
        {
          diags.push( mkDiag( {
            level: 'error',
            code: 'BACKDOOR_SUSPECT',
            node: id,
            message: `backdoor 探针 ${ id } 在正常输入上检出触发器响应`,
            messageEn: `backdoor probe ${ id } triggered on clean inputs`,
            fix: { kind: 'quarantine', at: id, detail: '隔离该节点（09 §6），并追查触发器来源' },
          } ) );
        }
      }
    }

    return {
      node,
      op: node.op,
      inputs,
      outputs,
      caps,
      effects,
      pure: !effectful,
      phase: declared,
    };
  }

  /** 端口两端的 shape/dtype 兼容检查（仅对声明了 shape/dtype 的端口）。 */
  function checkEdgeTypes (
    id: NodeId,
    port: NormPort,
    srcId: NodeId | undefined,
    srcOut: PortType | null,
  ): void
  {
    if ( srcId === undefined || !table[ srcId ] ) return;
    const srcTensor = asTensorPort( srcOut );

    // DTYPE_MISMATCH：契约声明端口 dtype 且与源不同（中间有 Cast 则放行）。
    if ( port.dtype !== undefined && srcTensor && srcTensor.dtype !== port.dtype )
    {
      if ( opNameOf( table[ srcId ].op ) !== 'Cast' )
      {
        diags.push( mkDiag( {
          level: 'error',
          code: 'DTYPE_MISMATCH',
          node: id,
          port: port.name,
          message: `端口 ${ port.name } 期望 ${ port.dtype }，但源 ${ srcId } 输出 ${ srcTensor.dtype }`,
          messageEn: `port ${ port.name } expects ${ port.dtype } but source ${ srcId } yields ${ srcTensor.dtype }`,
          fix: { kind: 'insert-op', to: 'Cast', at: id, detail: `在 ${ srcId } 与 ${ id } 之间插入 Cast(${ port.dtype })` },
        } ) );
      }
    }

    // SHAPE_MISMATCH：两端 shape 都已知且为具体数字时不相等。
    if ( port.shape !== undefined && srcTensor )
    {
      const want = concreteShape( port.shape, symbols );
      const got = concreteShape( srcTensor.shape, symbols );
      if ( want && got && !shapeEq( want, got ) )
      {
        diags.push( mkDiag( {
          level: 'error',
          code: 'SHAPE_MISMATCH',
          node: id,
          port: port.name,
          message: `端口 ${ port.name } 期望 shape [${ want.join( ', ' ) }]，但源 ${ srcId } 输出 [${ got.join( ', ' ) }]`,
          messageEn: `port ${ port.name } expects shape [${ want.join( ', ' ) }] but source ${ srcId } yields [${ got.join( ', ' ) }]`,
          fix: { kind: 'insert-op', to: 'Reshape', at: id, detail: `在 ${ srcId } 与 ${ id } 之间插入 Reshape 对齐形状` },
        } ) );
      }
    }
  }

  // ---- 图级诊断（在全部节点推导之后） -------------------------------------

  // STRATEGY_INCOMPAT：requires 显式声明了与策略矛盾的 caps。
  const declaredRequires = model.trainer?.requires;
  if ( declaredRequires && declaredRequires.length > 0 && strategy !== 'custom' )
  {
    const allowed = STRATEGY_REQUIRES[ strategy ] ?? [];
    const extra = [ ...new Set( declaredRequires.filter( ( c ) => !allowed.includes( c ) ) ) ];
    if ( extra.length > 0 )
    {
      diags.push( mkDiag( {
        level: 'error',
        code: 'STRATEGY_INCOMPAT',
        node: rootId,
        message: `trainer.requires 声明了 [${ extra.join( ', ' ) }]，与策略 ${ strategy } 的要求 [${ allowed.join( ', ' ) }] 矛盾`,
        messageEn: `trainer.requires declares [${ extra.join( ', ' ) }], contradicting strategy ${ strategy } requirements [${ allowed.join( ', ' ) }]`,
        fix: { kind: 'switch-strategy', at: rootId, detail: `移除多余的 requires，或改用支持这些能力位的策略` },
      } ) );
    }
  }

  // UNREPRODUCIBLE_RNG：策略需要确定性 RNG，但全图无任何节点契约声明 rng。
  const graphHasRng = ids.some( ( id ) => contractOf( id )?.rng !== undefined );
  if ( STRATEGIES_NEEDING_RNG.has( strategy ) && !graphHasRng )
  {
    diags.push( mkDiag( {
      level: 'error',
      code: 'UNREPRODUCIBLE_RNG',
      node: rootId,
      message: `策略 ${ strategy } 需要确定性 RNG，但全图没有任何节点声明 rng 段`,
      messageEn: `strategy ${ strategy } needs deterministic RNG, but no node declares an rng scope`,
      fix: { kind: 'add-annot', at: rootId, detail: '为相关随机算子声明 rng.counterScope，或改用无需 RNG 的策略' },
    } ) );
  }

  // PARAM_REUSE_CONFLICT：同 tie 参数但输入 shape 不一致。
  {
    const groups = new Map<string, NodeId[]>();
    for ( const id of order )
    {
      const tie = propString( table[ id ].props, 'tie', envFor( table[ id ] ) );
      if ( tie === undefined || tie.trim() === '' ) continue;
      const list = groups.get( tie );
      if ( list ) list.push( id );
      else groups.set( tie, [ id ] );
    }
    const ties = [ ...groups.keys() ].sort();
    for ( const tie of ties )
    {
      const group = groups.get( tie )!;
      if ( group.length < 2 ) continue;
      const shapes = group.map( ( id ) => typed[ id ] ? firstInputShape( id ) : undefined );
      for ( let i = 0; i < group.length; i++ )
        for ( let j = i + 1; j < group.length; j++ )
        {
          const a = shapes[ i ];
          const b = shapes[ j ];
          if ( a && b && !shapeEq( a, b ) )
          {
            diags.push( mkDiag( {
              level: 'error',
              code: 'PARAM_REUSE_CONFLICT',
              node: group[ i ],
              message: `节点 ${ group[ i ] } 与 ${ group[ j ] } 共享 tie="${ tie }"，但输入 shape 不一致（[${ a.join( ', ' ) }] vs [${ b.join( ', ' ) }]）`,
              messageEn: `nodes ${ group[ i ] } and ${ group[ j ] } tie "${ tie }" but input shapes differ ([${ a.join( ', ' ) }] vs [${ b.join( ', ' ) }])`,
              fix: { kind: 'change-prop', to: 'tie', at: group[ j ], detail: '对共享参数统一输入 shape，或拆分 tie' },
            } ) );
          }
        }
    }
  }

  // BUS_CYCLE / TRAP_PATH：Bus 的 provide/inject 配对与环。
  {
    const busIds = ids.filter( ( id ) => opNameOf( table[ id ].op ) === 'Bus' );
    if ( busIds.length > 0 )
    {
      const providerByName = new Map<string, NodeId>();
      for ( const id of busIds )
      {
        const provide = propString( table[ id ].props, 'provide', envFor( table[ id ] ) );
        if ( provide && provide.trim() !== '' && !providerByName.has( provide ) )
          providerByName.set( provide, id );
      }

      const busEdges = new Map<NodeId, NodeId[]>();
      const injectOf = new Map<NodeId, string>();
      for ( const id of busIds ) busEdges.set( id, [] );
      for ( const id of busIds )
      {
        const inject = propString( table[ id ].props, 'inject', envFor( table[ id ] ), '' ) ?? '';
        if ( inject.trim() === '' ) continue;
        injectOf.set( id, inject );
        const provider = providerByName.get( inject );
        if ( provider === undefined )
        {
          diags.push( mkDiag( {
            level: 'error',
            code: 'BUS_CYCLE',
            node: id,
            message: `Bus 的 inject="${ inject }" 没有配对的 provide`,
            messageEn: `Bus inject="${ inject }" has no paired provide`,
            fix: { kind: 're-wire', at: id, detail: '补上与 inject 同名的 provide，或移除该 inject' },
          } ) );
          continue;
        }
        const arr = busEdges.get( provider );
        if ( arr ) arr.push( id );
      }

      // 环检测（DFS 三色 + 栈）。
      const color = new Map<NodeId, number>(); // 0 未访问 / 1 在栈 / 2 完成
      const stack: NodeId[] = [];
      const cycleFlags = new Set<NodeId>();
      const visit = ( u: NodeId ): void =>
      {
        color.set( u, 1 );
        stack.push( u );
        for ( const v of busEdges.get( u ) ?? [] )
        {
          const c = color.get( v ) ?? 0;
          if ( c === 1 )
          {
            const at = stack.indexOf( v );
            for ( let k = at; k < stack.length; k++ ) cycleFlags.add( stack[ k ] );
          }
          else if ( c === 0 ) visit( v );
        }
        color.set( u, 2 );
        stack.pop();
      };
      for ( const id of busIds ) if ( ( color.get( id ) ?? 0 ) === 0 ) visit( id );

      for ( const id of busIds )
      {
        if ( !cycleFlags.has( id ) ) continue;
        diags.push( mkDiag( {
          level: 'error',
          code: 'BUS_CYCLE',
          node: id,
          message: `Bus 的 provide→inject 依赖成环（节点 ${ id }）`,
          messageEn: `Bus provide→inject dependency forms a cycle at node ${ id }`,
          fix: { kind: 're-wire', at: id, detail: '打断环：让注入点不再（间接）依赖其提供点' },
        } ) );
      }

      // TRAP_PATH：provide→inject 两端在 children 树上的深度差过大。
      const depth = childDepths( rootId );
      for ( const id of busIds )
      {
        const inject = injectOf.get( id );
        if ( inject === undefined ) continue;
        const provider = providerByName.get( inject );
        if ( provider === undefined ) continue;
        const dp = depth.get( provider );
        const di = depth.get( id );
        if ( dp !== undefined && di !== undefined && Math.abs( dp - di ) > TRAP_PATH_DEPTH )
        {
          diags.push( mkDiag( {
            level: 'warn',
            code: 'TRAP_PATH',
            node: id,
            message: `Bus provide→inject 沿 children 深度差 ${ Math.abs( dp - di ) } > ${ TRAP_PATH_DEPTH }，跨层耦合过深`,
            messageEn: `Bus provide→inject spans children depth gap ${ Math.abs( dp - di ) } > ${ TRAP_PATH_DEPTH }`,
          } ) );
        }
      }
    }
  }

  // ORPHAN_NODE：从根不可达的节点。
  for ( const id of ids )
  {
    if ( reachable.has( id ) ) continue;
    diags.push( mkDiag( {
      level: 'warn',
      code: 'ORPHAN_NODE',
      node: id,
      message: `节点 ${ id } 从根 ${ rootId } 不可达（无下游路径）`,
      messageEn: `node ${ id } is unreachable from root ${ rootId } (no downstream path)`,
    } ) );
  }

  // STABILITY_HINT：深/宽超阈值且全图无 QKNorm 也无 ZLoss。
  {
    const hasQKNorm = ids.some( ( id ) => opNameOf( table[ id ].op ) === 'QKNorm' );
    const hasZLoss = ids.some( ( id ) => opNameOf( table[ id ].op ) === 'ZLoss' );
    if ( !hasQKNorm && !hasZLoss )
    {
      for ( const id of order )
      {
        const node = table[ id ];
        const env = envFor( node );
        const depthVal = propNumber( node.props, 'layers', env ) ?? propNumber( node.props, 'depth', env );
        const widthVal = propNumber( node.props, 'dim', env );
        const deep = depthVal !== undefined && depthVal >= STABILITY_DEPTH_THRESHOLD;
        const wide = widthVal !== undefined && widthVal >= STABILITY_WIDTH_THRESHOLD;
        if ( deep || wide )
        {
          const what = deep ? `深度 ${ depthVal } ≥ ${ STABILITY_DEPTH_THRESHOLD }` : `宽度 ${ widthVal } ≥ ${ STABILITY_WIDTH_THRESHOLD }`;
          const whatEn = deep ? `depth ${ depthVal } ≥ ${ STABILITY_DEPTH_THRESHOLD }` : `width ${ widthVal } ≥ ${ STABILITY_WIDTH_THRESHOLD }`;
          diags.push( mkDiag( {
            level: 'warn',
            code: 'STABILITY_HINT',
            node: id,
            message: `节点 ${ id } ${ what }，但全图未启用 QKNorm/ZLoss（logits 可能爆炸）`,
            messageEn: `node ${ id } has ${ whatEn } without QKNorm/ZLoss in graph (logits may explode)`,
            fix: { kind: 'insert-op', to: 'QKNorm', at: id, detail: '在注意力前插入 QKNorm，或在损失上追加 ZLoss' },
          } ) );
        }
      }
    }
  }

  // BUDGET_EXCEEDED：给了预算且粗估超过它。
  if ( opts.budget )
  {
    let memBytes = 0;
    let flopsEst = 0;
    for ( const id of order )
    {
      const tn = typed[ id ];
      if ( !tn ) continue;
      for ( const key of Object.keys( tn.outputs ) )
      {
        const t = asTensorPort( tn.outputs[ key ] );
        if ( !t ) continue;
        const elems = elemCount( t.shape, symbols );
        memBytes += elems * ( DTYPE_BYTES[ t.dtype ] ?? 4 );
        flopsEst += elems;
      }
    }
    if ( typeof opts.budget.memMB === 'number' )
    {
      const limit = opts.budget.memMB * MIB;
      if ( memBytes > limit )
      {
        diags.push( mkDiag( {
          level: 'error',
          code: 'BUDGET_EXCEEDED',
          node: rootId,
          message: `估算显存 ${ memBytes } B 超过预算 ${ opts.budget.memMB } MiB`,
          messageEn: `estimated memory ${ memBytes } B exceeds budget ${ opts.budget.memMB } MiB`,
          fix: { kind: 'change-prop', at: rootId, detail: '降低 dim/layers 或开启 recompute/量化以压缩显存' },
        } ) );
      }
    }
    if ( typeof opts.budget.flops === 'number' && flopsEst > opts.budget.flops )
    {
      diags.push( mkDiag( {
        level: 'error',
        code: 'BUDGET_EXCEEDED',
        node: rootId,
        message: `估算 FLOPs ${ flopsEst } 超过预算 ${ opts.budget.flops }`,
        messageEn: `estimated FLOPs ${ flopsEst } exceeds budget ${ opts.budget.flops }`,
        fix: { kind: 'change-prop', at: rootId, detail: '缩小模型规模或降低计算精度以减少 FLOPs' },
      } ) );
    }
  }

  // ---- caps 摘要 ----------------------------------------------------------
  const allSet = new Set<Cap>();
  const nondiffSet = new Set<NodeId>();
  const effectfulNodes: NodeId[] = [];
  const phaseSet = new Set<Phase>();
  for ( const id of order )
  {
    const tn = typed[ id ];
    if ( !tn ) continue;
    for ( const c of tn.caps ) allSet.add( c );
    if ( tn.caps.includes( 'nondiff' ) ) nondiffSet.add( id );
    if ( tn.effects === 'effectful' ) effectfulNodes.push( id );
    for ( const p of tn.phase ) phaseSet.add( p );
  }
  const all = CAP_ORDER.filter( ( c ) => allSet.has( c ) );
  const required: Cap[] = [ ...new Set( model.trainer?.requires ?? STRATEGY_REQUIRES[ strategy ] ?? [] ) ];
  const missing = CAP_ORDER.filter( ( c ) => required.includes( c ) && !allSet.has( c ) );
  const caps: CapsSummary = {
    all,
    missing,
    effectfulNodes,
    nondiffNodes: order.filter( ( id ) => nondiffSet.has( id ) ),
    phases: PHASE_ORDER.filter( ( p ) => phaseSet.has( p ) ),
  };

  // ---- 组装 TypedGraph ----------------------------------------------------
  const graph: TypedGraph = {
    root: rootId,
    order,
    nodes: typed,
    children: childMap,
    incoming,
  };

  return { graph, diags, caps };

  // -------------------------------------------------------------------------
  // 图内闭包工具（需要访问 table / typed / symbols）
  // -------------------------------------------------------------------------

  function firstConcreteDim ( outs: Record<string, PortType | null> ): number | undefined
  {
    for ( const key of Object.keys( outs ) )
    {
      const t = asTensorPort( outs[ key ] );
      if ( !t ) continue;
      for ( const d of t.shape ) if ( typeof d === 'number' ) return d;
    }
    return undefined;
  }

  function firstInputShape ( id: NodeId ): number[] | undefined
  {
    const tn = typed[ id ];
    if ( !tn ) return undefined;
    for ( const key of Object.keys( tn.inputs ) )
    {
      const t = asTensorPort( tn.inputs[ key ] );
      if ( !t ) continue;
      const nums = concreteShape( t.shape, symbols );
      if ( nums ) return nums;
    }
    return undefined;
  }

  function childDepths ( root: NodeId ): Map<NodeId, number>
  {
    const d = new Map<NodeId, number>();
    const q: NodeId[] = [ root ];
    d.set( root, 0 );
    while ( q.length > 0 )
    {
      const id = q.shift()!;
      const cur = d.get( id )!;
      for ( const c of table[ id ]?.children ?? [] )
        if ( !d.has( c ) && table[ c ] )
        {
          d.set( c, cur + 1 );
          q.push( c );
        }
    }
    return d;
  }
}
