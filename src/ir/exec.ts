/**
 * `run()`：执行 `emit()` 产出的 artifact（design/06 §1 的第四步）。
 *
 * 设计里 run() 的定位是「**同一 artifact 每步复用**」——树是静态的，一次编译、每步复用
 * （这正是现有 `trainer.ts` 每步重建调用的反面，也是 ENG-V1 的要求）。所以这里：
 *   - **不做任何静态分析**（不调 infer/plan/emit），只按 `ir.graph.order` 求值；
 *   - 求值顺序 = 拓扑序；artifact 只贡献 `passBreaks/passCount` 这类可观测信息；
 *   - 一切形状/类型在 infer 阶段已判过，run 期只按**实际张量**算。
 *
 * 后端选择（06 §6）：WebGPU 是主编译目标，CPU 是**对拍基准**。本轮先落地 CPU 后端，
 * 因为它是唯一能在 Node 里无头验证的路径（也是"IR 语义到底对不对"的唯一硬判据）。
 *
 * 端口解析规则必须与 `infer()` **完全一致**（否则类型判过的图跑不对）：
 *   slot 命名插槽 > wiring 边（toPort 匹配）> children 位置对应 > props.bind
 */

import type { InferResult, Model, NodeId, Props } from './types';
import type { Artifact } from './emit';
import type { OpContract, OpRegistry } from './op';
import { builtinRegistry } from './op';
import { ensureBuiltinOps } from './ops';
import { requireTensor, type TensorTable, type TensorValue } from './binding';

// ---------------------------------------------------------------------------
// CPU op 实现契约
// ---------------------------------------------------------------------------

export interface CpuRunContext
{
  nodeId: NodeId;
  /** 不带版本的 op 名（用于实现内分支）。 */
  op: string;
  props: Props;
  /** 端口值（按 `io.in` 端口名）。无来源即 null。 */
  ins: Record<string, TensorValue | null>;
  /** `props.bind` 解析出的参数（权重/常量）。端口也可能来自这里。 */
  params: Record<string, TensorValue>;
  /** 子节点输出（按 children 顺序）——组合原语用。 */
  childOutputs: TensorValue[];
  /** 符号维绑定（如 `{ B: 2, T: 16 }`）。RoPE 这类需要 T 的算子读它。 */
  symbols: Record<string, number>;
}

export type CpuOpImpl = ( ctx: CpuRunContext ) => Record<string, TensorValue>;

export interface CpuImplRegistry
{
  get ( op: string ): CpuOpImpl | undefined;
  register ( op: string, impl: CpuOpImpl ): void;
  has ( op: string ): boolean;
  ops (): string[];
}

export function createImplRegistry (): CpuImplRegistry
{
  const map = new Map<string, CpuOpImpl>();
  return {
    get: ( op ) => map.get( op ),
    register: ( op, impl ) => { map.set( op, impl ); },
    has: ( op ) => map.has( op ),
    ops: () => [ ...map.keys() ].sort(),
  };
}

// ---------------------------------------------------------------------------
// run()
// ---------------------------------------------------------------------------

export interface RunBinding
{
  /** 权重与输入的统一张量表（`props.bind` 里的名字在此解析）。 */
  tensors: TensorTable;
  /** 符号维绑定（`T` 必须给，RoPE/Attention 用；`B` 用于诊断）。 */
  symbols?: Record<string, number>;
}

export interface RunTrace
{
  order: NodeId[];
  nodes: Record<NodeId, Record<string, TensorValue>>;
  passCount: number;
  passBreaks: number;
}

export interface RunResult
{
  trace: RunTrace;
  /** 根节点第一个输出端口的值（模型输出）。 */
  rootOutput: TensorValue;
  /** 取任一节点任一端口的输出。 */
  get ( nodeId: NodeId, port?: string ): TensorValue | undefined;
}

/** 从 `Name@version` 取回不带版本的 op 名。 */
function opNameOf ( ref: string ): string
{
  const at = ref.lastIndexOf( '@' );
  return at >= 0 ? ref.slice( 0, at ) : ref;
}

/** `props.bind` 归一化为 `{ 名字: 张量名 }`；非字符串项忽略。 */
export function bindMapOf ( props: Props ): Record<string, string>
{
  const raw = props[ 'bind' ];
  const out: Record<string, string> = {};
  if ( raw === null || raw === undefined || typeof raw !== 'object' || Array.isArray( raw ) ) return out;
  const src = raw as Record<string, unknown>;
  for ( const k of Object.keys( src ) )
  {
    const v = src[ k ];
    if ( typeof v === 'string' && v.trim() !== '' ) out[ k ] = v;
  }
  return out;
}

/** 端口名归一化：与 infer 的 `normPorts` 同规则（无名端口按位命名）。 */
function inPortNames ( contract: OpContract | undefined ): string[]
{
  const list = contract?.io?.in ?? [];
  return list.map( ( s, i ) =>
  {
    if ( typeof s === 'string' ) return s.trim() !== '' ? s : `in${ i }`;
    return s.name && s.name.trim() !== '' ? s.name : `in${ i }`;
  } );
}

function outPortNames ( contract: OpContract | undefined ): string[]
{
  const list = contract?.io?.out ?? [];
  return list.map( ( s, i ) =>
  {
    if ( typeof s === 'string' ) return s.trim() !== '' ? s : `out${ i }`;
    return s.name && s.name.trim() !== '' ? s.name : `out${ i }`;
  } );
}

/**
 * 执行一个已编译的 artifact。
 *
 * @param model  权威 IR
 * @param ir     infer 结果（提供拓扑序，避免 run 期重新分析）
 * @param artifact emit 结果（提供 pass 可观测信息；run 期不重新编译）
 * @param binding 张量绑定（权重 + 输入 + 符号维）
 * @param impls  CPU 实现表（由 `builtinCpuImpls()` 提供）
 */
export function run (
  model: Model,
  ir: InferResult,
  artifact: Artifact,
  binding: RunBinding,
  impls: CpuImplRegistry,
  registry: OpRegistry = builtinRegistry,
): RunResult
{
  ensureBuiltinOps();

  const symbols = binding.symbols ?? {};
  const order = ir.graph.order.length > 0 ? ir.graph.order : Object.keys( model.nodes ).sort();
  const incoming = ir.graph.incoming ?? {};

  // nodeId → 端口 → 值
  const values: Record<NodeId, Record<string, TensorValue>> = {};

  for ( const id of order )
  {
    const node = model.nodes[ id ];
    if ( !node ) throw new Error( `run: 节点表缺少 ${ id }` );
    const def = registry.get( node.op );
    const contract = def?.contract;
    const op = opNameOf( node.op );
    const impl = impls.get( op );
    if ( !impl )
      throw new Error( `run: op ${ node.op } 没有 CPU 实现（当前实现表：${ impls.ops().join( ', ' ) }）` );

    const bind = bindMapOf( node.props );
    const params: Record<string, TensorValue> = {};
    for ( const key of Object.keys( bind ) )
      params[ key ] = requireTensor( binding.tensors, bind[ key ], `${ id }.bind.${ key }` );

    // ---- 端口解析（与 infer 同优先级） ----
    const ins: Record<string, TensorValue | null> = {};
    const ports = inPortNames( contract );
    const children = node.children ?? [];
    for ( let k = 0; k < ports.length; k++ )
    {
      const port = ports[ k ];
      let v: TensorValue | null = null;

      if ( node.slot && Object.prototype.hasOwnProperty.call( node.slot, port ) )
      {
        v = firstOutput( values[ node.slot[ port ] ] );
      }
      else
      {
        const edge = ( incoming[ id ] ?? [] ).find( ( e ) => e.toPort === port );
        if ( edge ) v = values[ edge.from ]?.[ edge.fromPort ] ?? firstOutput( values[ edge.from ] );
        else if ( k < children.length ) v = firstOutput( values[ children[ k ] ] );
      }

      if ( v === null && bind[ port ] !== undefined ) v = params[ port ] ?? null;
      ins[ port ] = v;
    }

    // ---- 子节点输出（组合原语用） ----
    const childOutputs: TensorValue[] = [];
    for ( const c of children )
    {
      const o = firstOutput( values[ c ] );
      if ( o ) childOutputs.push( o );
    }

    const outs = impl( { nodeId: id, op, props: node.props, ins, params, childOutputs, symbols } );
    if ( !outs || typeof outs !== 'object' )
      throw new Error( `run: ${ node.op } 的实现没有返回输出对象` );

    // 端口名兜底：实现返回的 key 若与契约端口不一致，按声明顺序补名（便于组合原语只返回 out）。
    const declared = outPortNames( contract );
    if ( declared.length > 0 && outs[ declared[ 0 ] ] === undefined )
    {
      const first = Object.keys( outs )[ 0 ];
      if ( first !== undefined ) outs[ declared[ 0 ] ] = outs[ first ];
    }

    values[ id ] = outs;
  }

  const rootId = ir.graph.root ?? order[ 0 ];
  const rootOutput = firstOutput( values[ rootId ] );
  if ( !rootOutput ) throw new Error( `run: 根节点 ${ rootId } 没有产生输出` );

  return {
    trace: {
      order: [ ...order ],
      nodes: values,
      passCount: artifact.passes.length,
      passBreaks: artifact.passBreaks,
    },
    rootOutput,
    get: ( nodeId, port ) =>
    {
      const v = values[ nodeId ];
      if ( !v ) return undefined;
      if ( port === undefined ) return firstOutput( v ) ?? undefined;
      return v[ port ];
    },
  };
}

function firstOutput ( v: Record<string, TensorValue> | undefined ): TensorValue | null
{
  if ( !v ) return null;
  for ( const k of Object.keys( v ) ) return v[ k ];
  return null;
}
