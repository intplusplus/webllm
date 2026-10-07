/**
 * emit()：把 ExecutionPlan 落成可执行 artifact（design/06-执行与调度引擎.md §1/§3，ENG-V2）。
 *
 * 职责：
 *   1. 按拓扑序把每个节点展成一条 DispatchItem（kernel + workgroups）；
 *   2. **自动插 passBreak**：依据每个 op 契约的 `writesReads` 声明，维护"当前 pass 已读/
 *      已写区段集合"，遇到 WAW / WAR / RAW 冲突即开新 pass（把 BUG-001 的教训从
 *      "人记得"变成"机器算得出"）；
 *   3. 关键不变式（INV-2）：同一 pass 内任意两条 dispatch 之间，对同一 arena 区段
 *      都不得有冲突 —— 这是 ENG-V2 用随机图断言的性质。
 *
 * 纯粹、无副作用：不抛异常（异常一律转 warnings），不读时钟/随机源。
 */

import type { Dim, Diag, InferResult, Model, ModelNode, NodeId, TypedNode } from './types';
import type { OpRegistry } from './op';
import { builtinRegistry } from './op';
import { ensureBuiltinOps } from './ops';
import type { ArenaLayout, DispatchItem, ExecutionPlan, PassPlan } from './plan';

// ---------------------------------------------------------------------------
// 公开类型契约（06 §1）
// ---------------------------------------------------------------------------

export interface Artifact
{
  plan: ExecutionPlan;
  passes: PassPlan[];       // 自动插 passBreak 后的最终 pass 划分
  arena: ArenaLayout;
  passBreaks: number;       // 插了多少次 pass 边界
  warnings: Diag[];
}

// ---------------------------------------------------------------------------
// 确定性启发式（与 plan() 保持一致，此处独立实现以满足模块边界）
// ---------------------------------------------------------------------------

const ELEMS_PER_WORKGROUP = 64;
const MAX_DISPATCH_DIM = 65535;

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

// ---------------------------------------------------------------------------
// emit()
// ---------------------------------------------------------------------------

/**
 * 真正做事的实现（外层包 try/catch，异常转 warning）。
 */
function emitInner (
  model: Model,
  ir: InferResult,
  p: ExecutionPlan,
  warnings: Diag[],
): Artifact
{
  const registry: OpRegistry = builtinRegistry;
  const order: NodeId[] = ir.graph?.order ?? Object.keys( ir.graph?.nodes ?? {} ).sort();

  const nodeOf = ( id: NodeId ): ModelNode | undefined => model.nodes?.[ id ];

  // ---- 展 dispatch 并自动插 passBreak（ENG-V2） ---------------------------
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
    const node = nodeOf( id );
    if ( !node )
    {
      warnings.push( {
        level: 'warn', code: 'ORPHAN_NODE', node: id,
        message: `拓扑序中的节点 ${ id } 不在 nodes 表中，跳过`,
        messageEn: `node ${ id } in topo order missing from nodes table; skipped`,
      } );
      continue;
    }
    const contract = registry.get( node.op )?.contract;
    const reads = contract?.writesReads?.reads ?? [];
    const writes = contract?.writesReads?.writes ?? [];

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

    dispatches.push( {
      nodeId: id,
      kernel: contract?.impl?.entry ?? contract?.op ?? node.op,
      workgroups: workgroupsOf( ir.graph?.nodes?.[ id ] ),
    } );
    for ( const s of effReads ) curReads.add( s );
    for ( const s of writes ) curWrites.add( s );
    for ( const s of writes ) everWritten.add( s );
  }
  flush();

  // ---- 写回 plan（保持 passes 两者一致） ---------------------------------
  p.passes = passes;
  const arena = p.memory?.arena ?? { segments: [], totalBytes: 0, bindGroups: 0 };

  return { plan: p, passes, arena, passBreaks, warnings };
}

export function emit ( model: Model, ir: InferResult, p: ExecutionPlan ): Artifact
{
  ensureBuiltinOps();
  // 继承 plan 的 warn/error 诊断，使 artifact 自带完整告警面。
  const warnings: Diag[] = ( p.diags ?? [] ).filter( ( d ) => d.level !== 'info' );
  try
  {
    return emitInner( model, ir, p, warnings );
  }
  catch ( err )
  {
    // 不抛异常：任何意外都转成 warning，并返回当前（可能为空的）划分。
    warnings.push( {
      level: 'warn', code: 'SCHEDULE_UNSCHEDULABLE',
      message: `emit 期间发生意外，已降级为空 pass 序列：${ String( err ) }`,
      messageEn: `unexpected error during emit; degraded to empty passes: ${ String( err ) }`,
    } );
    p.passes = [];
    return {
      plan: p,
      passes: [],
      arena: p.memory?.arena ?? { segments: [], totalBytes: 0, bindGroups: 0 },
      passBreaks: 0,
      warnings,
    };
  }
}
