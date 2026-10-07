/**
 * 把损失接到 IR 图上（让 `run()`/`backward()` 有可微的根）。
 *
 * 设计上"损失是图的一部分"而不是训练循环里的一个后处理：这样
 * `caps: [forward, vjp]` 的 CrossEntropy 与其它算子同构，反向也走同一张 VJP 表。
 */

import type { Model, ModelNode } from './types';
import { builtinRegistry } from './op';
import { ensureBuiltinOps } from './ops';
import { u32, type TensorTable } from './binding';

export const LOSS_NODE_ID = 'loss';
export const TARGETS_TENSOR = 'input.targets';

/**
 * 在模型根之上挂一个 CrossEntropy 节点，使其成为新的根。
 * 幂等：已经挂过就直接返回原模型。
 */
export function attachCrossEntropy ( model: Model ): Model
{
  ensureBuiltinOps();
  if ( model.nodes[ LOSS_NODE_ID ] ) return model;

  const def = builtinRegistry.latest( 'CrossEntropy' );
  if ( !def ) throw new Error( 'CrossEntropy 未注册，无法接损失' );

  const rootId = model.graph.id;
  const node: ModelNode = {
    id: LOSS_NODE_ID,
    op: `${ def.contract.op }@${ def.contract.version }`,
    props: { reduction: 'mean', bind: { targets: TARGETS_TENSOR } },
    slot: { logits: rootId },
  };
  return {
    ...model,
    graph: node,
    nodes: { ...model.nodes, [ LOSS_NODE_ID ]: node },
  };
}

/** 把一批目标 token 写进张量表（长度必须是 B*T）。 */
export function setTargets ( tensors: TensorTable, targets: Uint32Array, B: number, T: number ): void
{
  if ( targets.length !== B * T )
    throw new Error( `setTargets: 长度 ${ targets.length } != B*T=${ B * T }` );
  tensors.set( TARGETS_TENSOR, u32( targets, [ B * T ] ) );
}
