/**
 * 把损失接到 IR 图上（让 `run()`/`backward()` 有可微的根）。
 *
 * 设计上"损失是图的一部分"而不是训练循环里的一个后处理：这样
 * `caps: [forward, vjp]` 的 CrossEntropy 与其它算子同构，反向也走同一张 VJP 表。
 */

import type { Model, ModelNode } from './types';
import { builtinRegistry } from './op';
import { ensureBuiltinOps } from './ops';
import { f32, u32, type TensorTable } from './binding';

export const LOSS_NODE_ID = 'loss';
export const TARGETS_TENSOR = 'input.targets';
export const LOSS_WEIGHTS_TENSOR = 'input.lossw';

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

/**
 * 在已挂 plain 损失的模型上，派生一份 **weighted** 根：loss 节点额外 bind 一个
 * 逐行系数端口（`weight`，[B*T] 的 f32）。其余节点与原模型**共享引用**（浅拷贝），
 * 于是两份 model 跑的是同一张权重表 —— CPU 训练器正是靠它同时持有
 * plain / weighted 两条编译产物。
 *
 * reduction 语义（与 GPU kernel ce_softmax_bwd 对齐，归一化由调用方负责）：
 *   - 'sum'（默认，训练台用）：loss=Σw·ce，dlogits=w·(softmax−onehot)
 *     —— SFT 传 1/N、DPO 传 ±β 精确系数
 *   - 'mean'：loss=Σw·ce/Σw，dlogits=…/Σw（通用加权平均）
 *
 * 注意必须传**未挂过 weighted** 的模型（plain 或原始根均可）。
 */
export function attachCrossEntropyWeighted (
  model: Model,
  opts: { weightKey?: string; reduction?: 'mean' | 'sum' } = {},
): Model
{
  const lossNode = model.nodes[ LOSS_NODE_ID ];
  if ( !lossNode )
    throw new Error( 'attachCrossEntropyWeighted：模型还没有 loss 节点（先 attachCrossEntropy）' );
  const prevBind = ( lossNode.props.bind ?? {} ) as Record<string, string>;
  const node: ModelNode = {
    ...lossNode,
    props: {
      ...lossNode.props,
      reduction: opts.reduction ?? 'sum',
      bind: { ...prevBind, weight: opts.weightKey ?? LOSS_WEIGHTS_TENSOR },
    },
  };
  return {
    ...model,
    graph: node,
    nodes: { ...model.nodes, [ LOSS_NODE_ID ]: node },
  };
}

/** 把一行权重写进张量表（长度必须是 B*T）。 */
export function setLossWeights ( tensors: TensorTable, weights: Float32Array, B: number, T: number ): void
{
  if ( weights.length !== B * T )
    throw new Error( `setLossWeights: 长度 ${ weights.length } != B*T=${ B * T }` );
  tensors.set( LOSS_WEIGHTS_TENSOR, f32( weights, [ B * T ] ) );
}
