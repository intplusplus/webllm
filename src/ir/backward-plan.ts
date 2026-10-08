/**
 * 反向传播「编译器排出的执行序列」（design/06 §4，缺口#2）。
 *
 * 设计立场（02 §2 R3）：**能力必须被声明**。正向由 `emit()` 排出 pass 序列，
 * 反向也应有自己的"编译器排出的执行序列"，而不是 `grad.ts` 里一段隐式的
 * `for` 逆序循环（解释执行）。
 *
 * 这里把"哪些节点、按什么逆序参与反向"显式排出为 `BackwardPlan`：
 *   - 求值顺序 = 拓扑逆序（与正向 emit 同源）；
 *   - 每个步骤标注其 op 与是否拥有 VJP 实现（与正向查 `CpuOpImpl` 对称地查 `VjpFn`）。
 * 梯度规约（acc 累加 + tie 归并 + 按 sources 路由）仍由 `backward()` 的执行核心完成；
 * 本文件只负责"排序列"，并为未来的 GPU 反向 pass（ENG-V6）铺好序列接口。
 */

import type { InferResult, Model, NodeId } from './types';
import type { GradRegistry } from './grad';

/** 取 op 名（去掉 `@版本` 后缀），与 exec.ts / infer.ts 的 opNameOf 同源。 */
function opNameOf ( ref: string ): string
{
  const at = ref.lastIndexOf( '@' );
  return at >= 0 ? ref.slice( 0, at ) : ref;
}

// ---------------------------------------------------------------------------
// 编译器排出的反向步骤
// ---------------------------------------------------------------------------

export interface BackwardStep
{
  /** 节点 id。 */
  nodeId: NodeId;
  /** 节点 op（opRef，与 grads 表键一致）。 */
  op: string;
  /** 该 op 是否有 VJP 实现（无则执行到它时若有梯度必抛，等同此前解释执行语义）。 */
  hasVjp: boolean;
}

export interface BackwardPlan
{
  /** 逆拓扑序：反向传播的执行序列（编译器排出）。 */
  steps: BackwardStep[];
  /** 反向序列的 pass 划分（当前 CPU 后端梯度缓冲彼此独立，恒为单 pass）。 */
  passBreaks: number;
}

/**
 * 排出反向 pass 序列（compile 阶段，纯函数、无副作用、不读前向值）。
 *
 * @param model  模型（取节点 opRef）
 * @param ir     静态分析结果（取拓扑序）
 * @param grads  VJP 注册表（判定每个 op 是否可参与反向）
 */
export function planBackward ( model: Model, ir: InferResult, grads: GradRegistry ): BackwardPlan
{
  const order = ir.graph?.order ?? Object.keys( ir.graph?.nodes ?? {} ).sort();
  const steps: BackwardStep[] = [];
  for ( let i = order.length - 1; i >= 0; i-- )
  {
    const id = order[ i ];
    const op = opNameOf( model.nodes?.[ id ]?.op ?? '' );
    steps.push( { nodeId: id, op, hasVjp: grads.has( op ) } );
  }
  // 当前 CPU 后向：每个节点的梯度缓冲相互独立（按张量身份累加到 acc），
  // 不存在 arena 段冲突，故恒为单 pass；留此字段为 GPU 后向分 pass 预留。
  return { steps, passBreaks: 0 };
}
