/**
 * Spec IR 公共出口（design/03 §1）。
 *
 * 这一层是 P0/P1/P2 的对外契约：上游（UI / AI 回路 / 联邦分发）只从这里取，
 * 不直接深挖内部文件。
 */

export * from './types';
export * from './op';
export * from './expr';
export * from './canonical';
export * from './hash';

export { ensureBuiltinOps, builtinOpList, builtinRegistry } from './ops';
export { migrate } from './migrate';
export { infer, type InferOptions } from './infer';
export {
  h,
  buildModel,
  toJsx,
  renderJsx,
  parseJsx,
  type JsxNode,
} from './jsx';
export {
  plan,
  type BackendId,
  type BackendCapability,
  type ClusterPeer,
  type ClusterCapability,
  type Budget,
  type PlanEnv,
  type ArenaSegment,
  type ArenaLayout,
  type DispatchItem,
  type PassPlan,
  type RNGPlan,
  type CommPlan,
  type CompressPlan,
  type TrustPlan,
  type BackendRule,
  type ExecutionPlan,
} from './plan';
export { emit, type Artifact } from './emit';
export {
  run,
  createImplRegistry,
  bindMapOf,
  type CpuRunContext,
  type CpuOpImpl,
  type CpuImplRegistry,
  type RunBinding,
  type RunResult,
  type RunTrace,
} from './exec';
export {
  tv,
  f32,
  u32,
  numel,
  createTensorTable,
  requireTensor,
  type TensorData,
  type TensorValue,
  type TensorTable,
} from './binding';
export { builtinCpuImpls } from './cpu-impls';
export { buildGptIr, bindBatch, type GptIr } from './tinygpt-ir';
export {
  backward,
  trainStep,
  createAdamW,
  createGradRegistry,
  dParamOf,
  zerosLike,
  asF32,
  addGrad,
  scalarValue,
  makeScalar,
  type GradContext,
  type VjpFn,
  type VjpResult,
  type GradRegistry,
  type BackwardOptions,
  type BackwardResult,
  type AdamWConfig,
  type Optimizer,
  type TrainStepResult,
} from './grad';
export { attachCrossEntropy, setTargets, LOSS_NODE_ID, TARGETS_TENSOR } from './train';
export { builtinCpuGrads } from './cpu-grads';

import type { Model } from './types';
import type { OpRegistry } from './op';
import { specHashInput } from './canonical';
import { sha256Hex } from './hash';

/**
 * 模型的规范指纹（03 §4）。
 * 全网同哈希 = 同结构 = 可聚合；device/shard/created 等不确定字段已被排除（INV-11）。
 */
export function specHash ( model: Model, registry?: OpRegistry ): string
{
  return sha256Hex( specHashInput( model, registry ) );
}
