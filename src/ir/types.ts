/**
 * Spec IR 数据模型与类型系统（design/03-IR与类型系统.md）。
 *
 * 定位：本文件是整个 P0 的**冻结契约**。所有上层（JSX、AI 生成、编译、联邦分发）
 * 都以这套类型为唯一中间表示。改动这里等于改全网数据契约 —— 只能新增可选字段。
 *
 * 四条设计约束（03 §2.1）：
 *   1. 节点 `op` 指向注册表（OpRef），不内联实现 ⇒ specHash 不含实现细节；
 *   2. `props` 只放结构化/可序列化数据，不含函数（行为差异用不同的 op 表达）；
 *   3. `children`（树）与 `wiring`（图）并存；
 *   4. `annot` 单列 ⇒ 执行策略与语义解耦（换 device/shard 不改 specHash，INV-11）。
 */

/** IR schema 版本；进 specHash，不同 major 不混聚（INV-9）。 */
export const IR_SCHEMA_VERSION = '1.0';

export type SchemaVersion = string;
export type NodeId = string;
export type OpName = string;
export type OpVersion = string;
/** `name@version`；同一名字不同版本是**不同 op**（05 §2）。 */
export type OpRef = string;

// ---------------------------------------------------------------------------
// 类型系统（03 §3.1）
// ---------------------------------------------------------------------------

export type DType = 'f32' | 'f16' | 'bf16' | 'i32' | 'i64' | 'u32';

/** DimConst（数字）或 DimSym（`B`/`T`… 动态维，infer 里做符号求解）。 */
export type Dim = number | string;
export type DimList = Dim[];

export type Layout =
  | { kind: 'dense' }
  | { kind: 'blocked'; block: number }
  | { kind: 'sparse'; scope: string };

export interface TensorType {
  shape: DimList;
  dtype: DType;
  layout?: Layout;
}

/** State 是显式生命周期的非张量值（KV cache / ScanState / buffer / memory，03 §3.1）。 */
export interface StateType {
  kind: 'kv' | 'scan' | 'buffer' | 'memory';
  shape: DimList;
  dtype: DType;
}

/** 一个端口的类型：张量 / 状态 / 分布（Dist）之一。 */
export type PortType = TensorType | StateType;

export function isTensorType ( t: PortType ): t is TensorType
{
  return ( t as StateType ).kind === undefined;
}

// ---------------------------------------------------------------------------
// props 与受限表达式（03 §3.3）
// ---------------------------------------------------------------------------

/**
 * 受限表达式：必须**可静态求值**。禁止 lambda / 闭包 / 外部调用。
 * 这样"模型里写 for/if"保留，但不会出现不可序列化的函数。
 */
export type Expr =
  | { kind: 'lit'; value: number | string | boolean }
  | { kind: 'ref'; path: string }                       // 如 "meta.B"、"props.dim"
  | { kind: 'bin'; op: '+' | '-' | '*' | '/' | '%'; a: Expr; b: Expr }
  | { kind: 'range'; from: Expr; to: Expr; step?: Expr }
  | { kind: 'cond'; test: Expr; then: Expr; else: Expr };

/** 结构化的嵌套 prop（如 `bind: { w: 'layers.0.wq.w' }`）。只允许纯数据。 */
export interface PropObject { [ key: string ]: PropValue }

export type PropValue = number | string | boolean | null | PropValue[] | PropObject | Expr;
export type Props = Record<string, PropValue>;

export function isExpr ( v: unknown ): v is Expr
{
  return typeof v === 'object' && v !== null && typeof ( v as Expr ).kind === 'string';
}

// ---------------------------------------------------------------------------
// annot / Edge / ModelNode / Model（03 §2）
// ---------------------------------------------------------------------------

export type Phase = 'train' | 'prefill' | 'decode' | 'sample' | 'rollout' | 'nudge' | 'free';
export type DeviceId = string;
export type Effects = 'pure' | 'effectful';
export type Precision = 'fp32' | 'fp16' | 'int4' | 'bf16';
export type EdgeKind = 'tensor' | 'state' | 'signal';

export interface ShardSpec {
  plan: 'layer' | 'tensor' | 'expert' | 'auto';
  peers?: number;
}

export interface Annot {
  phase?: Phase | Phase[];
  /** `auto` = 交 plan() 决定；两者都不进 specHash。 */
  device?: 'auto' | DeviceId;
  shard?: ShardSpec;
  precision?: Precision;
  recompute?: boolean;
  effects?: Effects;
  cost?: { flops?: string; memPeak?: string };
}

export interface Edge {
  from: NodeId;
  fromPort: string;
  to: NodeId;
  toPort: string;
  kind: EdgeKind;
}

export interface NodeMeta {
  note?: string;
  span?: SourceSpan;
}

export interface SourceSpan {
  file?: string;
  line: number;
  column: number;
  endLine?: number;
  endColumn?: number;
}

export interface ModelNode {
  id: NodeId;
  /** 指向注册表中的 op（含契约）。 */
  op: OpRef;
  props: Props;
  /** 有序子节点（组合原语使用）。 */
  children?: NodeId[];
  /** 数据/状态/信号边（非树结构用）。 */
  wiring?: Edge[];
  /** 命名输入插槽（CrossAttn cond 等）。 */
  slot?: Record<string, NodeId>;
  annot?: Annot;
  meta?: NodeMeta;
}

export interface TrainerSpec {
  strategy: StrategyId;
  params: Props;
  /** 由 strategy 推导；IR 可显式声明以提前校验。 */
  requires?: Cap[];
  schedule: { localSteps: number; syncEvery: number; precision: Precision };
}

export interface ModelMeta {
  author?: string;
  created?: string;
  /** 溯源：由谁 fork/派生。 */
  parentHash?: string;
  /** 训练用数据集指纹（全网一致的前提）。 */
  datasetHash?: string;
  /** SPDX 表达式（公地用）。 */
  license?: string;
  tags?: string[];
}

export interface Model {
  schemaVersion: SchemaVersion;
  /** 纯计算根节点。 */
  graph: ModelNode;
  /**
   * 平铺节点表：`graph` 与各 `children` 里的 NodeId 在此解析。
   * （03 §2 的 ModelNode.children 是 NodeId 引用，故需一张表来收敛。
   *  规范序列化时按 id 字典序输出，保证 canonical。）
   */
  nodes: Record<NodeId, ModelNode>;
  /** 学习策略（与 graph 正交，见 07）。 */
  trainer: TrainerSpec;
  meta: ModelMeta;
}

// ---------------------------------------------------------------------------
// 能力位与策略（03 §3.2、05 §3）
// ---------------------------------------------------------------------------

/** 8 个能力位。 */
export type Cap =
  | 'forward'
  | 'vjp'
  | 'jvp'
  | 'logprob'
  | 'localGrad'
  | 'perturbable'
  | 'invertible'
  | 'nondiff';

export type StrategyId =
  | 'backprop'
  | 'forward-mode'
  | 'spsa'
  | 'es'
  | 'reinforce'
  | 'local'
  | 'forward-forward'
  | 'equilibrium-prop'
  | 'variational'
  | 'custom';

/**
 * 策略对能力位的要求（03 §3.2 类型规则、05 §3.1）。
 * 注意 backprop 是特例：要求"损失路径上每个节点有 `vjp` 或 `localGrad`"，
 * 无法用一条静态集合表达，infer 里单独处理。
 */
export const STRATEGY_REQUIRES: Record<StrategyId, Cap[]> = {
  backprop: [ 'forward' ],
  'forward-mode': [ 'forward', 'jvp' ],
  spsa: [ 'forward', 'perturbable' ],
  es: [ 'forward', 'perturbable' ],
  reinforce: [ 'forward', 'logprob' ],
  local: [ 'forward', 'localGrad' ],
  'forward-forward': [ 'forward', 'localGrad' ],
  'equilibrium-prop': [ 'forward', 'invertible' ],
  variational: [ 'forward', 'logprob' ],
  custom: [ 'forward' ],
};

/** 需要确定性 RNG 的策略（06 §5）——若拓扑含 RNG 需求但无 RNG 声明 ⇒ UNREPRODUCIBLE_RNG。 */
export const STRATEGIES_NEEDING_RNG: ReadonlySet<StrategyId> = new Set<StrategyId>( [
  'spsa', 'es', 'reinforce', 'variational',
] );

export function isBackprop ( s: StrategyId ): boolean
{
  return s === 'backprop';
}

// ---------------------------------------------------------------------------
// 诊断（03 §5）
// ---------------------------------------------------------------------------

export type DiagLevel = 'error' | 'warn' | 'info';

/** 诊断码全表（03 §5.2）。新增只能追加（1.x 向后兼容）。 */
export type DiagCode =
  | 'SHAPE_MISMATCH'
  | 'DTYPE_MISMATCH'
  | 'MISSING_PARAM'
  | 'UNDEFINED_SLOT'
  | 'BUS_CYCLE'
  | 'PHASE_LEAK'
  | 'EFFECT_IN_PURE'
  | 'DYNAMIC_SHAPE_UNRESOLVED'
  | 'CAPABILITY_MISSING'
  | 'STRATEGY_INCOMPAT'
  | 'UNREPRODUCIBLE_RNG'
  | 'BUDGET_EXCEEDED'
  | 'SHARD_INFEASIBLE'
  | 'PARAM_REUSE_CONFLICT'
  | 'TRAP_PATH'
  | 'ORPHAN_NODE'
  | 'SCHEDULE_DEADLOCK'
  | 'SCHEDULE_STARVATION'
  | 'SCHEDULE_UNSCHEDULABLE'
  | 'STABILITY_HINT'
  | 'EVAL_UNDERPOWERED'
  | 'BACKDOOR_SUSPECT'
  // 以下为 plan() 的软判据（06 §2；03 §5.2 表未列，但 ENG-V11/V12 要求这些码）
  | 'ROOM_TOO_HETEROGENEOUS'
  | 'COMPRESSION_WITHOUT_EF'
  | 'BYZANTINE_BUDGET_INFEASIBLE'
  | 'COMPRESSION_STATE_MISSING';

export type AutoFixKind =
  | 'insert-op'
  | 'change-prop'
  | 're-wire'
  | 'switch-strategy'
  | 'add-annot'
  | 'change-shard'
  | 'quarantine'
  | 'resize-probe-pool';

export interface AutoFix {
  kind: AutoFixKind;
  /** 建议插入/改成的 op 名（insert-op / switch-strategy）。 */
  to?: string;
  at?: NodeId | Edge;
  detail?: string;
}

export interface Diag {
  level: DiagLevel;
  code: DiagCode;
  node?: NodeId;
  port?: string;
  /** 面向 agent 的自然语言（03 §5.1 要求中英双份）。 */
  message: string;
  messageEn?: string;
  fix?: AutoFix;
  at?: SourceSpan;
}

// ---------------------------------------------------------------------------
// infer() 输出（03 §5.1）
// ---------------------------------------------------------------------------

/** 一个完成类型推断的节点。 */
export interface TypedNode {
  node: ModelNode;
  op: OpRef;
  /** 按输入端口名索引（`in` 契约的顺序端口 + slot 命名端口）。 */
  inputs: Record<string, PortType | null>;
  outputs: Record<string, PortType | null>;
  caps: Cap[];
  effects: Effects;
  pure: boolean;
  phase: Phase[];
}

export interface CapsSummary {
  /** 全图出现过的能力位并集。 */
  all: Cap[];
  /** 策略要求但图中存在缺口的能力位。 */
  missing: Cap[];
  effectfulNodes: NodeId[];
  nondiffNodes: NodeId[];
  /** 全图涉及相位并集。 */
  phases: Phase[];
}

export interface TypedGraph {
  root: NodeId;
  /** 拓扑序（父→子）。 */
  order: NodeId[];
  nodes: Record<NodeId, TypedNode>;
  /** 每个节点解析出的子节点 id（组合原语用）。 */
  children: Record<NodeId, NodeId[]>;
  /** 反向边索引：to → Edge[]。 */
  incoming: Record<NodeId, Edge[]>;
}

export interface InferResult {
  graph: TypedGraph;
  diags: Diag[];
  caps: CapsSummary;
}

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

/** WebGPU 硬约束（06 §3、INV-5）。 */
export const MAX_BIND_GROUPS = 4;
export const MAX_BUFFER_SIZE = 2 * 1024 * 1024 * 1024; // 2 GiB

/** 稳定化启发式阈值（05 §5.3 `STABILITY_HINT`）。 */
export const STABILITY_DEPTH_THRESHOLD = 48;
export const STABILITY_WIDTH_THRESHOLD = 4096;
