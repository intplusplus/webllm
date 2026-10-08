/**
 * op 契约与注册表（design/05-算子与内核层.md §2/§4）。
 *
 * 算子层是**唯一碰 GPU 的层**：对外只暴露 op 契约，对内藏 WGSL 实现。
 *
 * 两条关键设计：
 *   1. **契约（数据）与实现（函数）分离**：`OpContract` 是纯数据，进 `specHash`
 *      （见 03 §4.1：算子实现版本 + 依赖版本必须进哈希）；而 `shape` 这类
 *      运行时函数挂在校验后的 `OpDefinition` 上，**不进哈希**。
 *   2. **`defineOp` 是受信逃生门（INV-12）**：进图的 op 必须是"已审阅贡献"
 *      （带 codeHash；将来带签名），不是运行时 eval 一段字符串。
 */

import type { Cap, DType, Dim, Effects, OpName, OpRef, Phase, PortType, Props } from './types';

// ---------------------------------------------------------------------------
// 契约
// ---------------------------------------------------------------------------

export type PropType = 'int' | 'float' | 'bool' | 'string' | 'dims' | 'enum';

export interface OpPropSpec {
  type: PropType;
  required?: boolean;
  default?: number | string | boolean | Array<number | string>;
  min?: number;
  max?: number;
  /** type='enum' 时的取值域。 */
  values?: string[];
  doc?: string;
}

/** 端口的声明式描述（`io.in`/`io.out` 里每一项）。 */
export interface OpPortSpec {
  /** 端口名；`in` 数组里的无名端口按位置命名为 `in0/in1/...`。 */
  name?: string;
  /** 期望形状（符号表达式，如 `["B","T",{"ref":"props.dim"}]` 的简化写法）。 */
  shape?: Dim[];
  dtype?: DType;
  /** 端口是否可选（如 Matmul 的 bias）。 */
  optional?: boolean;
  /**
   * 是否允许"位置兜底"：当端口未通过 slot/edge/param 绑定时，是否允许用
   * 第 k 个子节点兜底（children[k]）。默认 true。
   * 设为 false 时该端口**只能**来自 slot/edge/param；缺省则保持 null，
   * 由实现层抛出清晰报错（而非静默复用 child）。Residual.x 用此开关杜绝"缺 x 时
   * 把首个 child 当 x、输出翻倍"的静默错误（W3）。
   */
  positional?: boolean;
  doc?: string;
}

export interface OpIoSchema {
  in: Array<OpPortSpec | string>;
  out: Array<OpPortSpec | string>;
  /** 命名输入插槽（CrossAttn 的 cond 等）。 */
  slots?: Record<string, string[]>;
}

export interface OpCost {
  flops?: string;
  mem?: string;
}

/** 读写依赖声明：`plan()` 据此**自动插 passBreak**（06 §3）。 */
export interface OpWritesReads {
  writes?: string[];
  reads?: string[];
}

/** op 契约：全部是可序列化数据，进 specHash。 */
export interface OpContract {
  op: OpName;
  version: string;
  props: Record<string, OpPropSpec>;
  io: OpIoSchema;
  caps: Cap[];
  phase: Phase[];
  pure: boolean;
  effects?: string[];
  cost?: OpCost;
  impl?: {
    kind: 'kernel' | 'composite' | 'builtin';
    entry?: string;
    /** 实现代码哈希（进 specHash）。 */
    codeHash?: string;
    /** 依赖版本（进 specHash；Marin 教训，03 §4.1）。 */
    depVersions?: string[];
  };
  /** 需要 RNG 的 op 声明 counter 段（06 §5；plan() 分配不重叠区间）。 */
  rng?: { counterScope: string; shape?: string };
  /** arena 区段读写（用于自动 passBreak）。 */
  writesReads?: OpWritesReads;
}

export function opRef ( c: Pick<OpContract, 'op' | 'version'> ): OpRef
{
  return `${ c.op }@${ c.version }`;
}

/** 契约的版本化指纹输入（specHash 用）。 */
export function contractFingerprint ( c: OpContract ): string
{
  return [
    opRef( c ),
    ( c.impl?.codeHash ?? '' ),
    ( c.impl?.depVersions ?? [] ).join( ',' ),
  ].join( '#' );
}

// ---------------------------------------------------------------------------
// 形状函数（运行时，不进哈希）
// ---------------------------------------------------------------------------

export interface ShapeContext {
  props: Props;
  /** 符号维取值表（`B`→4、`T`→128）；infer 填充。 */
  symbols: Record<string, number | undefined>;
  /** 节点 id，用于诊断。 */
  nodeId: string;
}

export type ShapeFn = (
  ins: Record<string, PortType | null>,
  ctx: ShapeContext,
) => Record<string, PortType>;

/** 一个已注册的 op = 契约（数据）+ 形状函数（运行时）。 */
export interface OpDefinition {
  contract: OpContract;
  /**
   * 输出形状/类型推导。缺省时 infer 只在端口 dtype 已知的情况下做透传，
   * 并对无法确定的符号维报 `DYNAMIC_SHAPE_UNRESOLVED`（warn）。
   */
  shape?: ShapeFn;
  /** 组合原语：把子节点「内联」为父节点语义时的输出（可选）。 */
  doc?: string;
}

// ---------------------------------------------------------------------------
// defineOp：受信逃生门（INV-12）
// ---------------------------------------------------------------------------

/** 一次 op 贡献（05 §4）。进注册表前必须经过对拍 + 审核。 */
export interface OpContribution {
  contract: OpContract;
  shape?: ShapeFn;
  /** 审核记录 id（人类 review）。 */
  reviewId: string;
  /** 实现代码哈希（WGSL/JS 源码哈希）。 */
  codeHash: string;
  /** 参考实现对拍用例名列表（OP-V1）。 */
  referenceTests: string[];
  doc?: string;
}

export interface ContributionAudit {
  ok: boolean;
  problems: string[];
}

/**
 * 审核贡献是否具备进图资格（INV-12）。
 * 只做**结构检查**；数值对拍（OP-V1/OP-V2）由测试层负责。
 */
export function auditContribution ( c: OpContribution ): ContributionAudit
{
  const problems: string[] = [];
  if ( !c.reviewId ) problems.push( '缺少 reviewId：受信逃生门要求人类审核记录' );
  if ( !c.codeHash || c.codeHash.length < 8 ) problems.push( '缺少 codeHash：实现必须内容寻址' );
  if ( !c.referenceTests || c.referenceTests.length === 0 )
    problems.push( '缺少 referenceTests：贡献必须有 CPU 参考对拍用例' );
  if ( c.contract.caps.length === 0 ) problems.push( 'caps 为空：至少声明 forward' );
  if ( !c.contract.caps.includes( 'forward' ) )
    problems.push( 'caps 缺 forward：所有 op 必须可前向求值' );
  if ( c.contract.pure === false && !( c.contract.effects && c.contract.effects.length > 0 ) )
    problems.push( 'effectful op 必须显式列出 effects' );
  if ( c.contract.caps.includes( 'nondiff' ) && c.contract.pure !== false )
    problems.push( 'nondiff op 应为 effectful/离散，需要显式 effects' );
  return { ok: problems.length === 0, problems };
}

// ---------------------------------------------------------------------------
// 注册表
// ---------------------------------------------------------------------------

export class OpRegistry
{
  private readonly byRef = new Map<OpRef, OpDefinition>();
  private readonly byName = new Map<OpName, Set<string>>();

  /** 注册一个已定义的 op。同 opRef 重复注册会覆盖（幂等，便于模块热重载）。 */
  register ( def: OpDefinition ): OpDefinition
  {
    const ref = opRef( def.contract );
    this.byRef.set( ref, def );
    let versions = this.byName.get( def.contract.op );
    if ( !versions )
    {
      versions = new Set();
      this.byName.set( def.contract.op, versions );
    }
    versions.add( def.contract.version );
    return def;
  }

  /** 走受信逃生门注册一次贡献；审核不过则抛（INV-12）。 */
  define ( c: OpContribution ): OpDefinition
  {
    const audit = auditContribution( c );
    if ( !audit.ok )
      throw new Error( `defineOp 被拒（INV-12）：${ c.contract.op } → ${ audit.problems.join( '; ' ) }` );
    return this.register( {
      contract: {
        ...c.contract,
        impl: { ...c.contract.impl, kind: c.contract.impl?.kind ?? 'kernel', codeHash: c.codeHash },
      },
      shape: c.shape,
      doc: c.doc,
    } );
  }

  get ( ref: OpRef ): OpDefinition | undefined
  {
    return this.byRef.get( ref );
  }

  has ( ref: OpRef ): boolean
  {
    return this.byRef.has( ref );
  }

  /** 按名字取**最新**版本（字典序最大）。 */
  latest ( name: OpName ): OpDefinition | undefined
  {
    const versions = this.byName.get( name );
    if ( !versions || versions.size === 0 ) return undefined;
    const ref = [ ...versions ].sort().at( -1 )!;
    return this.byRef.get( `${ name }@${ ref }` );
  }

  names (): OpName[]
  {
    return [ ...this.byName.keys() ].sort();
  }

  all (): OpDefinition[]
  {
    return [ ...this.byRef.values() ];
  }

  get size (): number
  {
    return this.byRef.size;
  }
}

/** 全网内置 op 的注册表（预置 catalog 由 ops.ts 填充）。 */
export const builtinRegistry = new OpRegistry();

/** 便捷构造：少写样板地声明一个契约。 */
export function contract (
  op: OpName,
  version: string,
  spec: Omit<OpContract, 'op' | 'version'>,
): OpContract
{
  return { op, version, ...spec };
}

/** 便捷构造一个内建 op（impl.kind 默认 builtin；可覆盖）。 */
export function builtin (
  c: OpContract,
  shape?: ShapeFn,
  doc?: string,
): OpDefinition
{
  return {
    contract: {
      ...c,
      impl: { kind: 'builtin', ...c.impl },
    },
    shape,
    doc,
  };
}

export type { OpContract as OpContractType };
export type { Effects };
