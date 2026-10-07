/**
 * 第一批内置算子目录（design/05-算子与内核层.md §5）。
 *
 * 分两类：
 *   叶原语（无子节点，直接映射 kernel）+ 组合原语（有语义、无自身参数）。
 * 现有 35 个 WGSL kernel 通过 `impl.entry` 指向；组合原语 `impl.kind='composite'`。
 *
 * 契约中的 io/caps/cost 必须与实现**测试对齐**（P1 逐 op 对拍 + gradcheck，OP-V3）。
 * 这里的 `shape` 函数是运行时类型推导（**不进 specHash**，03 §2.1）。
 */

import { builtin, builtinRegistry, type OpDefinition, type ShapeContext } from './op';
import { propBool, propNumber, propString } from './expr';
import type { Cap, Dim, DType, PortType, TensorType } from './types';

// ---------------------------------------------------------------------------
// 辅助
// ---------------------------------------------------------------------------

const A_FWD: Cap[] = [ 'forward' ];
const A_FWD_VJP: Cap[] = [ 'forward', 'vjp' ];
const A_FWD_VJP_JVP: Cap[] = [ 'forward', 'vjp', 'jvp' ];

const P_ALL = [ 'train', 'prefill', 'decode' ] as const;

function tensor ( shape: Dim[], dtype: DType = 'f32' ): TensorType
{
  return { shape, dtype };
}

function asTensor ( p: PortType | null | undefined ): TensorType | undefined
{
  if ( !p ) return undefined;
  return ( p as TensorType ).shape ? ( p as TensorType ) : undefined;
}

/** 元素级一元：输出形状 = 第一个输入形状。 */
function sameShape ( ins: Record<string, PortType | null> ): Record<string, PortType>
{
  const first = ins.in ?? ins.x ?? ins[ Object.keys( ins )[ 0 ] ];
  const t = asTensor( first );
  return { out: t ? { ...t } : tensor( [ 'B', 'T' ] ) };
}

/** 取符号维的具体值；未绑定则回退该符号本身。 */
function sym ( ctx: ShapeContext, d: Dim ): Dim
{
  if ( typeof d === 'number' ) return d;
  return ctx.symbols[ d ] ?? d;
}

/** 从 props 取维度参数（可能来自 props.dim 或符号）。 */
function dimProp ( ctx: ShapeContext, key: string, fallback: Dim ): Dim
{
  const v = propNumber( ctx.props, key, { symbols: ctx.symbols } );
  return v ?? fallback;
}

// ---------------------------------------------------------------------------
// 叶原语
// ---------------------------------------------------------------------------

function defs (): OpDefinition[]
{
  const list: OpDefinition[] = [];
  const add = ( d: OpDefinition ): void =>
  {
    list.push( d );
    builtinRegistry.register( d );
  };

  // ---- ① 变换 -------------------------------------------------------------

  add( builtin( {
    op: 'Embed', version: '1.0',
    props: {
      vocab: { type: 'int', required: true },
      dim: { type: 'int', required: true },
    },
    io: {
      in: [ { name: 'ids', shape: [ 'B', 'T' ], dtype: 'i32' }, { name: 'weight' } ],
      out: [ { name: 'out', shape: [ 'B', 'T', 'dim' ] } ],
    },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: 'B*T*dim', mem: 'O(vocab*dim)' },
    writesReads: { reads: [ 'weight' ], writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/embedding/embedding.wgsl' },
  }, ( ins, ctx ) =>
  {
    const ids = asTensor( ins.ids );
    const dim = dimProp( ctx, 'dim', 'dim' );
    return { out: tensor( [ ...( ids?.shape ?? [ 'B', 'T' ] ), sym( ctx, dim ) ] ) };
  }, 'token 索引 → 嵌入向量（embed_gather）' ) );

  add( builtin( {
    op: 'Matmul', version: '1.0',
    props: {
      transposeA: { type: 'bool', default: false },
      transposeB: { type: 'bool', default: false },
      hasBias: { type: 'bool', default: false },
      /** 输出特征数；权重经 props.bind 绑定时 infer 靠它推出形状。 */
      n: { type: 'int' },
    },
    io: {
      in: [ { name: 'a' }, { name: 'b' }, { name: 'bias', optional: true } ],
      out: [ { name: 'out' } ],
    },
    caps: A_FWD_VJP_JVP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '2*B*M*N*K', mem: 'O(B*M*N + M*K + K*N)' },
    writesReads: { reads: [ 'weight' ], writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/gemm/gemm_nt.wgsl' },
  }, ( ins, ctx ) =>
  {
    const a = asTensor( ins.a )?.shape ?? [ 'B', 'T', 'K' ];
    const b = asTensor( ins.b )?.shape;
    const tB = propBool( ctx.props, 'transposeB', { symbols: ctx.symbols } );
    // 输出特征数：优先 `props.n`（权重经 props.bind 绑定时 infer 看不到它的形状），
    // 否则退回第二个操作数的形状。输出形状 = A 的前缀 + [n]，保持秩不变。
    const nFromB: Dim | undefined = b ? ( tB ? b[ 0 ] : b[ b.length - 1 ] ) : undefined;
    const nDim = dimProp( ctx, 'n', nFromB ?? 'N' );
    const out = a.slice();
    out[ out.length - 1 ] = sym( ctx, nDim );
    return { out: tensor( out ) };
  }, '通用矩阵乘（NT/NN/TN 三个 kernel 共享契约；输出形状 = A 前缀 + props.n）' ) );

  add( builtin( {
    op: 'RMSNorm', version: '1.0',
    props: { dim: { type: 'int', required: true }, eps: { type: 'float', default: 1e-5 } },
    io: { in: [ { name: 'x' }, { name: 'weight' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '6*B*T*dim', mem: 'O(B*T*dim)' },
    writesReads: { reads: [ 'weight' ], writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/norm/rmsnorm.wgsl' },
  }, sameShape, 'RMS 归一化' ) );

  add( builtin( {
    op: 'LayerNorm', version: '1.0',
    props: {
      dim: { type: 'int', required: true },
      eps: { type: 'float', default: 1e-5 },
      elementwiseAffine: { type: 'bool', default: true },
    },
    io: { in: [ { name: 'x' }, { name: 'weight', optional: true }, { name: 'bias', optional: true } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '8*B*T*dim', mem: 'O(B*T*dim)' },
    writesReads: { reads: [ 'weight' ], writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/norm/layernorm.wgsl' },
  }, sameShape, '层归一化（ln_stats/ln_dx/ln_dwdb 组合）' ) );

  add( builtin( {
    op: 'GELU', version: '1.0',
    props: { approx: { type: 'enum', values: [ 'tanh', 'erf' ], default: 'tanh' } },
    io: { in: [ { name: 'x' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '8*N', mem: 'O(N)' },
    writesReads: { writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/activation/gelu.wgsl' },
  }, sameShape, 'GELU 激活' ) );

  add( builtin( {
    op: 'SiLU', version: '1.0',
    props: { gate: { type: 'bool', default: false } },
    io: { in: [ { name: 'x' }, { name: 'gate', optional: true } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '5*N', mem: 'O(N)' },
    writesReads: { writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/activation/silu_mul.wgsl' },
  }, ( ins ) =>
  {
    const x = asTensor( ins.x ) ?? tensor( [ 'B', 'T', 'dim' ] );
    const g = asTensor( ins.gate );
    return { out: g ? tensor( x.shape, x.dtype ) : x };
  }, 'SiLU / SwiGLU 门控（silu_mul）' ) );

  add( builtin( {
    op: 'Softmax', version: '1.0',
    props: { axis: { type: 'int', default: -1 }, causal: { type: 'bool', default: false } },
    io: { in: [ { name: 'x' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '5*N', mem: 'O(N)' },
    writesReads: { reads: [ 'act' ], writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/attention/softmax.wgsl' },
  }, sameShape, 'Softmax（可带 causal mask）' ) );

  add( builtin( {
    op: 'Reshape', version: '1.0',
    props: { shape: { type: 'dims', required: true } },
    io: { in: [ { name: 'x' } ], out: [ { name: 'out' } ] },
    caps: [ 'forward', 'vjp', 'jvp', 'invertible', 'perturbable' ], phase: [ ...P_ALL ], pure: true,
    cost: { flops: '0', mem: 'O(1)' },
    writesReads: { writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/misc/vec_add.wgsl' },
  }, ( ins, ctx ) =>
  {
    const raw = ctx.props.shape;
    const dims = ( Array.isArray( raw ) ? raw : [ 'B', 'T' ] ) as Dim[];
    return { out: tensor( dims.map( ( d ) => sym( ctx, d ) ), asTensor( ins.x )?.dtype ?? 'f32' ) };
  }, '形变（零拷贝视图：可微、可逆，故 caps 无 nondiff）' ) );

  add( builtin( {
    op: 'Transpose', version: '1.0',
    props: { perm: { type: 'dims', default: [ 1, 0 ] } },
    io: { in: [ { name: 'x' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP_JVP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '0', mem: 'O(N)' },
    writesReads: { writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/misc/vec_add.wgsl' },
  }, ( ins, ctx ) =>
  {
    const s = asTensor( ins.x )?.shape ?? [ 'B', 'T' ];
    const perm = Array.isArray( ctx.props.perm ) ? ( ctx.props.perm as number[] ) : [ 1, 0 ];
    return { out: tensor( perm.map( ( i ) => s[ i ] ?? i ), asTensor( ins.x )?.dtype ?? 'f32' ) };
  }, '转置' ) );

  add( builtin( {
    op: 'Cast', version: '1.0',
    props: { dtype: { type: 'enum', values: [ 'f32', 'f16', 'bf16', 'i32' ], required: true } },
    io: { in: [ { name: 'x' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ 'prefill', 'decode' ], pure: true,
    cost: { flops: '0', mem: 'O(N)' },
    writesReads: { writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/misc/vec_add.wgsl' },
  }, ( ins, ctx ) =>
  {
    const x = asTensor( ins.x );
    const dt = ( propString( ctx.props, 'dtype', { symbols: ctx.symbols }, 'f32' ) ?? 'f32' ) as DType;
    return { out: tensor( x?.shape ?? [ 'B', 'T' ], dt ) };
  }, '类型转换（仅推理相位；训练禁用 f16，INV-1）' ) );

  add( builtin( {
    op: 'Concat', version: '1.0',
    props: { axis: { type: 'int', default: -1 } },
    io: { in: [ { name: 'a' }, { name: 'b' } ], out: [ { name: 'out' } ] },
    caps: [ 'forward', 'vjp', 'jvp' ], phase: [ ...P_ALL ], pure: true,
    cost: { flops: '0', mem: 'O(N)' },
    writesReads: { writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/misc/vec_add.wgsl' },
  }, ( ins, ctx ) =>
  {
    const a = asTensor( ins.a )?.shape ?? [ 'B', 'T', 'd' ];
    const b = asTensor( ins.b )?.shape ?? a;
    const axisRaw = propNumber( ctx.props, 'axis', { symbols: ctx.symbols }, -1 ) ?? -1;
    const axis = axisRaw < 0 ? a.length + axisRaw : axisRaw;
    const out = a.slice();
    out[ axis ] = `${ a[ axis ] }+${ b[ axis ] }`;
    return { out: tensor( out ) };
  }, '拼接（轴维记符号和，交由 infer 解析）' ) );

  add( builtin( {
    op: 'Split', version: '1.0',
    props: { axis: { type: 'int', default: -1 }, sizes: { type: 'dims', required: true } },
    io: { in: [ { name: 'x' } ], out: [ { name: 'a' }, { name: 'b' } ] },
    caps: [ 'forward', 'vjp', 'jvp' ], phase: [ ...P_ALL ], pure: true,
    cost: { flops: '0', mem: 'O(N)' },
    writesReads: { writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/misc/vec_add.wgsl' },
  }, ( ins, ctx ) =>
  {
    const x = asTensor( ins.x )?.shape ?? [ 'B', 'T', 'd' ];
    const axisRaw = propNumber( ctx.props, 'axis', { symbols: ctx.symbols }, -1 ) ?? -1;
    const axis = axisRaw < 0 ? x.length + axisRaw : axisRaw;
    const sizes = Array.isArray( ctx.props.sizes ) ? ( ctx.props.sizes as Dim[] ) : [ x[ axis ], x[ axis ] ];
    const a = x.slice(); a[ axis ] = sizes[ 0 ];
    const b = x.slice(); b[ axis ] = sizes[ 1 ] ?? x[ axis ];
    return { a: tensor( a ), b: tensor( b ) };
  }, '切分' ) );

  add( builtin( {
    op: 'Gather', version: '1.0',
    props: { axis: { type: 'int', default: 0 } },
    io: { in: [ { name: 'x' }, { name: 'index', dtype: 'i32' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '0', mem: 'O(N)' },
    writesReads: { reads: [ 'weight' ], writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/embedding/embedding.wgsl' },
  }, ( ins, ctx ) =>
  {
    const idx = asTensor( ins.index )?.shape ?? [ 'B', 'T' ];
    const x = asTensor( ins.x )?.shape ?? [ 'V', 'd' ];
    const axis = propNumber( ctx.props, 'axis', { symbols: ctx.symbols }, 0 ) ?? 0;
    const rest = x.filter( ( _, i ) => i !== axis );
    return { out: tensor( [ ...idx, ...rest ] ) };
  }, '按索引取行（反向为 scatter-add，需两段式规约）' ) );

  add( builtin( {
    op: 'Scatter', version: '1.0',
    props: { axis: { type: 'int', default: 0 }, reduction: { type: 'enum', values: [ 'add', 'set' ], default: 'add' } },
    io: { in: [ { name: 'x' }, { name: 'index', dtype: 'i32' }, { name: 'src' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '0', mem: 'O(N)' },
    writesReads: { reads: [ 'weight' ], writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/embedding/embedding_bwd.wgsl' },
  }, ( ins ) => ( { out: asTensor( ins.x ) ?? tensor( [ 'V', 'd' ] ) } ), '散射累加（embed_scatter_add）' ) );

  add( builtin( {
    op: 'QKNorm', version: '1.0',
    props: { dim: { type: 'int', required: true }, eps: { type: 'float', default: 1e-6 } },
    io: { in: [ { name: 'q' }, { name: 'k' } ], out: [ { name: 'qOut' }, { name: 'kOut' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '6*B*H*T*d', mem: 'O(B*H*T*d)' },
    writesReads: { writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/norm/rmsnorm.wgsl' },
  }, ( ins ) => ( { qOut: asTensor( ins.q ) ?? tensor( [ 'B', 'H', 'T', 'd' ] ), kOut: asTensor( ins.k ) ?? tensor( [ 'B', 'H', 'T', 'd' ] ) } ),
    'QK-Norm：抑制注意力 logits 爆炸（05 §5.3，Marin 32B 实证）' ) );

  add( builtin( {
    op: 'RoPE', version: '1.0',
    props: {
      heads: { type: 'int', required: true },
      headDim: { type: 'int', required: true },
      base: { type: 'float', default: 10000 },
      seqLen: { type: 'int', default: 0 },
    },
    io: { in: [ { name: 'x', shape: [ 'B', 'T', 'H', 'D'] } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: 'O(B*T*H*D)', mem: 'O(B*T*H*D)' },
    writesReads: { writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/rope/rope.wgsl' },
  }, sameShape, '旋转位置编码（GPT-NeoX 相邻成对；rope/rope_half 两个 kernel 共享契约）' ) );

  add( builtin( {
    op: 'Attention', version: '1.0',
    props: {
      heads: { type: 'int', required: true },
      headDim: { type: 'int', required: true },
      causal: { type: 'bool', default: true },
      scale: { type: 'float', default: 0 },
    },
    io: { in: [ { name: 'q' }, { name: 'k' }, { name: 'v' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '4*B*H*T*T*D', mem: 'O(B*H*T*T)' },
    writesReads: { reads: [ 'act' ], writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/attention/attention.wgsl' },
  }, ( ins ) => ( { out: asTensor( ins.q ) ?? tensor( [ 'B', 'T', 'H', 'D' ] ) } ),
    '因果自注意力（q/k/v 形状 [B*T, H, D]；attention/attention_gqa 共享契约）' ) );

  add( builtin( {
    op: 'CrossEntropy', version: '1.0',
    props: { reduction: { type: 'enum', values: [ 'mean', 'sum' ], default: 'mean' } },
    io: { in: [ { name: 'logits' }, { name: 'targets', dtype: 'i32' } ], out: [ { name: 'loss' } ] },
    caps: [ 'forward', 'vjp' ], phase: [ 'train' ], pure: true,
    cost: { flops: 'O(M*V)', mem: 'O(M*V)' },
    writesReads: { reads: [ 'act' ], writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/misc/ce_softmax_bwd.wgsl' },
  }, () => ( { loss: tensor( [], 'f32' ) } ), '交叉熵损失（前向出标量，反向出 dlogits；与 ce_softmax_bwd 共用实现）' ) );

  add( builtin( {
    op: 'ZLoss', version: '1.0',
    props: { coef: { type: 'float', default: 1e-4 } },
    io: { in: [ { name: 'logits' } ], out: [ { name: 'loss' } ] },
    caps: [ 'forward', 'vjp', 'jvp' ], phase: [ 'train' ], pure: true,
    cost: { flops: '2*N', mem: 'O(1)' },
    impl: { kind: 'kernel', entry: 'kernels/misc/ce_softmax_bwd.wgsl' },
  }, () => ( { loss: tensor( [], 'f32' ) } ), 'Z-Loss：logits 平方惩罚（防爆炸，05 §5.3）' ) );

  add( builtin( {
    op: 'Quant', version: '1.0',
    props: { scheme: { type: 'enum', values: [ 'int4', 'int8', 'nf4' ], default: 'int4' }, groupSize: { type: 'int', default: 128 } },
    io: { in: [ { name: 'x' } ], out: [ { name: 'out' }, { name: 'scale' } ] },
    caps: [ 'forward', 'nondiff' ], phase: [ 'prefill', 'decode' ], pure: true,
    cost: { flops: 'O(N)', mem: 'O(N/2)' },
    writesReads: { writes: [ 'weight' ] },
    impl: { kind: 'kernel', entry: 'kernels/misc/vec_add.wgsl' },
  }, ( ins ) => ( { out: asTensor( ins.x ) ?? tensor( [ 'V', 'd' ] ), scale: tensor( [ 'G', 'd' ] ) } ),
    '量化（非可微；不进训练路径）' ) );

  add( builtin( {
    op: 'Dequant', version: '1.0',
    props: { scheme: { type: 'enum', values: [ 'int4', 'int8', 'nf4' ], default: 'int4' }, groupSize: { type: 'int', default: 128 } },
    io: { in: [ { name: 'x' }, { name: 'scale' } ], out: [ { name: 'out' } ] },
    caps: [ 'forward', 'nondiff' ], phase: [ 'prefill', 'decode' ], pure: true,
    cost: { flops: 'O(N)', mem: 'O(N)' },
    writesReads: { reads: [ 'weight' ], writes: [ 'act' ] },
    impl: { kind: 'kernel', entry: 'kernels/misc/vec_add.wgsl' },
  }, ( ins ) => ( { out: tensor( asTensor( ins.scale )?.shape ?? [ 'V', 'd' ] ) } ), '反量化（GPTQ int4 加载）' ) );

  // ---- ⑤ 随机 -------------------------------------------------------------

  add( builtin( {
    op: 'Dist', version: '1.0',
    props: {
      family: { type: 'enum', values: [ 'categorical', 'gaussian', 'bernoulli' ], default: 'categorical' },
      temperature: { type: 'float', default: 1 },
    },
    io: { in: [ { name: 'logits' } ], out: [ { name: 'dist' } ] },
    caps: [ 'forward', 'logprob' ], phase: [ 'sample', 'rollout' ], pure: true,
    effects: [ 'rng' ],
    rng: { counterScope: 'dist', shape: 'B*T' },
    cost: { flops: '5*N', mem: 'O(N)' },
    impl: { kind: 'kernel', entry: 'kernels/attention/softmax.wgsl' },
  }, ( ins ) => ( { dist: asTensor( ins.logits ) ?? tensor( [ 'B', 'T', 'V' ] ) } ), '分布对象，提供 logprob()' ) );

  add( builtin( {
    op: 'Sample', version: '1.0',
    props: { mode: { type: 'enum', values: [ 'greedy', 'topk', 'topp' ], default: 'greedy' }, k: { type: 'int', default: 50 }, p: { type: 'float', default: 0.9 } },
    io: { in: [ { name: 'dist' } ], out: [ { name: 'ids', dtype: 'i32' } ] },
    caps: [ 'forward', 'logprob', 'nondiff' ], phase: [ 'decode', 'sample' ], pure: true,
    effects: [ 'rng' ],
    rng: { counterScope: 'sample', shape: 'B' },
    cost: { flops: '2*N', mem: 'O(N)' },
    impl: { kind: 'builtin' },
  }, ( ins ) =>
  {
    const d = asTensor( ins.dist )?.shape ?? [ 'B', 'T', 'V' ];
    return { ids: tensor( d.slice( 0, -1 ), 'i32' ) };
  }, '采样（重参数化/Gumbel；需确定性 RNG，INV-10）' ) );

  add( builtin( {
    op: 'Discretize', version: '1.0',
    props: { straightThrough: { type: 'bool', default: true } },
    io: { in: [ { name: 'p' } ], out: [ { name: 'y' } ] },
    caps: [ 'forward', 'vjp' ], phase: [ 'train', 'sample' ], pure: true, effects: [ 'rng' ],
    rng: { counterScope: 'discretize', shape: 'B*T' },
    cost: { flops: 'O(N)', mem: 'O(N)' },
    impl: { kind: 'builtin' },
  }, sameShape, '离散化（前向不可微；Straight-Through 提供代理 vjp）' ) );

  // ---- ⑥ 循环 -------------------------------------------------------------

  add( builtin( {
    op: 'Scan', version: '1.0',
    props: { kind: { type: 'enum', values: [ 'cumsum', 'cummax', 'associative' ], default: 'cumsum' }, axis: { type: 'int', default: 0 } },
    io: { in: [ { name: 'x' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: 'O(N*T)', mem: 'O(N*T)' },
    writesReads: { writes: [ 'state' ] },
    impl: { kind: 'builtin' },
  }, sameShape, '解释档扫描' ) );

  add( builtin( {
    op: 'FusedScan', version: '1.0',
    props: { kind: { type: 'enum', values: [ 'cumsum', 'cummax', 'associative' ], default: 'cumsum' }, axis: { type: 'int', default: 0 } },
    io: { in: [ { name: 'x' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: 'O(N*logT)', mem: 'O(N)' },
    writesReads: { writes: [ 'state' ] },
    impl: { kind: 'kernel', entry: 'kernels/misc/sum_rows.wgsl' },
  }, sameShape, '融合档扫描（并行前缀）' ) );

  add( builtin( {
    op: 'Solve', version: '1.0',
    props: { method: { type: 'enum', values: [ 'fixed-point', 'ode', 'argmin' ], default: 'fixed-point' }, iters: { type: 'int', default: 20 }, tol: { type: 'float', default: 1e-5 } },
    io: { in: [ { name: 'f' }, { name: 'x0' } ], out: [ { name: 'x' } ] },
    caps: [ 'forward', 'invertible', 'vjp', 'localGrad' ], phase: [ 'train' ], pure: true,
    cost: { flops: 'iters*O(N)', mem: 'O(N)' },
    writesReads: { writes: [ 'state' ] },
    impl: { kind: 'builtin' },
  }, ( ins ) => ( { x: asTensor( ins.x0 ) ?? tensor( [ 'B', 'T' ] ) } ), '隐式求解（隐式微分，非显式反传）' ) );

  // ---- ⑦ 交互（effectful） -----------------------------------------------

  add( builtin( {
    op: 'Env', version: '1.0',
    props: { name: { type: 'string', required: true }, maxSteps: { type: 'int', default: 0 } },
    io: { in: [ { name: 'action' } ], out: [ { name: 'obs' }, { name: 'reward' } ] },
    caps: [ 'forward', 'nondiff' ], phase: [ 'rollout' ], pure: false,
    effects: [ 'env', 'nondeterministic' ],
    cost: { mem: 'O(1)' },
    impl: { kind: 'builtin' },
  }, () => ( { obs: tensor( [ 'B', 'H' ] ), reward: tensor( [ 'B' ] ) } ), '环境交互（黑盒，无梯度）' ) );

  add( builtin( {
    op: 'Tool', version: '1.0',
    props: { name: { type: 'string', required: true }, schema: { type: 'string', default: '' } },
    io: { in: [ { name: 'input' } ], out: [ { name: 'output' } ] },
    caps: [ 'forward', 'nondiff' ], phase: [ 'rollout' ], pure: false,
    effects: [ 'io', 'external' ],
    cost: { mem: 'O(1)' },
    impl: { kind: 'builtin' },
  }, () => ( { output: tensor( [ 'B', 'T' ] ) } ), '外部工具调用（effectful，禁入纯子图）' ) );

  // ---- 其余叶原语 ---------------------------------------------------------

  add( builtin( {
    op: 'Einsum', version: '1.0',
    props: { equation: { type: 'string', required: true } },
    io: { in: [ { name: 'a' }, { name: 'b', optional: true } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP_JVP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '取决于 equation' },
    writesReads: { reads: [ 'weight' ], writes: [ 'act' ] },
    impl: { kind: 'builtin' },
  }, ( _ins, ctx ) => ( { out: tensor( resolveEinsumOut( String( ctx.props.equation ?? 'ij->ij' ) ) ) } ),
    '爱因斯坦求和（编译期降为 Matmul/Transpose）' ) );

  add( builtin( {
    op: 'Conv2d', version: '1.0',
    props: { kernel: { type: 'dims', default: [ 3, 3 ] }, stride: { type: 'dims', default: [ 1, 1 ] }, padding: { type: 'dims', default: [ 1, 1 ] }, outChannels: { type: 'int', required: true } },
    io: { in: [ { name: 'x' }, { name: 'w' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '2*B*H*W*C*K*K*Cout' },
    writesReads: { reads: [ 'weight' ], writes: [ 'act' ] },
    impl: { kind: 'builtin' },
  }, ( ins, ctx ) =>
  {
    const x = asTensor( ins.x )?.shape ?? [ 'B', 'C', 'H', 'W' ];
    const oc = dimProp( ctx, 'outChannels', 'Cout' );
    return { out: tensor( [ x[ 0 ], sym( ctx, oc ), 'H', 'W' ] ) };
  }, '2D 卷积' ) );

  add( builtin( {
    op: 'Pool', version: '1.0',
    props: { kind: { type: 'enum', values: [ 'max', 'avg' ], default: 'max' }, kernel: { type: 'dims', default: [ 2, 2 ] }, stride: { type: 'dims', default: [ 2, 2 ] } },
    io: { in: [ { name: 'x' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: 'O(N)' },
    writesReads: { writes: [ 'act' ] },
    impl: { kind: 'builtin' },
  }, sameShape, '池化' ) );

  add( builtin( {
    op: 'PatchEmbed', version: '1.0',
    props: { patch: { type: 'int', default: 16 }, dim: { type: 'int', required: true } },
    io: { in: [ { name: 'image' } ], out: [ { name: 'out' } ] },
    caps: A_FWD_VJP, phase: [ ...P_ALL ], pure: true,
    cost: { flops: '2*B*N*dim*P*P*C' },
    writesReads: { reads: [ 'weight' ], writes: [ 'act' ] },
    impl: { kind: 'composite' },
  }, ( _ins, ctx ) => ( { out: tensor( [ 'B', 'N', sym( ctx, dimProp( ctx, 'dim', 'dim' ) ) ] ) } ), 'ViT 式 patch 嵌入（Conv2d 特例）' ) );

  // ---- Probe：统一的能力/质量探针（14 §7.2；11 §2.4） ----------------------

  add( builtin( {
    op: 'Probe', version: '1.0',
    props: {
      target: { type: 'enum', values: [ 'model', 'node' ], default: 'model' },
      kind: { type: 'enum', values: [ 'atomic', 'backdoor' ], default: 'atomic' },
      /** 探针池 item 数（INV-21 的统计功效输入）。 */
      poolItems: { type: 'int', default: 0 },
      /** 声明要检出的原子能力差（用于 MDE 反算所需池大小）。 */
      effect: { type: 'float', default: 0 },
      /** backdoor 探针在正常输入上是否检出触发器响应。 */
      detected: { type: 'bool', default: false },
    },
    io: { in: [ { name: 'subject' } ], out: [ { name: 'result' } ] },
    caps: [ 'forward', 'nondiff' ], phase: [ 'free' ], pure: false,
    effects: [ 'eval' ],
    cost: { mem: 'O(poolItems)' },
    impl: { kind: 'builtin' },
  }, () => ( { result: tensor( [ 'P' ] ) } ), '内容寻址、可复算的探针（验收即探针的原语）' ) );

  // ---- 组合原语 -----------------------------------------------------------

  add( composite( 'Seq', {}, A_FWD_VJP_JVP, [ ...P_ALL ], ( ins ) =>
  {
    // Seq 的输出 = 最后一个子节点的输出，infer 会做端口透传；这里给个宽松上限。
    return { out: passthrough( ins ) };
  }, '顺序组合：`x → f1 → f2 → …`' ) );

  add( composite( 'Fan', { mode: { type: 'enum', values: [ 'sum', 'mean', 'concat' ], default: 'sum' } }, A_FWD_VJP_JVP, [ ...P_ALL ], ( ins ) =>
  {
    return { out: passthrough( ins ) };
  }, '多分支扇出（Residual 的原型）' ) );

  add( composite( 'Residual', { scale: { type: 'float', default: 1 } }, A_FWD_VJP_JVP, [ ...P_ALL ], ( ins ) =>
  {
    return { out: passthrough( ins ) };
  }, 'Fan(sum) 的特例：x + f(x)' ) );

  add( composite( 'Bus', { provide: { type: 'string', required: true }, inject: { type: 'string', default: '' } }, A_FWD, [ ...P_ALL ], ( ins ) =>
  {
    return { out: passthrough( ins ) };
  }, 'provide/inject 侧通道（infer 报 BUS_CYCLE/TRAP_PATH）' ) );

  add( composite( 'Map', { axis: { type: 'int', default: 0 } }, A_FWD_VJP_JVP, [ ...P_ALL ], ( ins ) =>
  {
    return { out: passthrough( ins ) };
  }, '沿轴广播映射' ) );

  add( composite( 'Gate', { topk: { type: 'int', default: 1 }, balance: { type: 'enum', values: [ 'aux-loss', 'quantile-bias' ], default: 'quantile-bias' }, capacityFactor: { type: 'float', default: 1.15 } }, A_FWD_VJP, [ ...P_ALL ], ( ins ) =>
  {
    return { out: passthrough( ins ) };
  }, 'MoE 路由门（top-k + 超容丢弃；quantile-bias 免辅助损失）' ) );

  add( composite( 'Repeat', { times: { type: 'int', required: true } }, A_FWD_VJP_JVP, [ ...P_ALL ], ( ins ) =>
  {
    return { out: passthrough( ins ) };
  }, '重复应用（等价于 for 循环展开）' ) );

  add( composite( 'Memory', { size: { type: 'int', required: true } }, A_FWD_VJP, [ ...P_ALL ], ( ins ) =>
  {
    return { out: passthrough( ins ) };
  }, '外部记忆（State 生命周期显式）' ) );

  add( composite( 'AdaLn', { dim: { type: 'int', required: true } }, A_FWD_VJP, [ ...P_ALL ], ( ins ) =>
  {
    return { out: passthrough( ins ) };
  }, '自适应层归一化（条件缩放/平移）' ) );

  add( composite( 'CrossAttn', { heads: { type: 'int', required: true }, dim: { type: 'int', required: true } }, A_FWD_VJP, [ ...P_ALL ], ( ins ) =>
  {
    return { out: passthrough( ins ) };
  }, '交叉注意力（cond 从 slot 注入）' ) );

  add( composite( 'Cache', { kind: { type: 'enum', values: [ 'kv', 'scan' ], default: 'kv' } }, A_FWD, [ 'prefill', 'decode' ], ( ins ) =>
  {
    return { out: passthrough( ins ) };
  }, 'KV / Scan 状态缓存（必须声明 phase，否则 PHASE_LEAK）', [ 'state' ] ) );

  add( composite( 'Rollout', { horizon: { type: 'int', default: 1 } }, A_FWD, [ 'rollout' ], ( ins ) =>
  {
    return { out: passthrough( ins ) };
  }, '轨迹展开（采样相位；配合 nondiff Env）' ) );

  add( composite( 'If', { test: { type: 'string', required: true } }, A_FWD_VJP_JVP, [ ...P_ALL ], ( ins ) =>
  {
    return { out: passthrough( ins ) };
  }, 'build 期条件（生成不同子图）' ) );

  return list;
}

/** 组合原语的统一构造（caps 是所辖子图能力的上界，须由子节点满足）。 */
function composite (
  op: string,
  props: Record<string, import('./op').OpPropSpec>,
  caps: Cap[],
  phase: readonly string[],
  shape: ( ins: Record<string, PortType | null> ) => Record<string, PortType>,
  doc: string,
  statePorts: string[] = [],
): OpDefinition
{
  return builtin( {
    op,
    version: '1.0',
    props,
    io: {
      in: statePorts.length > 0 ? [ { name: 'x' }, ...statePorts.map( ( s ) => ( { name: s, optional: true } ) ) ] : [ { name: 'x' } ],
      out: [ { name: 'out' } ],
    },
    caps,
    phase: phase as import('./types').Phase[],
    pure: true,
    cost: { mem: 'O(子树)' },
    impl: { kind: 'composite' },
  }, shape, doc );
}

function passthrough ( ins: Record<string, PortType | null> ): PortType
{
  return ins.x ?? ins.in ?? { shape: [ 'B', 'T', 'd' ], dtype: 'f32' };
}

/** 极简 einsum 输出推导：`...->out` 右侧拆成维度名。 */
function resolveEinsumOut ( eq: string ): Dim[]
{
  const rhs = eq.includes( '->' ) ? eq.split( '->' )[ 1 ] : eq.split( ',' )[ 0 ];
  return rhs.trim().split( '' ).filter( ( c ) => /[A-Za-z]/.test( c ) );
}

// ---------------------------------------------------------------------------
// 注册与导出
// ---------------------------------------------------------------------------

let initialized = false;

/** 幂等地把内置算子灌进 `builtinRegistry`。 */
export function ensureBuiltinOps (): void
{
  if ( initialized ) return;
  initialized = true;
  defs();
}

ensureBuiltinOps();

/** 全部内置算子定义。 */
export function builtinOpList (): OpDefinition[]
{
  return builtinRegistry.all();
}

export { builtinRegistry };
