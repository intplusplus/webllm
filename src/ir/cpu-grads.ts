/**
 * CPU 反向（VJP）实现表（design/06 §4：`Strategy = backprop` 的第二刀）。
 *
 * 与正向的 `cpu-impls.ts` 完全对称：正向把「端口 / props 契约」翻成参考实现，
 * 这里把「输出梯度 → 输入端口梯度」翻成参考实现。
 *
 * 三条硬约定（见 grad.ts 的驱动逻辑）：
 *   1. 参数梯度走**端口**：只返回 `dIns[<端口名>]`，驱动器按 `ctx.sources[port]`
 *      自动累加到 `param` 或原路回流到上游节点——所以绝不能猜参数名字；
 *   2. 组合原语的梯度走 `dChildren`（按 children 顺序，元素可为 null）；
 *   3. 每个返回的梯度都与**正向对应张量同长度**，shape 也用它的 shape。
 *
 * 这里刻意不做全局单例：`builtinCpuGrads()` 每次新建一份，避免测试 / 多模型并行互相污染。
 */

import {
  createGradRegistry,
  asF32,
  type GradContext,
  type GradRegistry,
} from './grad';
import { f32, numel, type TensorValue } from './binding';
import { propNumber, propString, type ExprEnv } from './expr';
import {
  attentionBwdRef,
  ceSoftmaxBwdRef,
  embeddingBwdRef,
  geluBwdRef,
  gemmNNRef,
  gemmTNRef,
  layernormBwdRef,
  ropeBwdRef,
  sumRowsRef,
} from '../reference/ops';

// ---------------------------------------------------------------------------
// 内部辅助
// ---------------------------------------------------------------------------

/** 为 `propNumber/propString` 造求值环境（符号表 + 当前 props）。 */
function exprEnv ( ctx: GradContext ): ExprEnv
{
  return { symbols: ctx.symbols, props: ctx.props };
}

/** 任一 op 的必填数字 prop；缺失或非数字即抛（避免静默按 0 算错）。 */
function needNumber ( ctx: GradContext, key: string, who: string ): number
{
  const v = propNumber( ctx.props, key, exprEnv( ctx ) );
  if ( v === undefined || Number.isNaN( v ) )
    throw new Error( `${ who } 反向：props.${ key } 缺失或非数字` );
  return v;
}

/** 张量数据统一读成 Float32Array（整型数组逐元素转换，不改原张量）。 */
function toF32 ( t: TensorValue ): Float32Array
{
  if ( t.data instanceof Float32Array ) return t.data;
  const out = new Float32Array( t.data.length );
  for ( let i = 0; i < t.data.length; i++ ) out[ i ] = t.data[ i ];
  return out;
}

/** token / targets 索引统一成 Uint32Array：源可能是 Float32 存的整数，先四舍五入。 */
function toU32 ( t: TensorValue ): Uint32Array
{
  if ( t.data instanceof Uint32Array ) return t.data;
  const out = new Uint32Array( t.data.length );
  for ( let i = 0; i < t.data.length; i++ ) out[ i ] = Math.round( t.data[ i ] );
  return out;
}

/**
 * 用**正向对应张量**的形状包装一段梯度数据。
 * 长度必须与 `ref` 的元素数一致——不一致说明正反向维度不配套，必须报错而不是硬塞。
 */
function pack ( data: Float32Array, ref: TensorValue, who: string ): TensorValue
{
  const want = numel( ref.shape );
  if ( data.length !== want )
    throw new Error( `${ who }：梯度长度 ${ data.length } 与正向形状 [${ ref.shape.join( ',' ) }]=${ want } 不匹配` );
  return f32( data, ref.shape.slice() );
}

/** 取某个输入端口：端口值优先，其次 `props.bind` 解析出的同名参数。 */
function portOrParam ( ctx: GradContext, port: string ): TensorValue | undefined
{
  return ctx.ins[ port ] ?? ctx.params[ port ] ?? undefined;
}

/**
 * 沿**最后一维**把输出梯度切成各分支（Concat / Fan(concat) 反向共用）。
 * 切出的每片 shape 用对应分支的正向 shape，保证与子节点梯度同形。
 */
function splitLast ( dOut: TensorValue, branches: TensorValue[], who: string ): TensorValue[]
{
  if ( branches.length === 0 ) throw new Error( `${ who } 反向：没有可切分的分支` );
  const rank = branches[ 0 ].shape.length;
  for ( const t of branches )
    if ( t.shape.length !== rank )
      throw new Error( `${ who } 反向：分支秩不一致（${ rank } vs ${ t.shape.length }）` );
  const outer = numel( branches[ 0 ].shape.slice( 0, rank - 1 ) );
  const widths = branches.map( ( t ) => t.shape[ rank - 1 ] );
  const total = widths.reduce( ( a, b ) => a + b, 0 );
  const src = toF32( dOut );
  if ( src.length !== outer * total )
    throw new Error(
      `${ who } 反向：输出梯度元素数 ${ src.length } 与分支末维和 ${ outer }×${ total } 不匹配`,
    );
  const pieces = branches.map( ( t ) => new Float32Array( numel( t.shape ) ) );
  for ( let o = 0; o < outer; o++ )
  {
    let off = o * total;
    for ( let j = 0; j < branches.length; j++ )
    {
      const w = widths[ j ];
      pieces[ j ].set( src.subarray( off, off + w ), o * w );
      off += w;
    }
  }
  return branches.map( ( t, j ) => f32( pieces[ j ], t.shape.slice() ) );
}

// ---------------------------------------------------------------------------
// 实现表
// ---------------------------------------------------------------------------

/** 正向输出 = 最后一个子节点输出的透传组（语义在 emit / infer 阶段已落到子图）。 */
const PASSTHROUGH = [
  'Seq', 'Repeat', 'If', 'Map', 'Bus', 'Gate', 'Memory', 'AdaLn',
  'CrossAttn', 'Rollout', 'Cache',
];

/**
 * 构造一份全新的 CPU 反向实现表。刻意不做全局可变单例。
 */
export function builtinCpuGrads (): GradRegistry
{
  const reg = createGradRegistry();

  // ---- ① 变换 -------------------------------------------------------------

  reg.register( 'Embed', ( ctx, dOut ) =>
  {
    const weight = portOrParam( ctx, 'weight' );
    const ids = ctx.ins.ids;
    const dy = dOut.out;
    if ( !weight ) throw new Error( 'Embed 反向：缺少权重 weight' );
    if ( !ids ) throw new Error( 'Embed 反向：缺少 ids' );
    if ( !dy ) throw new Error( 'Embed 反向：缺少输出梯度 out' );
    const D = weight.shape[ 1 ];
    if ( D === undefined ) throw new Error( `Embed 反向：权重应为 [V,D]，实际 [${ weight.shape.join( ',' ) }]` );
    // scatter-add：dW[v,:] = Σ_{row: tokens[row]==v} dY[row,:]。ids 无梯度。
    const dW = embeddingBwdRef( toU32( ids ), toF32( dy ), weight.shape[ 0 ], D );
    return { dIns: { weight: pack( dW, weight, 'Embed.dW' ) } };
  } );

  reg.register( 'Matmul', ( ctx, dOut ) =>
  {
    const a = ctx.ins.a;
    const b = portOrParam( ctx, 'b' );
    const dy = dOut.out;
    if ( !a ) throw new Error( 'Matmul 反向：缺少输入 a' );
    if ( !b ) throw new Error( 'Matmul 反向：缺少权重 b' );
    if ( !dy ) throw new Error( 'Matmul 反向：缺少输出梯度 out' );
    if ( b.shape.length < 2 ) throw new Error( `Matmul 反向：B 应为 [N,K]，实际秩 ${ b.shape.length }` );
    const N = b.shape[ 0 ];
    const K = b.shape[ 1 ];
    if ( K === 0 ) throw new Error( 'Matmul 反向：K 维为 0' );
    const M = numel( a.shape ) / K;
    if ( !Number.isInteger( M ) )
      throw new Error( `Matmul 反向：A 元素数 ${ numel( a.shape ) } 不能被 K=${ K } 整除` );

    // a / y / dy 全部按 2D 处理（末维连续，展平布局不变）。
    const aData = toF32( a );
    const bData = toF32( b );
    const dyData = toF32( dy );
    // y = a @ b^T ⇒ dA = dy @ b、dB = dy^T @ a。
    const dA = gemmNNRef( dyData, bData, M, K, N );
    const dB = gemmTNRef( dyData, aData, N, K, M );
    const dIns: Record<string, TensorValue> = {
      a: pack( dA, a, 'Matmul.dA' ),
      b: pack( dB, b, 'Matmul.dB' ),
    };
    const bias = portOrParam( ctx, 'bias' );
    if ( bias ) dIns.bias = pack( sumRowsRef( dyData, M, N ), bias, 'Matmul.dBias' );
    return { dIns };
  } );

  reg.register( 'LayerNorm', ( ctx, dOut ) =>
  {
    const x = ctx.ins.x;
    const w = portOrParam( ctx, 'weight' );
    const bias = portOrParam( ctx, 'bias' );
    const dy = dOut.out;
    if ( !x ) throw new Error( 'LayerNorm 反向：缺少输入 x' );
    if ( !w ) throw new Error( 'LayerNorm 反向：缺少 weight（无法确定 D）' );
    if ( !dy ) throw new Error( 'LayerNorm 反向：缺少输出梯度 out' );
    const D = w.shape[ 0 ];
    if ( D === undefined || D === 0 ) throw new Error( 'LayerNorm 反向：weight 形状应为 [D]' );
    const rows = numel( x.shape ) / D;
    if ( !Number.isInteger( rows ) )
      throw new Error( `LayerNorm 反向：元素数 ${ numel( x.shape ) } 不能被 D=${ D } 整除` );
    const eps = propNumber( ctx.props, 'eps', exprEnv( ctx ), 1e-5 ) ?? 1e-5;
    const { dx, dw, db } = layernormBwdRef( toF32( x ), rows, D, toF32( w ), toF32( dy ), eps );
    const dIns: Record<string, TensorValue> = { x: pack( dx, x, 'LayerNorm.dx' ) };
    dIns.weight = pack( dw, w, 'LayerNorm.dw' );
    // bias 只在前向确实带 bias 时才回传，否则与正向的分支不对齐。
    if ( bias ) dIns.bias = pack( db, bias, 'LayerNorm.db' );
    return { dIns };
  } );

  reg.register( 'RMSNorm', ( ctx, dOut ) =>
  {
    const x = ctx.ins.x;
    const w = portOrParam( ctx, 'weight' );
    const dy = dOut.out;
    if ( !x ) throw new Error( 'RMSNorm 反向：缺少输入 x' );
    if ( !w ) throw new Error( 'RMSNorm 反向：缺少 weight（无法确定 D）' );
    if ( !dy ) throw new Error( 'RMSNorm 反向：缺少输出梯度 out' );
    const D = w.shape[ 0 ];
    if ( D === undefined || D === 0 ) throw new Error( 'RMSNorm 反向：weight 形状应为 [D]' );
    const rows = numel( x.shape ) / D;
    if ( !Number.isInteger( rows ) )
      throw new Error( `RMSNorm 反向：元素数 ${ numel( x.shape ) } 不能被 D=${ D } 整除` );
    const eps = propNumber( ctx.props, 'eps', exprEnv( ctx ), 1e-5 ) ?? 1e-5;

    const xd = toF32( x );
    const wd = toF32( w );
    const dyd = toF32( dy );
    const dx = new Float32Array( xd.length );
    const dw = new Float32Array( D );
    for ( let r = 0; r < rows; r++ )
    {
      const base = r * D;
      let ss = 0;
      for ( let d = 0; d < D; d++ ) ss += xd[ base + d ] * xd[ base + d ];
      // rstd = 1/sqrt(mean(x²)+eps)，y = x·rstd·w。
      const rstd = 1 / Math.sqrt( ss / D + eps );
      const rstd3 = rstd * rstd * rstd;
      // 关键中间量：c = Σ_j dy_j·w_j·x_j。它对 x 的依赖来自 rstd 随 x 变化。
      // 推导：drstd/dx_i = -x_i·rstd³/D；
      //   dx_i = Σ_j dy_j·d(y_j)/dx_i
      //        = dy_i·rstd·w_i + (Σ_j dy_j·x_j·w_j)·(-x_i·rstd³/D)
      //        = w_i·rstd·dy_i - x_i·rstd³·mean_j(dy_j·w_j·x_j)。
      let c = 0;
      for ( let d = 0; d < D; d++ ) c += dyd[ base + d ] * wd[ d ] * xd[ base + d ];
      const meanC = c / D;
      for ( let d = 0; d < D; d++ )
      {
        const xv = xd[ base + d ];
        dx[ base + d ] = wd[ d ] * rstd * dyd[ base + d ] - xv * rstd3 * meanC;
        dw[ d ] += dyd[ base + d ] * xv * rstd;
      }
    }
    return { dIns: { x: pack( dx, x, 'RMSNorm.dx' ), weight: pack( dw, w, 'RMSNorm.dw' ) } };
  } );

  reg.register( 'GELU', ( ctx, dOut ) =>
  {
    const x = ctx.ins.x;
    const dy = dOut.out;
    if ( !x ) throw new Error( 'GELU 反向：缺少输入 x' );
    if ( !dy ) throw new Error( 'GELU 反向：缺少输出梯度 out' );
    return { dIns: { x: pack( geluBwdRef( toF32( x ), toF32( dy ) ), x, 'GELU.dx' ) } };
  } );

  reg.register( 'SiLU', ( ctx, dOut ) =>
  {
    const x = ctx.ins.x;
    const gate = ctx.ins.gate;
    const dy = dOut.out;
    if ( !x ) throw new Error( 'SiLU 反向：缺少输入 x' );
    if ( !dy ) throw new Error( 'SiLU 反向：缺少输出梯度 out' );
    const xd = toF32( x );
    const dyd = toF32( dy );
    const gd = gate ? toF32( gate ) : undefined;
    const dx = new Float32Array( xd.length );
    const dgate = gd ? new Float32Array( xd.length ) : undefined;
    for ( let i = 0; i < xd.length; i++ )
    {
      const v = xd[ i ];
      const s = 1 / ( 1 + Math.exp( -v ) );       // sigmoid(v)
      const silu = v * s;                          // SiLU 取值
      const dsilu = s + v * s * ( 1 - s );         // SiLU'(v)
      // 有 gate 时正向 y = SiLU(x)·g，故 dx 还要乘上 g。
      dx[ i ] = gd ? dyd[ i ] * dsilu * gd[ i ] : dyd[ i ] * dsilu;
      if ( dgate ) dgate[ i ] = dyd[ i ] * silu;
    }
    const dIns: Record<string, TensorValue> = { x: pack( dx, x, 'SiLU.dx' ) };
    if ( gate && dgate ) dIns.gate = pack( dgate, gate, 'SiLU.dgate' );
    return { dIns };
  } );

  reg.register( 'Softmax', ( ctx, dOut ) =>
  {
    const x = ctx.ins.x;
    const dy = dOut.out;
    if ( !x ) throw new Error( 'Softmax 反向：缺少输入 x' );
    if ( !dy ) throw new Error( 'Softmax 反向：缺少输出梯度 out' );
    const D = x.shape[ x.shape.length - 1 ];
    if ( D === undefined || D === 0 ) throw new Error( 'Softmax 反向：末维为 0' );
    const rows = numel( x.shape ) / D;
    if ( !Number.isInteger( rows ) ) throw new Error( 'Softmax 反向：末维与元素数不整除' );
    const xd = toF32( x );
    const dyd = toF32( dy );
    const dx = new Float32Array( xd.length );
    const s = new Float32Array( D );
    for ( let r = 0; r < rows; r++ )
    {
      const base = r * D;
      let m = -Infinity;
      for ( let d = 0; d < D; d++ ) m = Math.max( m, xd[ base + d ] );
      let sum = 0;
      for ( let d = 0; d < D; d++ ) { s[ d ] = Math.exp( xd[ base + d ] - m ); sum += s[ d ]; }
      for ( let d = 0; d < D; d++ ) s[ d ] /= sum;
      // softmax 雅可比：dx_i = s_i·(dy_i - Σ_j dy_j·s_j)。
      let dot = 0;
      for ( let d = 0; d < D; d++ ) dot += dyd[ base + d ] * s[ d ];
      for ( let d = 0; d < D; d++ ) dx[ base + d ] = s[ d ] * ( dyd[ base + d ] - dot );
    }
    return { dIns: { x: pack( dx, x, 'Softmax.dx' ) } };
  } );

  reg.register( 'CrossEntropy', ( ctx, dOut ) =>
  {
    const logits = ctx.ins.logits;
    const targets = ctx.ins.targets;
    if ( !logits ) throw new Error( 'CrossEntropy 反向：缺少 logits' );
    if ( !targets ) throw new Error( 'CrossEntropy 反向：缺少 targets' );
    const V = logits.shape[ logits.shape.length - 1 ];
    if ( V === undefined || V === 0 ) throw new Error( 'CrossEntropy 反向：末维为 0' );
    const M = numel( logits.shape ) / V;
    if ( !Number.isInteger( M ) ) throw new Error( 'CrossEntropy 反向：元素数不能被 V 整除' );
    // 参考实现已含 1/M；根收到的标量种子（1）忽略——损失对自身的导数恒为 1。
    void dOut;
    const dlogits = ceSoftmaxBwdRef( toF32( logits ), M, V, toU32( targets ) );
    return { dIns: { logits: pack( dlogits, logits, 'CrossEntropy.dlogits' ) } };
  } );

  reg.register( 'Reshape', ( ctx, dOut ) =>
  {
    const x = ctx.ins.x;
    const dy = dOut.out;
    if ( !x ) throw new Error( 'Reshape 反向：缺少输入 x' );
    if ( !dy ) throw new Error( 'Reshape 反向：缺少输出梯度 out' );
    // 形变可逆：梯度只换形状、不动数据。
    return { dIns: { x: pack( asF32( dy ), x, 'Reshape.dx' ) } };
  } );

  reg.register( 'Cast', ( ctx, dOut ) =>
  {
    const x = ctx.ins.x;
    const dy = dOut.out;
    if ( !x ) throw new Error( 'Cast 反向：缺少输入 x' );
    if ( !dy ) throw new Error( 'Cast 反向：缺少输出梯度 out' );
    // 位宽转换在 CPU 后端只换 dtype 标注，梯度数据原样回流（shape 改回 x 的）。
    return { dIns: { x: pack( asF32( dy ), x, 'Cast.dx' ) } };
  } );

  reg.register( 'Concat', ( ctx, dOut ) =>
  {
    const dy = dOut.out;
    if ( !dy ) throw new Error( 'Concat 反向：缺少输出梯度 out' );
    // 组合树：按各 child 分支末维宽度切开。
    if ( ctx.childOutputs.length > 0 )
      return { dChildren: splitLast( dy, ctx.childOutputs, 'Concat' ) };
    // 叶用法：从端口 a / b 切。
    const a = ctx.ins.a;
    const b = ctx.ins.b;
    const list = [ a, b ].filter( ( t ): t is TensorValue => !!t );
    if ( list.length === 0 ) throw new Error( 'Concat 反向：既无子节点也无 a/b 端口' );
    const pieces = splitLast( dy, list, 'Concat' );
    const dIns: Record<string, TensorValue> = {};
    if ( a ) dIns.a = pieces[ 0 ];
    if ( b ) dIns.b = pieces[ a ? 1 : 0 ];
    return { dIns };
  } );

  reg.register( 'RoPE', ( ctx, dOut ) =>
  {
    const x = ctx.ins.x;
    const dy = dOut.out;
    if ( !x ) throw new Error( 'RoPE 反向：缺少输入 x' );
    if ( !dy ) throw new Error( 'RoPE 反向：缺少输出梯度 out' );
    const H = needNumber( ctx, 'heads', 'RoPE' );
    const D = needNumber( ctx, 'headDim', 'RoPE' );
    // 位置数 T 与正向一致：符号绑定优先，其次 props.seqLen。
    const T = ctx.symbols[ 'T' ] ?? propNumber( ctx.props, 'seqLen', exprEnv( ctx ), 0 ) ?? 0;
    const base = propNumber( ctx.props, 'base', exprEnv( ctx ), 10000 ) ?? 10000;
    if ( !( T > 0 ) ) throw new Error( `RoPE 反向：需要正的 T，实际 ${ T }` );
    const rows = numel( x.shape ) / ( H * D );
    if ( !Number.isInteger( rows ) ) throw new Error( 'RoPE 反向：元素数不能被 H×D 整除' );
    // 旋转正交 ⇒ 反向就是逆旋转（转置）。
    const dx = ropeBwdRef( toF32( dy ), rows, H, D, T, base );
    return { dIns: { x: pack( dx, x, 'RoPE.dx' ) } };
  } );

  reg.register( 'Attention', ( ctx, dOut ) =>
  {
    const q = ctx.ins.q;
    const k = ctx.ins.k;
    const v = ctx.ins.v;
    const dy = dOut.out;
    if ( !q || !k || !v ) throw new Error( 'Attention 反向：缺少 q/k/v 之一' );
    if ( !dy ) throw new Error( 'Attention 反向：缺少输出梯度 out' );
    const H = needNumber( ctx, 'heads', 'Attention' );
    const D = needNumber( ctx, 'headDim', 'Attention' );
    const B = ctx.symbols[ 'B' ];
    const T = ctx.symbols[ 'T' ];
    if ( B === undefined || T === undefined )
      throw new Error( `Attention 反向：需要 symbols.B 与 symbols.T（当前 B=${ B } T=${ T }）` );
    const { dq, dk, dv } = attentionBwdRef( toF32( q ), toF32( k ), toF32( v ), toF32( dy ), B, T, H, D );
    return {
      dIns: {
        q: pack( dq, q, 'Attention.dq' ),
        k: pack( dk, k, 'Attention.dk' ),
        v: pack( dv, v, 'Attention.dv' ),
      },
    };
  } );

  // ---- ③ 组合原语 ---------------------------------------------------------

  // 透传组：正向输出 = 最后一个子节点输出 ⇒ 梯度只回最后一个子节点。
  // dChildren 长度 = 子节点数，只有末项非 null（其余显式 null）。
  for ( const op of PASSTHROUGH )
    reg.register( op, ( ctx, dOut ) =>
    {
      const dy = dOut.out;
      if ( !dy ) throw new Error( `${ ctx.op } 反向：缺少输出梯度 out` );
      const n = ctx.childOutputs.length;
      if ( n === 0 )
      {
        const x = ctx.ins.x;
        if ( !x ) throw new Error( `${ ctx.op } 反向：既无子节点也无 x 端口` );
        return { dIns: { x: pack( asF32( dy ), x, `${ ctx.op }.dx` ) } };
      }
      const dChildren: Array<TensorValue | null> = new Array( n ).fill( null );
      dChildren[ n - 1 ] = pack( asF32( dy ), ctx.childOutputs[ n - 1 ], `${ ctx.op }.dlast` );
      return { dChildren };
    } );

  reg.register( 'Fan', ( ctx, dOut ) =>
  {
    const dy = dOut.out;
    if ( !dy ) throw new Error( 'Fan 反向：缺少输出梯度 out' );
    const mode = propString( ctx.props, 'mode', exprEnv( ctx ), 'sum' ) ?? 'sum';
    const list = ctx.childOutputs;
    // 叶用法（无子节点，只有 x 端口）：sum/mean/concat 都退化成恒等（单分支）。
    if ( list.length === 0 )
    {
      const x = ctx.ins.x;
      if ( !x ) throw new Error( 'Fan 反向：既无子节点也无 x 端口' );
      return { dIns: { x: pack( asF32( dy ).slice(), x, 'Fan.dx' ) } };
    }
    // concat：把输出梯度按各分支末维宽度切开。
    if ( mode === 'concat' ) return { dChildren: splitLast( dy, list, 'Fan' ) };

    // sum ⇒ 每支拿同一份；mean ⇒ 再乘 1/分支数。
    const g = asF32( dy );
    const scale = mode === 'mean' ? 1 / list.length : 1;
    const dChildren = list.map( ( t ) =>
    {
      let data: Float32Array;
      if ( scale === 1 ) data = g.slice();
      else
      {
        data = new Float32Array( g.length );
        for ( let i = 0; i < g.length; i++ ) data[ i ] = g[ i ] * scale;
      }
      return pack( data, t, 'Fan' );
    } );
    return { dChildren };
  } );

  reg.register( 'Residual', ( ctx, dOut ) =>
  {
    const x = ctx.ins.x;
    const dy = dOut.out;
    if ( !x ) throw new Error( 'Residual 反向：缺少输入 x' );
    if ( !dy ) throw new Error( 'Residual 反向：缺少输出梯度 out' );
    const child = ctx.childOutputs[ 0 ];
    if ( !child ) throw new Error( 'Residual 反向：缺少子节点输出' );
    // y = x + f(x)：两支都拿 dy（各自用自己的正向 shape 包装）。
    return {
      dIns: { x: pack( asF32( dy ), x, 'Residual.dx' ) },
      dChildren: [ pack( asF32( dy ), child, 'Residual.dchild' ) ],
    };
  } );

  return reg;
}
