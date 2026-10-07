/**
 * CPU 后端算子实现表（design/06 §6）。
 *
 * `run()` 只按拓扑序让每个节点调一次 `impl(ctx)`；本文件把各 op 的**端口 / props 契约**
 * 翻译成 `src/reference/ops.ts` 里的纯数值参考实现。CPU 后端是"IR 语义到底对不对"的
 * 唯一可无头判据，所以这里刻意写得直白：不图快，只求与参考实现逐元素一致。
 *
 * 约定（与 `exec.ts` 的端口解析规则配套）：
 *   - 输入一律从 `ctx.ins`（端口）取，端口为空再退回 `ctx.params`（props.bind 解析出的权重）；
 *   - 组合原语从 `ctx.childOutputs` 取子节点输出；
 *   - 内部一律用 `Float32Array` 计算（`TensorValue.data` 可能是整型数组，读前判类型）；
 *   - 任何缺参 / 形状不相容都抛**中文 Error**——run 期静默返回 0 会把错误藏到数值里。
 */

import {
  createImplRegistry,
  type CpuImplRegistry,
  type CpuRunContext,
} from './exec';
import { f32, numel, type TensorValue } from './binding';
import type { DType, Props } from './types';
import { propBool, propNumber, propString, type ExprEnv } from './expr';
import {
  attentionRef,
  embeddingRef,
  geluRef,
  gemmNTRef,
  layernormRef,
  rmsnormRef,
  ropeRef,
  softmaxRef,
} from '../reference/ops';

// ---------------------------------------------------------------------------
// 内部辅助
// ---------------------------------------------------------------------------

/** 为 `propNumber/propBool` 造求值环境（符号表 + 当前 props）。 */
function makeEnv ( ctx: CpuRunContext ): ExprEnv
{
  return { symbols: ctx.symbols, props: ctx.props };
}

/** 张量数据统一读成 Float32Array；整型数组逐元素转换，不改原张量。 */
function toF32 ( t: TensorValue ): Float32Array
{
  if ( t.data instanceof Float32Array ) return t.data;
  const out = new Float32Array( t.data.length );
  for ( let i = 0; i < t.data.length; i++ ) out[ i ] = t.data[ i ];
  return out;
}

/** token 索引统一成 Uint32Array：源可能是 Float32 存的整数，必须先四舍五入。 */
function toU32 ( data: TensorValue[ 'data' ] ): Uint32Array
{
  if ( data instanceof Uint32Array ) return data;
  const out = new Uint32Array( data.length );
  for ( let i = 0; i < data.length; i++ ) out[ i ] = Math.round( data[ i ] );
  return out;
}

/** 端口优先、其次 props.bind 参数；都没有即抛（缺参不能静默）。 */
function take (
  a: TensorValue | null | undefined,
  b: TensorValue | undefined,
  who: string,
): TensorValue
{
  const t = a ?? b;
  if ( !t ) throw new Error( `${ who }：缺少必需的输入张量` );
  return t;
}

/** 必填数字 prop；缺失或非数字即抛（如 heads / headDim）。 */
function needNumber ( props: Props, key: string, env: ExprEnv, who: string ): number
{
  const v = propNumber( props, key, env );
  if ( v === undefined || Number.isNaN( v ) ) throw new Error( `${ who }：props.${ key } 缺失或非数字` );
  return v;
}

/** 把 `props.shape` / `props.sizes` / `props.perm` 这类维度数组解析成具体数字（符号走 symbols）。 */
function dimsOf ( raw: unknown, symbols: Record<string, number>, who: string ): number[]
{
  if ( !Array.isArray( raw ) ) throw new Error( `${ who }：期望维度数组，实际为 ${ String( raw ) }` );
  return raw.map( ( d ) =>
  {
    if ( typeof d === 'number' ) return d;
    if ( typeof d === 'string' )
    {
      const v = symbols[ d ];
      if ( v === undefined ) throw new Error( `${ who }：符号维 ${ d } 未绑定` );
      return v;
    }
    throw new Error( `${ who }：无法解析维度 ${ String( d ) }` );
  } );
}

/** 沿**最后一维**拼接一组张量（行主序下末维连续，故按外层索引整块拷贝）。 */
function concatLast ( list: TensorValue[], who: string ): TensorValue
{
  if ( list.length === 0 ) throw new Error( `${ who }：没有可拼接的输入` );
  const rank = list[ 0 ].shape.length;
  for ( const t of list )
  {
    if ( t.shape.length !== rank )
      throw new Error( `${ who }：拼接要求秩一致（${ rank } vs ${ t.shape.length }）` );
    for ( let i = 0; i < rank - 1; i++ )
      if ( t.shape[ i ] !== list[ 0 ].shape[ i ] )
        throw new Error( `${ who }：拼接要求除末维外形状一致（维 ${ i }：${ list[ 0 ].shape[ i ] } vs ${ t.shape[ i ] }）` );
  }
  const lastDims = list.map( ( t ) => t.shape[ rank - 1 ] );
  const outLast = lastDims.reduce( ( a, b ) => a + b, 0 );
  const outer = numel( list[ 0 ].shape.slice( 0, rank - 1 ) );
  const out = new Float32Array( outer * outLast );
  for ( let o = 0; o < outer; o++ )
  {
    let off = o * outLast;
    for ( let j = 0; j < list.length; j++ )
    {
      const w = lastDims[ j ];
      out.set( toF32( list[ j ] ).subarray( o * w, o * w + w ), off );
      off += w;
    }
  }
  const outShape = list[ 0 ].shape.slice();
  outShape[ rank - 1 ] = outLast;
  return f32( out, outShape );
}

// ---------------------------------------------------------------------------
// 实现表
// ---------------------------------------------------------------------------

/**
 * 构造一份全新的 CPU 实现表。刻意不做全局单例：调用方各自持有，
 * 避免测试 / 多模型并行时互相污染。
 */
export function builtinCpuImpls (): CpuImplRegistry
{
  const registry = createImplRegistry();

  // ---- ① 变换 -------------------------------------------------------------

  registry.register( 'Embed', ( ctx ) =>
  {
    const ids = take( ctx.ins.ids, undefined, 'Embed.ids' );
    const W = take( ctx.ins.weight, ctx.params.weight, 'Embed.weight' );
    const D = W.shape[ 1 ];
    if ( D === undefined ) throw new Error( `Embed：权重形状应形如 [V,D]，实际 [${ W.shape.join( ',' ) }]` );
    // ids 可能以 Float32 存整数，必须先规整成 Uint32 再查表。
    const tok = toU32( ids.data );
    const rows = ids.data.length;
    return { out: f32( embeddingRef( toF32( W ), D, tok ), [ rows, D ] ) };
  } );

  registry.register( 'Matmul', ( ctx ) =>
  {
    const a = take( ctx.ins.a, undefined, 'Matmul.a' );
    const b = take( ctx.ins.b, ctx.params.b, 'Matmul.b' );
    const bias = ctx.ins.bias ?? ctx.params.bias ?? undefined;
    if ( b.shape.length < 2 )
      throw new Error( `Matmul：B 应为 [N,K]，实际秩 ${ b.shape.length }` );
    const K = b.shape[ 1 ];
    const N = b.shape[ 0 ];
    if ( K === 0 ) throw new Error( 'Matmul：K 维为 0' );
    const M = a.data.length / K;
    if ( !Number.isInteger( M ) )
      throw new Error( `Matmul：A 元素数 ${ a.data.length } 不能被 K=${ K } 整除` );
    // 输出形状 = A 的最后一维换成 N（保持秩不变）。
    const outShape = a.shape.slice();
    outShape[ outShape.length - 1 ] = N;
    return {
      out: f32(
        gemmNTRef( toF32( a ), toF32( b ), M, N, K, bias ? toF32( bias ) : undefined ),
        outShape,
      ),
    };
  } );

  registry.register( 'LayerNorm', ( ctx ) =>
  {
    const env = makeEnv( ctx );
    const x = take( ctx.ins.x, undefined, 'LayerNorm.x' );
    const w = take( ctx.ins.weight, ctx.params.weight, 'LayerNorm.weight' );
    const b = ctx.ins.bias ?? ctx.params.bias ?? undefined;
    const D = w.shape[ 0 ];
    if ( D === undefined || D === 0 ) throw new Error( 'LayerNorm：weight 形状应为 [D]' );
    const rows = numel( x.shape ) / D;
    if ( !Number.isInteger( rows ) )
      throw new Error( `LayerNorm：元素数 ${ numel( x.shape ) } 不能被 D=${ D } 整除` );
    const eps = propNumber( ctx.props, 'eps', env, 1e-5 ) ?? 1e-5;
    const hasBias = !!b && propBool( ctx.props, 'elementwiseAffine', env, true );
    // hasBias=false 时参考实现不会读 b，用零数组占位即可。
    const bb = b ? toF32( b ) : new Float32Array( D );
    return { out: f32( layernormRef( toF32( x ), rows, D, toF32( w ), bb, eps, hasBias ), x.shape.slice() ) };
  } );

  registry.register( 'RMSNorm', ( ctx ) =>
  {
    const env = makeEnv( ctx );
    const x = take( ctx.ins.x, undefined, 'RMSNorm.x' );
    const w = take( ctx.ins.weight, ctx.params.weight, 'RMSNorm.weight' );
    const D = w.shape[ 0 ];
    if ( D === undefined || D === 0 ) throw new Error( 'RMSNorm：weight 形状应为 [D]' );
    const rows = numel( x.shape ) / D;
    if ( !Number.isInteger( rows ) )
      throw new Error( `RMSNorm：元素数 ${ numel( x.shape ) } 不能被 D=${ D } 整除` );
    const eps = propNumber( ctx.props, 'eps', env, 1e-5 ) ?? 1e-5;
    return { out: f32( rmsnormRef( toF32( x ), rows, D, toF32( w ), eps ), x.shape.slice() ) };
  } );

  registry.register( 'GELU', ( ctx ) =>
  {
    const x = take( ctx.ins.x, undefined, 'GELU.x' );
    return { out: f32( geluRef( toF32( x ) ), x.shape.slice() ) };
  } );

  registry.register( 'SiLU', ( ctx ) =>
  {
    const x = take( ctx.ins.x, undefined, 'SiLU.x' );
    const xd = toF32( x );
    const out = new Float32Array( xd.length );
    for ( let i = 0; i < xd.length; i++ ) out[ i ] = xd[ i ] / ( 1 + Math.exp( -xd[ i ] ) );
    // SwiGLU：有 gate 时逐元素再乘一次。
    const gate = ctx.ins.gate;
    if ( gate )
    {
      const g = toF32( gate );
      if ( g.length !== out.length )
        throw new Error( `SiLU：gate 元素数 ${ g.length } 与 x ${ out.length } 不一致` );
      for ( let i = 0; i < out.length; i++ ) out[ i ] *= g[ i ];
    }
    return { out: f32( out, x.shape.slice() ) };
  } );

  registry.register( 'Softmax', ( ctx ) =>
  {
    const x = take( ctx.ins.x, undefined, 'Softmax.x' );
    const D = x.shape[ x.shape.length - 1 ];
    if ( D === undefined || D === 0 ) throw new Error( 'Softmax：末维为 0' );
    const rows = numel( x.shape ) / D;
    return { out: f32( softmaxRef( toF32( x ), rows, D ), x.shape.slice() ) };
  } );

  registry.register( 'Reshape', ( ctx ) =>
  {
    const x = take( ctx.ins.x, undefined, 'Reshape.x' );
    const want = dimsOf( ctx.props[ 'shape' ], ctx.symbols, 'Reshape.shape' );
    // 形变不改变元素数——不守恒说明图里维度算错了，必须报错而不是硬改。
    if ( numel( want ) !== numel( x.shape ) )
      throw new Error(
        `Reshape：元素数不守恒（[${ x.shape.join( ',' ) }]=${ numel( x.shape ) } → [${ want.join( ',' ) }]=${ numel( want ) }）`,
      );
    // 零拷贝视图：同一份底层数据换形状（与 op.ts 的「视图」语义一致）。
    return { out: { data: x.data, shape: want, dtype: x.dtype } };
  } );

  registry.register( 'Transpose', ( ctx ) =>
  {
    const x = take( ctx.ins.x, undefined, 'Transpose.x' );
    if ( x.shape.length !== 2 )
      throw new Error( `Transpose：只支持 2D，实际秩 ${ x.shape.length }` );
    const raw = ctx.props[ 'perm' ];
    const perm = raw === undefined ? [ 1, 0 ] : dimsOf( raw, ctx.symbols, 'Transpose.perm' );
    if ( perm.length !== 2 || perm[ 0 ] !== 1 || perm[ 1 ] !== 0 )
      throw new Error( `Transpose：只支持 perm=[1,0]，实际 [${ perm.join( ',' ) }]` );
    const [ R, C ] = x.shape;
    const src = toF32( x );
    const out = new Float32Array( R * C );
    for ( let r = 0; r < R; r++ )
      for ( let c = 0; c < C; c++ )
        out[ c * R + r ] = src[ r * C + c ];
    return { out: f32( out, [ C, R ] ) };
  } );

  registry.register( 'Cast', ( ctx ) =>
  {
    const x = take( ctx.ins.x, undefined, 'Cast.x' );
    const env = makeEnv( ctx );
    const dt = ( propString( ctx.props, 'dtype', env, x.dtype ) ?? x.dtype ) as DType;
    // 内部一律 fp32；这里只换 dtype 标注（真正的位宽转换是 GPU 后端的事）。
    return { out: { data: x.data, shape: x.shape.slice(), dtype: dt } };
  } );

  registry.register( 'Concat', ( ctx ) =>
  {
    // 组合树上一般给 children；叶用法给端口 a/b。
    const list = ctx.childOutputs.length > 0
      ? ctx.childOutputs
      : [ ctx.ins.a, ctx.ins.b ].filter( ( t ): t is TensorValue => t !== null && t !== undefined );
    return { out: concatLast( list, 'Concat' ) };
  } );

  registry.register( 'Split', ( ctx ) =>
  {
    const x = take( ctx.ins.x, undefined, 'Split.x' );
    const sizes = dimsOf( ctx.props[ 'sizes' ], ctx.symbols, 'Split.sizes' );
    if ( sizes.length !== 2 ) throw new Error( `Split：只支持 2 片，实际 ${ sizes.length }` );
    const rank = x.shape.length;
    const D = x.shape[ rank - 1 ];
    if ( sizes[ 0 ] + sizes[ 1 ] !== D )
      throw new Error( `Split：sizes 和不等于末维（${ sizes[ 0 ] }+${ sizes[ 1 ] } ≠ ${ D }）` );
    const outer = numel( x.shape.slice( 0, rank - 1 ) );
    const src = toF32( x );
    const a = new Float32Array( outer * sizes[ 0 ] );
    const b = new Float32Array( outer * sizes[ 1 ] );
    for ( let o = 0; o < outer; o++ )
    {
      a.set( src.subarray( o * D, o * D + sizes[ 0 ] ), o * sizes[ 0 ] );
      b.set( src.subarray( o * D + sizes[ 0 ], o * D + D ), o * sizes[ 1 ] );
    }
    const shapeA = x.shape.slice(); shapeA[ rank - 1 ] = sizes[ 0 ];
    const shapeB = x.shape.slice(); shapeB[ rank - 1 ] = sizes[ 1 ];
    return { a: f32( a, shapeA ), b: f32( b, shapeB ) };
  } );

  registry.register( 'RoPE', ( ctx ) =>
  {
    const env = makeEnv( ctx );
    const x = take( ctx.ins.x, undefined, 'RoPE.x' );
    const H = needNumber( ctx.props, 'heads', env, 'RoPE' );
    const D = needNumber( ctx.props, 'headDim', env, 'RoPE' );
    // 位置数 T 优先取符号绑定，其次 props.seqLen；都没有则无法确定 pos 周期。
    const symT: number | undefined = ctx.symbols[ 'T' ];
    const T = symT ?? propNumber( ctx.props, 'seqLen', env, 0 ) ?? 0;
    const base = propNumber( ctx.props, 'base', env, 10000 ) ?? 10000;
    if ( !( T > 0 ) ) throw new Error( `RoPE：需要正的 T（symbols.T 或 props.seqLen），实际 ${ T }` );
    const rows = numel( x.shape ) / ( H * D );
    return { out: f32( ropeRef( toF32( x ), rows, H, D, T, base ), x.shape.slice() ) };
  } );

  registry.register( 'Attention', ( ctx ) =>
  {
    const env = makeEnv( ctx );
    const q = take( ctx.ins.q, undefined, 'Attention.q' );
    const k = take( ctx.ins.k, undefined, 'Attention.k' );
    const v = take( ctx.ins.v, undefined, 'Attention.v' );
    const H = needNumber( ctx.props, 'heads', env, 'Attention' );
    const D = needNumber( ctx.props, 'headDim', env, 'Attention' );
    const B: number | undefined = ctx.symbols[ 'B' ];
    const T: number | undefined = ctx.symbols[ 'T' ];
    if ( B === undefined || T === undefined )
      throw new Error( `Attention：需要 symbols.B 与 symbols.T（当前 B=${ B } T=${ T }）` );
    const causal = propBool( ctx.props, 'causal', env, true );
    return { out: f32( attentionRef( toF32( q ), toF32( k ), toF32( v ), B, T, H, D, causal ), q.shape.slice() ) };
  } );

  // ---- ③ 组合原语 ---------------------------------------------------------

  // 这些原语在 run 期只做「输出 = 最后一个子节点的输出」的透传：
  // 语义（循环展开 / 条件 / 路由）在 emit / infer 阶段已落到子图上。
  const passthrough = [
    'Seq', 'Repeat', 'If', 'Map', 'Bus', 'Gate', 'Memory', 'AdaLn',
    'CrossAttn', 'Rollout', 'Cache',
  ];
  for ( const name of passthrough )
    registry.register( name, ( ctx ) =>
    {
      const outs = ctx.childOutputs;
      const last = outs.length > 0 ? outs[ outs.length - 1 ] : ( ctx.ins.x ?? null );
      if ( !last ) throw new Error( `${ ctx.op }：没有子节点输出，也没有 x 端口` );
      return { out: last };
    } );

  registry.register( 'Fan', ( ctx ) =>
  {
    const env = makeEnv( ctx );
    const mode = propString( ctx.props, 'mode', env, 'sum' ) ?? 'sum';
    const list = ctx.childOutputs.length > 0
      ? ctx.childOutputs
      : ( ctx.ins.x ? [ ctx.ins.x ] : [] );
    if ( list.length === 0 ) throw new Error( 'Fan：没有分支输入' );
    if ( mode === 'concat' ) return { out: concatLast( list, 'Fan' ) };
    const n = numel( list[ 0 ].shape );
    const out = new Float32Array( n );
    for ( const t of list )
    {
      const d = toF32( t );
      // 逐元素相加要求元素数一致（与 Residual 同判据：只比 numel，不比形状）。
      if ( d.length !== n ) throw new Error( `Fan：分支元素数不一致（${ n } vs ${ d.length }）` );
      for ( let i = 0; i < n; i++ ) out[ i ] += d[ i ];
    }
    if ( mode === 'mean' ) for ( let i = 0; i < n; i++ ) out[ i ] /= list.length;
    return { out: f32( out, list[ 0 ].shape.slice() ) };
  } );

  registry.register( 'Residual', ( ctx ) =>
  {
    const x = take( ctx.ins.x, undefined, 'Residual.x' );
    const y = ctx.childOutputs[ 0 ];
    if ( !y ) throw new Error( 'Residual：缺少子节点输出' );
    const xd = toF32( x );
    const yd = toF32( y );
    // 只比元素数：上层 x 可能是 [B,T,C]，子输出退化成 [B,C]，但两者元素数相等即合法。
    if ( xd.length !== yd.length )
      throw new Error( `Residual：元素数不一致（x=${ xd.length } vs f(x)=${ yd.length }）` );
    const out = new Float32Array( xd.length );
    for ( let i = 0; i < out.length; i++ ) out[ i ] = xd[ i ] + yd[ i ];
    return { out: f32( out, x.shape.slice() ) };
  } );

  return registry;
}
