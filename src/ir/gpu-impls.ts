/**
 * GPU 后端算子表（design/06 §1 第四步：让 `run()` 在真实显卡上跑 IR 编译产物）。
 *
 * 与 `model/tiny-gpt.ts` 的手写前向**逐字对齐**：同一批 WGSL 源、同样的 binding 下标顺序、
 * 同样的 uniform 字段顺序、同样的 workgroup 数。这样 IR 路径与手写路径的数值才可能对拍。
 * 差异只有两处（gpu-exec.ts 顶注也记了）：
 *   1. 这里按 IR 的**节点粒度** dispatch，不把「残差加 + LayerNorm」融合成 add_layernorm；
 *   2. uniform 走 `rt.uniform(key, …)`，key 里带上形状参数，避免每步重复新建 buffer。
 *
 * 为什么不 import './index'：index.ts 会把整条 GPU 依赖链带进 Node 无头自检的依赖图，
 * 而 GPU 后端本就只在浏览器 / Vite 里跑（无头判据用 CPU 后端）。
 */

import { tensorResource, type Tensor } from '../gpu/buffer';
import type { GpuKernelSource, GpuOpImpl, GpuRunContext, GpuRuntime } from './gpu-exec';
import { propNumber, propString } from './expr';

// 内核源。`?raw` 由 Vite 处理（tsconfig 的 types 已含 vite/client）。
import embeddingWgsl from '../gpu/kernels/embedding/embedding.wgsl?raw';
import vecAddWgsl from '../gpu/kernels/misc/vec_add.wgsl?raw';
import layernormWgsl from '../gpu/kernels/norm/layernorm.wgsl?raw';
import gemmNtWgsl from '../gpu/kernels/gemm/gemm_nt.wgsl?raw';
import ropeWgsl from '../gpu/kernels/rope/rope.wgsl?raw';
import attentionWgsl from '../gpu/kernels/attention/attention.wgsl?raw';
import geluWgsl from '../gpu/kernels/activation/gelu.wgsl?raw';

/** 必须与 attention.wgsl 里的 `MAX_T` 一致；超出会被 kernel 静默截断，故在这里先抛错。 */
const MAX_T = 512;

// ---------------------------------------------------------------------------
// 内核清单
// ---------------------------------------------------------------------------

/** 本次实现用到的 WGSL 源；`key` 即实现表里 `rt.dispatch( key, … )` 引用的名字。 */
export function builtinGpuKernels (): GpuKernelSource[]
{
  return [
    { key: 'embedding', wgsl: embeddingWgsl, entry: 'main' },
    { key: 'vec_add', wgsl: vecAddWgsl, entry: 'main' },
    { key: 'layernorm', wgsl: layernormWgsl, entry: 'main' },
    { key: 'gemm_nt', wgsl: gemmNtWgsl, entry: 'main' },
    { key: 'rope', wgsl: ropeWgsl, entry: 'main' },
    { key: 'attention', wgsl: attentionWgsl, entry: 'main' },
    { key: 'gelu', wgsl: geluWgsl, entry: 'main' },
  ];
}

// ---------------------------------------------------------------------------
// 内部辅助
// ---------------------------------------------------------------------------

type Resource = GPUBuffer | GPUBufferBinding;

/** 求值环境：符号表 + 当前 props（供 propNumber / propString 解析受限表达式）。 */
function envOf ( ctx: GpuRunContext ): { symbols: Record<string, number | undefined>; props: GpuRunContext[ 'props' ] }
{
  return { symbols: ctx.symbols, props: ctx.props };
}

/** 必填数字 prop；缺失或非数字即抛（heads / headDim / n / dim 这些不能默认）。 */
function needNum ( ctx: GpuRunContext, key: string, who: string ): number
{
  const v = propNumber( ctx.props, key, envOf( ctx ) );
  if ( v === undefined || Number.isNaN( v ) ) throw new Error( `${ who }：props.${ key } 缺失或非数字` );
  return v;
}

/** 可选数字 prop，带默认值。 */
function optNum ( ctx: GpuRunContext, key: string, fallback: number ): number
{
  return propNumber( ctx.props, key, envOf( ctx ), fallback ) ?? fallback;
}

/** 张量与裸 buffer 的判别：端口张量一定带 `shape`（哪怕为空数组），GPUBuffer 没有。 */
function isTensor ( v: Tensor | GPUBuffer ): v is Tensor
{
  return Array.isArray( ( v as Tensor ).shape );
}

/** 统一成 bind group 资源：张量按其 [offset, size] 切片，裸 buffer 直接绑定。 */
function asResource ( v: Tensor | GPUBuffer ): Resource
{
  return isTensor( v ) ? tensorResource( v ) : v;
}

/** 端口张量必需；缺失即抛（缺参不能静默当 0）。 */
function requireTensor ( v: Tensor | null | undefined, who: string ): Tensor
{
  if ( !v ) throw new Error( `${ who }：缺少必需的输入张量` );
  return v;
}

/**
 * 4 字节全零 storage buffer，供「本节点没有 bias」的 kernel 占位。
 * gemm_nt / layernorm 的 bias 端口在 WGSL 里是必绑的（不可省），而 IR 里 bias 是可选的，
 * 所以缺 bias 时绑这个零 buffer，同时把 uniform 的 hasBias 置 0（kernel 不会真的去读它）。
 * 按 device 缓存，避免每次 Matmul 都新建。
 */
const dummyBiasByDevice = new WeakMap<GPUDevice, GPUBuffer>();

function dummyBias ( rt: GpuRuntime ): GPUBuffer
{
  const device = rt.gpuDevice;
  let buf = dummyBiasByDevice.get( device );
  if ( !buf )
  {
    buf = device.createBuffer( { label: 'gpu-impls:dummy-bias', size: 4, usage: GPUBufferUsage.STORAGE } );
    dummyBiasByDevice.set( device, buf );
  }
  return buf;
}

// ---------------------------------------------------------------------------
// 实现表
// ---------------------------------------------------------------------------

/**
 * 构造一份全新的 GPU 实现表：op 名（**不带版本**）→ 实现。
 * 刻意不做全局单例，与 `builtinCpuImpls()` 对称，避免多模型并行时互相污染。
 */
export function builtinGpuImpls (): Map<string, GpuOpImpl>
{
  const impls = new Map<string, GpuOpImpl>();

  // ---- Embed：ids → 嵌入向量 ---------------------------------------------
  // 端口张量的 shape 为空（见 gpu-exec 端口解析），所以 C 只能取 props.dim。
  impls.set( 'Embed', ( ctx, rt ) =>
  {
    const ids = requireTensor( ctx.ins.ids, 'Embed.ids' );
    const W = ctx.ins.weight ?? ctx.params.weight;
    if ( !W ) throw new Error( 'Embed：缺少权重（端口 weight 或 props.bind.weight）' );
    const C = needNum( ctx, 'dim', 'Embed' );
    const M = ids.length;

    const uni = rt.uniform( `embed:${ M }:${ C }`, ( p ) =>
    {
      p.u32( M );
      p.u32( C );
      p.u32( 0 );
      p.u32( 0 );
    } );

    const out = rt.alloc( M * C, [ M, C ] );
    rt.dispatch(
      'embedding',
      [ asResource( W ), asResource( ids ), tensorResource( out ), uni ],
      Math.ceil( ( M * C ) / 64 ),
    );
    return { out };
  } );

  // ---- Matmul：C = A @ W^T (+ bias) --------------------------------------
  // N 取 props.n、K 由权重元素数反推、M 由 A 元素数反推——因为权重经 bind 绑定时
  // 端口张量的 shape 是空的（gpu-exec 端口解析），除元素数外拿不到任何形状信息。
  impls.set( 'Matmul', ( ctx, rt ) =>
  {
    const A = requireTensor( ctx.ins.a, 'Matmul.a' );
    const W = ctx.ins.b ?? ctx.params.b;
    if ( !W ) throw new Error( 'Matmul：缺少权重 b（端口或 props.bind.b）' );
    if ( !isTensor( W ) )
      throw new Error( 'Matmul：权重端口没有元素长度信息，无法反推 K（请把权重绑定为张量表条目）' );
    const N = needNum( ctx, 'n', 'Matmul' );
    if ( N === 0 ) throw new Error( 'Matmul：N 维为 0' );
    if ( W.length % N !== 0 ) throw new Error( `Matmul：权重元素数 ${ W.length } 不能被 N=${ N } 整除` );
    const K = W.length / N;
    if ( K === 0 ) throw new Error( 'Matmul：K 维为 0' );
    if ( A.length % K !== 0 ) throw new Error( `Matmul：A 长度 ${ A.length } 不能被 K=${ K } 整除` );
    const M = A.length / K;

    // 输出形状 = A 的前缀 + [N]（保持秩不变）。
    const outShape = A.shape.length > 0 ? A.shape.slice() : [ M, N ];
    outShape[ outShape.length - 1 ] = N;

    const bias = ctx.ins.bias ?? ctx.params.bias ?? null;
    const hasBias = bias !== null && bias !== undefined;

    const uni = rt.uniform( `gemm:${ M }:${ N }:${ K }:${ hasBias ? 1 : 0 }`, ( p ) =>
    {
      p.u32( M );
      p.u32( N );
      p.u32( K );
      p.u32( hasBias ? 1 : 0 );
    } );

    const out = rt.alloc( M * N, outShape );
    rt.dispatch(
      'gemm_nt',
      [
        tensorResource( A ),
        asResource( W ),
        tensorResource( out ),
        bias ? asResource( bias ) : dummyBias( rt ),
        uni,
      ],
      [ Math.ceil( M / 64 ), Math.ceil( N / 64 ), 1 ],
    );
    return { out };
  } );

  // ---- LayerNorm：每行一个 workgroup --------------------------------------
  impls.set( 'LayerNorm', ( ctx, rt ) =>
  {
    const x = requireTensor( ctx.ins.x, 'LayerNorm.x' );
    const W = ctx.ins.weight ?? ctx.params.weight;
    if ( !W ) throw new Error( 'LayerNorm：缺少权重（端口 weight 或 props.bind.weight）' );
    const b = ctx.ins.bias ?? ctx.params.bias ?? null;
    const hasBias = b !== null && b !== undefined;
    // C 取权重元素数；万一权重是裸 buffer（无长度）再退回 props.dim。
    const C = isTensor( W ) ? W.length : needNum( ctx, 'dim', 'LayerNorm' );
    if ( C === 0 ) throw new Error( 'LayerNorm：C 维为 0' );
    if ( x.length % C !== 0 ) throw new Error( `LayerNorm：元素数 ${ x.length } 不能被 C=${ C } 整除` );
    const M = x.length / C;
    const eps = optNum( ctx, 'eps', 1e-5 );

    const uni = rt.uniform( `ln:${ M }:${ C }:${ hasBias ? 1 : 0 }:${ eps }`, ( p ) =>
    {
      p.u32( M );
      p.u32( C );
      p.f32( eps );
      p.u32( hasBias ? 1 : 0 );
    } );

    const out = rt.alloc( x.length, x.shape.length > 0 ? x.shape.slice() : [ M, C ] );
    rt.dispatch(
      'layernorm',
      [
        tensorResource( x ),
        asResource( W ),
        b ? asResource( b ) : dummyBias( rt ),
        tensorResource( out ),
        uni,
      ],
      M,
    );
    return { out };
  } );

  // ---- GELU ---------------------------------------------------------------
  impls.set( 'GELU', ( ctx, rt ) =>
  {
    const x = requireTensor( ctx.ins.x, 'GELU.x' );
    const n = x.length;

    const uni = rt.uniform( `gelu:${ n }`, ( p ) =>
    {
      p.u32( n );
      p.u32( 0 );
      p.u32( 0 );
      p.u32( 0 );
    } );

    const out = rt.alloc( n, x.shape.length > 0 ? x.shape.slice() : [ n ] );
    rt.dispatch( 'gelu', [ tensorResource( x ), tensorResource( out ), uni ], Math.ceil( n / 64 ) );
    return { out };
  } );

  // ---- RoPE：位置数 T 来自符号表，缺失则退回 props.seqLen ----------------
  impls.set( 'RoPE', ( ctx, rt ) =>
  {
    const x = requireTensor( ctx.ins.x, 'RoPE.x' );
    const H = needNum( ctx, 'heads', 'RoPE' );
    const D = needNum( ctx, 'headDim', 'RoPE' );
    const seqLen = propNumber( ctx.props, 'seqLen', envOf( ctx ) );
    const T = ctx.symbols[ 'T' ] ?? seqLen;
    if ( T === undefined || T <= 0 )
      throw new Error( 'RoPE：需要正的 T（symbols.T 或 props.seqLen），否则 pos = row % T 无定义' );
    const base = optNum( ctx, 'base', 10000 );
    if ( H <= 0 || D <= 0 ) throw new Error( `RoPE：heads/headDim 必须为正（H=${ H } D=${ D }）` );
    if ( x.length % ( H * D ) !== 0 )
      throw new Error( `RoPE：元素数 ${ x.length } 不能被 H*D=${ H * D } 整除` );
    const rows = x.length / ( H * D );

    const uni = rt.uniform( `rope:${ rows }:${ H }:${ D }:${ T }:${ base }`, ( p ) =>
    {
      p.u32( rows );
      p.u32( H );
      p.u32( D );
      p.u32( T );
      p.f32( base );
      p.f32( 0 );
      p.f32( 0 );
      p.f32( 0 );
    } );

    const out = rt.alloc( x.length, x.shape.length > 0 ? x.shape.slice() : [ rows, H, D ] );
    rt.dispatch(
      'rope',
      [ tensorResource( x ), tensorResource( out ), uni ],
      Math.ceil( ( rows * H * ( D / 2 ) ) / 64 ),
    );
    return { out };
  } );

  // ---- Attention：q/k/v 形状 [B*T, H, D] ---------------------------------
  impls.set( 'Attention', ( ctx, rt ) =>
  {
    const q = requireTensor( ctx.ins.q, 'Attention.q' );
    const k = requireTensor( ctx.ins.k, 'Attention.k' );
    const v = requireTensor( ctx.ins.v, 'Attention.v' );
    const H = needNum( ctx, 'heads', 'Attention' );
    const D = needNum( ctx, 'headDim', 'Attention' );
    const B = ctx.symbols[ 'B' ];
    const T = ctx.symbols[ 'T' ];
    if ( B === undefined || T === undefined )
      throw new Error( `Attention：需要 symbols.B 与 symbols.T（当前 B=${ B } T=${ T }）` );
    // kernel 用 shared 数组存 scores，上限 MAX_T；超了必须显式报错而不是算错。
    if ( T > MAX_T )
      throw new Error( `Attention：T=${ T } 超过 attention.wgsl 的 MAX_T=${ MAX_T }` );
    const want = B * T * H * D;
    if ( q.length !== want )
      throw new Error( `Attention：q 元素数 ${ q.length } 与 B*T*H*D=${ want } 不一致` );

    const uni = rt.uniform( `attn:${ B }:${ T }:${ H }:${ D }`, ( p ) =>
    {
      p.u32( B );
      p.u32( T );
      p.u32( H );
      p.u32( D );
      p.f32( 1 / Math.sqrt( D ) );
      p.f32( 0 );
      p.f32( 0 );
      p.f32( 0 );
    } );

    const out = rt.alloc( q.length, q.shape.length > 0 ? q.shape.slice() : [ B * T, H, D ] );
    rt.dispatch(
      'attention',
      [ tensorResource( q ), tensorResource( k ), tensorResource( v ), tensorResource( out ), uni ],
      [ T, H, B ],
    );
    return { out };
  } );

  // ---- Fan：多分支逐元素相加（含 emb = tok + pos） ------------------------
  impls.set( 'Fan', ( ctx, rt ) =>
  {
    const mode = propString( ctx.props, 'mode', envOf( ctx ), 'sum' ) ?? 'sum';
    const list = ctx.childOutputs.length > 0
      ? ctx.childOutputs
      : ( ctx.ins.x ? [ ctx.ins.x ] : [] );
    if ( list.length === 0 ) throw new Error( 'Fan：没有分支输入' );

    // mean / concat 都需要额外 kernel（缩放 / 按轴拼接），不能靠 vec_add 硬凑，明确未实现。
    if ( mode === 'concat' )
      throw new Error( 'Fan：concat 模式在 GPU 后端未实现（缺按轴拼接的 kernel）' );
    if ( mode === 'mean' )
      throw new Error( 'Fan：mean 模式在 GPU 后端未实现（缺逐元素缩放的 kernel）' );

    // sum：只比元素数（与 CPU 参考实现同判据，不比形状）。
    const len = list[ 0 ].length;
    for ( const t of list )
      if ( t.length !== len ) throw new Error( `Fan：分支元素数不一致（${ len } vs ${ t.length }）` );

    // 单分支的 sum 就是它自己：直接透传，省一次 dispatch。
    if ( list.length === 1 ) return { out: list[ 0 ] };

    const shape = list[ 0 ].shape.length > 0 ? list[ 0 ].shape.slice() : [ len ];
    // 多于 2 个分支时链式相加：acc = a; acc = vec_add(acc, next) …（每个中间结果独立分配，
    // 因为同一 pass 内读写同一 buffer 是未定义行为）。
    let acc = list[ 0 ];
    for ( let i = 1; i < list.length; i++ )
    {
      const out = rt.alloc( len, shape );
      rt.dispatch(
        'vec_add',
        [ tensorResource( acc ), tensorResource( list[ i ] ), tensorResource( out ) ],
        Math.ceil( len / 64 ),
      );
      acc = out;
    }
    return { out: acc };
  } );

  // ---- Residual：out = x + child(x) --------------------------------------
  impls.set( 'Residual', ( ctx, rt ) =>
  {
    const x = requireTensor( ctx.ins.x, 'Residual.x' );
    const y = ctx.childOutputs[ 0 ];
    if ( !y ) throw new Error( 'Residual：缺少子节点输出' );
    if ( x.length !== y.length )
      throw new Error( `Residual：元素数不一致（x=${ x.length } vs f(x)=${ y.length }）` );

    const out = rt.alloc( x.length, x.shape.length > 0 ? x.shape.slice() : [ x.length ] );
    rt.dispatch(
      'vec_add',
      [ tensorResource( x ), tensorResource( y ), tensorResource( out ) ],
      Math.ceil( x.length / 64 ),
    );
    return { out };
  } );

  // ---- 透传组：语义（循环展开 / 条件 / 路由）在 emit / infer 期已落到子图上 --
  // 这些 op 在 run 期只做「输出 = 最后一个子节点输出（或 x 端口）」的透传，不派发。
  // 注册它们是让意图显式：即便不注册，gpu-exec 的 PASSTHROUGH_OPS 也会兜底。
  const passthroughOps = [
    'Seq', 'Repeat', 'If', 'Map', 'Bus', 'Gate', 'Memory', 'AdaLn',
    'CrossAttn', 'Rollout', 'Cache',
  ];
  for ( const name of passthroughOps )
  {
    impls.set( name, ( ctx ) =>
    {
      const outs = ctx.childOutputs;
      const last = outs.length > 0 ? outs[ outs.length - 1 ] : ( ctx.ins.x ?? null );
      if ( !last ) throw new Error( `${ name }：没有子节点输出，也没有 x 端口` );
      return { out: last };
    } );
  }

  return impls;
}
