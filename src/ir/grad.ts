/**
 * 反向传播落到 IR（design/06 §4：`Strategy = backprop` 的第一刀）。
 *
 * 设计立场（02 §2 R3）：**能力必须被声明**。所以反向不走"猜每个算子的导数"，
 * 而是查每个 op 的 `vjp` 实现表——与正向的 `CpuOpImpl` 表完全对称。
 *
 * 关键结构决定：**梯度路由复用正向解析出的 `sources`**（exec.ts 记录在 ctx 里）。
 * 于是"正向怎么连"与"反向往哪流"不可能分歧——这是这一层最重要的正确性来源。
 */

import type { InferResult, Model, NodeId } from './types';
import type { Artifact } from './emit';
import { ensureBuiltinOps } from './ops';
import type { TensorData, TensorTable, TensorValue } from './binding';
import type { CpuImplRegistry, CpuRunContext } from './exec';
import { run } from './exec';
import { planBackward } from './backward-plan';

// ---------------------------------------------------------------------------
// VJP 契约
// ---------------------------------------------------------------------------

/** 反向时交给算子的上下文：正向上下文 + 各输出端口的正向值。 */
export interface GradContext extends CpuRunContext
{
  outs: Record<string, TensorValue>;
}

export interface VjpResult
{
  /** 输入端口梯度（按端口名）。来源是参数时会被路由到参数梯度。 */
  dIns?: Record<string, TensorValue | null | undefined>;
  /** 参数梯度，键是 `props.bind` 里的名字（与正向绑定同名）。 */
  dParams?: Record<string, TensorValue>;
  /** 组合原语：按 `children` 顺序把梯度直接投给子节点。 */
  dChildren?: Array<TensorValue | null | undefined>;
}

export type VjpFn = ( ctx: GradContext, dOut: Record<string, TensorValue> ) => VjpResult;

export interface GradRegistry
{
  get ( op: string ): VjpFn | undefined;
  register ( op: string, fn: VjpFn ): void;
  has ( op: string ): boolean;
  ops (): string[];
}

export function createGradRegistry (): GradRegistry
{
  const map = new Map<string, VjpFn>();
  return {
    get: ( op ) => map.get( op ),
    register: ( op, fn ) => { map.set( op, fn ); },
    has: ( op ) => map.has( op ),
    ops: () => [ ...map.keys() ].sort(),
  };
}

// ---------------------------------------------------------------------------
// 梯度小工具（VJP 实现会大量用到）
// ---------------------------------------------------------------------------

export function zerosLike ( t: TensorValue ): TensorValue
{
  return { data: new Float32Array( t.data.length ), shape: [ ...t.shape ], dtype: 'f32' };
}

export function makeScalar ( v: number ): TensorValue
{
  return { data: new Float32Array( [ v ] ), shape: [], dtype: 'f32' };
}

export function scalarValue ( t: TensorValue ): number
{
  return Number( t.data[ 0 ] );
}

export function asF32 ( t: TensorValue ): Float32Array
{
  return t.data instanceof Float32Array ? t.data : new Float32Array( t.data );
}

/** 逐元素相加（形状以 a 为准；长度必须一致）。 */
export function addGrad ( a: TensorValue, b: TensorValue ): TensorValue
{
  const af = asF32( a );
  const bf = asF32( b );
  if ( af.length !== bf.length )
    throw new Error( `addGrad: 长度不一致 ${ af.length } vs ${ bf.length }` );
  const out = new Float32Array( af.length );
  for ( let i = 0; i < af.length; i++ ) out[ i ] = af[ i ] + bf[ i ];
  return { data: out, shape: [ ...a.shape ], dtype: 'f32' };
}

// ---------------------------------------------------------------------------
// 反向驱动
// ---------------------------------------------------------------------------

export interface BackwardOptions
{
  /** 根节点的输出端口（默认取第一个）。 */
  rootPort?: string;
  /** 种子梯度（默认标量 1，适用于根是标量损失的图）。 */
  seed?: TensorValue;
}

export interface BackwardResult
{
  /** 前向损失（根的标量输出；根不是标量时取其第一个元素）。 */
  loss: number;
  /** 参数名 → 梯度。名字与正向 `props.bind` 一致。 */
  dParams: Record<string, Float32Array>;
  /** 每个节点每个端口的输出梯度（诊断/调试用）。 */
  dNodes: Record<NodeId, Record<string, TensorValue>>;
  /** 正向耗时相关的可观测信息。 */
  forward: { passCount: number; passBreaks: number; nodeCount: number };
}

export function backward (
  model: Model,
  ir: InferResult,
  artifact: Artifact,
  binding: { tensors: TensorTable; symbols?: Record<string, number> },
  impls: CpuImplRegistry,
  grads: GradRegistry,
  opts: BackwardOptions = {},
): BackwardResult
{
  ensureBuiltinOps();

  const fwd = run( model, ir, artifact, binding, impls );
  const ctxs = fwd.trace.contexts;
  const values = fwd.trace.nodes;
  const order = fwd.trace.order;

  const acc: Record<NodeId, Record<string, TensorValue>> = {};
  const addTo = ( nodeId: NodeId, port: string, g: TensorValue ): void =>
  {
    let slot = acc[ nodeId ];
    if ( !slot ) { slot = {}; acc[ nodeId ] = slot; }
    const prev = slot[ port ];
    slot[ port ] = prev ? addGrad( prev, g ) : g;
  };

  const firstPortOf = ( nodeId: NodeId ): string | undefined =>
  {
    const v = values[ nodeId ];
    if ( !v ) return undefined;
    for ( const k of Object.keys( v ) ) return k;
    return undefined;
  };

  const rootId = ir.graph.root ?? order[ order.length - 1 ];
  const rootPort = opts.rootPort ?? firstPortOf( rootId );
  if ( !rootPort ) throw new Error( `backward: 根节点 ${ rootId } 没有输出端口` );
  addTo( rootId, rootPort, opts.seed ?? makeScalar( 1 ) );

  const dParams: Record<string, Float32Array> = {};

  /**
   * 参数共享（tie）归并：不同绑定名可能指向**同一份内存**（`lmHead.w` 与 `wte`）。
   * 梯度必须按张量身份累加，否则优化器会把同一个参数更新两次、还会各配一套 m/v。
   * 归并规则：按名字字典序取第一个作为规范名（确定性，不依赖遍历顺序）。
   */
  const canonical = new Map<string, string>();
  {
    const byData = new Map<unknown, string>();
    for ( const name of binding.tensors.names() )
    {
      const t = binding.tensors.get( name );
      if ( !t ) continue;
      const prev = byData.get( t.data );
      if ( prev === undefined ) byData.set( t.data, name );
      else canonical.set( name, prev );
    }
  }
  const canon = ( name: string ): string => canonical.get( name ) ?? name;

  /** 写入参数梯度（走规范名）。 */
  const addParam = ( name: string, g: TensorValue ): void =>
  {
    const k = canon( name );
    dParams[ k ] = dParams[ k ] ? inPlaceAdd( dParams[ k ], g ) : asF32( g ).slice();
  };

  // 缺口#2：反向逆序序列由编译器显式排出（planBackward），不再隐式 for 循环。
  const bplan = planBackward( model, ir, grads );
  for ( const step of bplan.steps )
  {
    const id = step.nodeId;
    const dOut = acc[ id ];
    if ( !dOut || Object.keys( dOut ).length === 0 ) continue;
    const ctx = ctxs[ id ];
    if ( !ctx ) continue;

    const vjp = grads.get( ctx.op );
    if ( !vjp )
      throw new Error( `backward: op ${ ctx.op } 没有 VJP 实现（当前表：${ grads.ops().join( ', ' ) }）` );

    const gctx: GradContext = { ...ctx, outs: values[ id ] ?? {} };
    const res = vjp( gctx, dOut ) ?? {};

    // (a) 参数梯度：显式 dParams + 端口来源是 param 的 dIns
    for ( const k of Object.keys( res.dParams ?? {} ) )
    {
      const g = res.dParams![ k ];
      if ( g ) addParam( k, g );
    }
    for ( const k of Object.keys( res.dIns ?? {} ) )
    {
      const g = res.dIns![ k ];
      if ( !g ) continue;
      const src = ctx.sources[ k ];
      if ( src && 'param' in src ) addParam( src.param, g );
    }

    // (b) 节点梯度：按正向记录的来源原路回流
    for ( const k of Object.keys( res.dIns ?? {} ) )
    {
      const g = res.dIns![ k ];
      if ( !g ) continue;
      const src = ctx.sources[ k ];
      if ( src && 'nodeId' in src ) addTo( src.nodeId, src.outPort, g );
    }

    // (c) 组合原语：按 children 顺序把梯度投给子节点
    const childIds = model.nodes[ id ]?.children ?? [];
    const dCh = res.dChildren ?? [];
    for ( let c = 0; c < childIds.length; c++ )
    {
      const g = dCh[ c ];
      if ( !g ) continue;
      const cid = childIds[ c ];
      const port = firstPortOf( cid );
      if ( port ) addTo( cid, port, g );
    }
  }

  const loss = Number( fwd.rootOutput.data[ 0 ] );
  return {
    loss,
    dParams,
    dNodes: acc,
    forward: { passCount: fwd.trace.passCount, passBreaks: fwd.trace.passBreaks, nodeCount: order.length },
  };
}

function inPlaceAdd ( dst: Float32Array, g: TensorValue ): Float32Array
{
  const gf = asF32( g );
  if ( dst.length !== gf.length ) throw new Error( `inPlaceAdd: 长度不一致 ${ dst.length } vs ${ gf.length }` );
  for ( let i = 0; i < dst.length; i++ ) dst[ i ] += gf[ i ];
  return dst;
}

/**
 * 按**张量身份**取参数梯度。
 *
 * 因为 tie 归并（`lmHead.w` 与 `wte` 是同一份内存）后，梯度只存在**规范名**下，
 * 直接用别名查会得到 undefined。这个辅助把"别名 → 规范名"的解析收在一处，
 * 避免每个调用方各写一遍（也避免误以为"梯度丢了"）。
 */
export function dParamOf (
  dParams: Record<string, Float32Array>,
  tensors: TensorTable,
  name: string,
): Float32Array | undefined
{
  const direct = dParams[ name ];
  if ( direct ) return direct;
  const t = tensors.get( name );
  if ( !t ) return undefined;
  for ( const k of Object.keys( dParams ) )
  {
    const tk = tensors.get( k );
    if ( tk && tk.data === t.data ) return dParams[ k ];
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// AdamW（与 src/train/trainer.ts 同一套超参语义；这里只作用在被绑定的张量上）
// ---------------------------------------------------------------------------

export interface AdamWConfig
{
  lr: number;
  b1?: number;
  b2?: number;
  eps?: number;
  wd?: number;
  /** 梯度裁剪（按全局 L2 范数）。0 = 不裁。 */
  clip?: number;
}

export interface Optimizer
{
  /** 用一批参数梯度更新张量（就地）。 */
  step ( dParams: Record<string, Float32Array>, tensors: TensorTable ): void;
  readonly steps: number;
  /** 最近一步的梯度范数（裁剪前）。 */
  readonly lastGradNorm: number;
}

export function createAdamW ( cfg: AdamWConfig ): Optimizer
{
  const b1 = cfg.b1 ?? 0.9;
  const b2 = cfg.b2 ?? 0.95;
  const eps = cfg.eps ?? 1e-8;
  const wd = cfg.wd ?? 0.0;
  const clip = cfg.clip ?? 0;
  const m = new Map<string, Float32Array>();
  const v = new Map<string, Float32Array>();
  let t = 0;
  let lastGradNorm = 0;

  return {
    get steps () { return t; },
    get lastGradNorm () { return lastGradNorm; },
    step ( dParams, tensors )
    {
      // 全局梯度范数（裁剪前）
      let sq = 0;
      for ( const k of Object.keys( dParams ) )
      {
        const g = dParams[ k ];
        if ( !g ) continue;
        for ( let i = 0; i < g.length; i++ ) sq += g[ i ] * g[ i ];
      }
      lastGradNorm = Math.sqrt( sq );
      const scale = clip > 0 && lastGradNorm > clip ? clip / ( lastGradNorm + 1e-12 ) : 1;

      t += 1;
      const b1t = Math.pow( b1, t );
      const b2t = Math.pow( b2, t );
      const bc1 = 1 - b1t;
      const bc2 = 1 - b2t;

      for ( const k of Object.keys( dParams ).sort() )
      {
        const g = dParams[ k ];
        if ( !g ) continue;
        const tv = tensors.get( k );
        if ( !tv ) continue;
        const p = tv.data instanceof Float32Array ? tv.data : new Float32Array( tv.data );
        if ( p.length !== g.length ) continue;
        let mi = m.get( k );
        let vi = v.get( k );
        if ( !mi || mi.length !== g.length ) { mi = new Float32Array( g.length ); m.set( k, mi ); }
        if ( !vi || vi.length !== g.length ) { vi = new Float32Array( g.length ); v.set( k, vi ); }
        for ( let i = 0; i < p.length; i++ )
        {
          const gi = g[ i ] * scale;
          mi[ i ] = b1 * mi[ i ] + ( 1 - b1 ) * gi;
          vi[ i ] = b2 * vi[ i ] + ( 1 - b2 ) * gi * gi;
          const mhat = mi[ i ] / bc1;
          const vhat = vi[ i ] / bc2;
          p[ i ] = p[ i ] - cfg.lr * ( mhat / ( Math.sqrt( vhat ) + eps ) + wd * p[ i ] );
        }
        // 若原 data 不是 Float32Array（如 u32 输入），把更新写回其持有者。
        if ( !( tv.data instanceof Float32Array ) ) tv.data = p;
      }
    },
  };
}

// ---------------------------------------------------------------------------
// 一步训练
// ---------------------------------------------------------------------------

export interface TrainStepResult { loss: number; gradNorm: number; }

/** 一步：前向 → 反向 → 优化器更新（全部在 CPU 后端、同一份 artifact 上）。 */
export function trainStep (
  model: Model,
  ir: InferResult,
  artifact: Artifact,
  binding: { tensors: TensorTable; symbols?: Record<string, number> },
  impls: CpuImplRegistry,
  grads: GradRegistry,
  opt: Optimizer,
): TrainStepResult
{
  const b = backward( model, ir, artifact, binding, impls, grads );
  opt.step( b.dParams, binding.tensors );
  return { loss: b.loss, gradNorm: opt.lastGradNorm };
}

export type { TensorData, TensorValue };
