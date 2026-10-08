/**
 * `runGpu()`：执行器 —— 在 WebGPU 上跑 `emit()` 产出的 artifact。
 *
 * 这是 design/06 §1 第四步的**真实后端**（此前只有 CPU 参考实现）。
 * 与 `exec.ts`（CPU）的分工完全对称：
 *   - 相同的端口解析来源（`ctx.sources` 由正向解析一次，GPU/CPU 共用同一套 IR 语义）
 *   - 相同的"不做静态分析"原则：只按 `ir.graph.order` 走，pass 边界**听 artifact 的**
 *   - 不同的只是"算子怎么算"：CPU 查 `cpu-impls.ts`，GPU 查 `gpu-impls.ts`
 *
 * 与手写 `model/tiny-gpt.ts` 的关键差异（诚实记录）：
 *   手写版把「残差加 + LayerNorm」融合成一个 `add_layernorm` dispatch，并且每步重建调用；
 *   这里按 IR 的节点粒度**分两次 dispatch**，且复用同一个 artifact。
 *   所以数值会一致（同样的算术顺序），但 dispatch 数更多 —— 这正是 ENG-V6
 *   「吞吐退化 <10%」要测量的东西，是**已知且预期**的差距，不是 bug。
 */

import type { GpuContext } from '../gpu/device';
import { Arena } from '../gpu/arena';
import {
  alignTo,
  createUniform,
  readbackF32,
  StructPacker,
  writeF32,
  writeU32,
  type Tensor,
} from '../gpu/buffer';
import { CommandBatch, createComputePipeline } from '../gpu/pipeline';
import type { InferResult, Model, NodeId, Props } from './types';
import type { Artifact } from './emit';
import { ensureBuiltinOps } from './ops';
import { builtinRegistry } from './op';
import type { TensorTable } from './binding';

// ---------------------------------------------------------------------------
// 内核源
// ---------------------------------------------------------------------------

/** 一个要用到的 WGSL 内核（`key` 是实现表里引用的名字）。 */
export interface GpuKernelSource
{
  key: string;
  wgsl: string;
  entry?: string;
}

// ---------------------------------------------------------------------------
// 实现契约
// ---------------------------------------------------------------------------

export interface GpuRunContext
{
  nodeId: NodeId;
  op: string;
  props: Props;
  /** 端口张量（GPU buffer 切片）。 */
  ins: Record<string, Tensor | null>;
  /** `props.bind` 对应的持久 buffer（权重/常量）。 */
  params: Record<string, GPUBuffer>;
  /** 子节点输出（组合原语用）。 */
  childOutputs: Tensor[];
  symbols: Record<string, number>;
}

/**
 * 一个算子的 GPU 实现。
 * 返回**输出端口名 → Tensor**；纯透传（Seq/Map/…）可以直接把输入张量当输出返回，
 * 此时不会产生 dispatch（executor 会跳过没有新 buffer 的节点是不可能的，
 * 所以实现需要显式调用 `rt.markPassthrough()` 或直接返回输入张量——
 * executor 只看返回的 Tensor 是否等于某个输入，不做额外判断）。
 */
export type GpuOpImpl = ( ctx: GpuRunContext, rt: GpuRuntime ) => Record<string, Tensor>;

// ---------------------------------------------------------------------------
// 运行时
// ---------------------------------------------------------------------------

export class GpuRuntime
{
  readonly arena: Arena;
  readonly batch: CommandBatch;
  readonly symbols: Record<string, number>;

  private readonly device: GPUDevice;
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private readonly sources = new Map<string, GpuKernelSource>();
  private readonly weights = new Map<string, GPUBuffer>();
  private readonly lengths = new Map<string, number>();
  private readonly uniforms = new Map<string, GPUBuffer>();
  private readonly owned: GPUBuffer[] = [];

  /** 统计（验收用）。 */
  dispatches = 0;
  passBreaks = 0;

  constructor ( gpu: GpuContext, sources: GpuKernelSource[], tensors: TensorTable, symbols: Record<string, number> )
  {
    this.device = gpu.device;
    this.arena = new Arena( gpu.device, 64 << 20 );
    this.batch = new CommandBatch( gpu.device );
    this.symbols = symbols;
    for ( const s of sources ) this.sources.set( s.key, s );
    this.uploadTensors( tensors );
  }

  get gpuDevice (): GPUDevice { return this.device; }

  /** 把整张张量表上传为持久 buffer（f32 走 STORAGE，整数走 u32）。 */
  private uploadTensors ( tensors: TensorTable ): void
  {
    for ( const name of tensors.names() )
    {
      const t = tensors.get( name );
      if ( !t ) continue;
      const buf = this.device.createBuffer( {
        label: `w:${ name }`,
        size: Math.max( alignTo( t.data.length * 4 ), 4 ),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      } );
      if ( t.data instanceof Float32Array ) writeF32( this.device, buf, 0, t.data );
      else writeU32( this.device, buf, 0, new Uint32Array( t.data ) );
      this.weights.set( name, buf );
      this.lengths.set( name, t.data.length );
      this.owned.push( buf );
    }
  }

  /** 张量表里某个名字的**元素个数**（绑为端口时用来伪造 Tensor 的 length）。 */
  lengthOf ( name: string ): number
  {
    return this.lengths.get( name ) ?? 0;
  }

  weight ( name: string ): GPUBuffer
  {
    const b = this.weights.get( name );
    if ( !b ) throw new Error( `GPU 运行时缺少张量 "${ name }"` );
    return b;
  }

  hasWeight ( name: string ): boolean
  {
    return this.weights.has( name );
  }

  /** 惰性创建并缓存 pipeline。 */
  kernel ( key: string ): GPUComputePipeline
  {
    const hit = this.pipelines.get( key );
    if ( hit ) return hit;
    const src = this.sources.get( key );
    if ( !src ) throw new Error( `未知内核 "${ key }"（已注册：${ [ ...this.sources.keys() ].sort().join( ', ' ) }）` );
    const p = createComputePipeline( this.device, src.wgsl, src.entry ?? 'main', key );
    this.pipelines.set( key, p );
    return p;
  }

  /** 按 key 缓存 uniform（同一形状重复前向时复用）。 */
  uniform ( key: string, build: ( p: StructPacker ) => void ): GPUBuffer
  {
    const hit = this.uniforms.get( key );
    if ( hit ) return hit;
    const packer = new StructPacker();
    build( packer );
    const buf = createUniform( this.device, packer.bytes(), `u:${ key }` );
    this.uniforms.set( key, buf );
    this.owned.push( buf );
    return buf;
  }

  alloc ( length: number, shape: number[] ): Tensor
  {
    return this.arena.allocFloat32( length, shape );
  }

  /** 绑定并派发（资源按下标对应 binding 0..n-1；uniform 放最后）。 */
  dispatch ( key: string, resources: ( GPUBuffer | GPUBufferBinding )[], workgroups: number | readonly number[] ): void
  {
    const pipeline = this.kernel( key );
    const entries: GPUBindGroupEntry[] = resources.map( ( resource, binding ) => ( { binding, resource } ) );
    const bindGroup = this.device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries } );
    this.batch.dispatch( pipeline, bindGroup, workgroups );
    this.dispatches += 1;
  }

  /** 结束当前 pass（下一个 dispatch 会开新 pass）——由 artifact 的 pass 边界驱动。 */
  endPass (): void
  {
    this.batch.endPass();
    this.passBreaks += 1;
  }

  submit (): void
  {
    this.batch.submit();
  }

  async readback ( t: Tensor ): Promise<Float32Array>
  {
    await this.device.queue.onSubmittedWorkDone();
    return readbackF32( this.device, t.buffer, t.offset / 4, t.length );
  }

  dispose (): void
  {
    for ( const b of this.owned ) { try { b.destroy(); } catch { /* 忽略 */ } }
    this.owned.length = 0;
  }
}

// ---------------------------------------------------------------------------
// runGpu
// ---------------------------------------------------------------------------

export interface GpuRunOptions
{
  /** 语义检查：哪些节点允许没有 GPU 实现（默认必须全有）。 */
  impls: Map<string, GpuOpImpl>;
  /** 根输出读回长度（不传则按根节点的张量长度）。 */
  readbackRoot?: boolean;
}

export interface GpuRunStats
{
  dispatches: number;
  passBreaks: number;
  /** artifact 声明的 pass 数（用于和实际一致性的对照）。 */
  plannedPasses: number;
  /** 因 op 无 GPU 实现而改为透传的节点。 */
  passthrough: string[];
}

export interface GpuRunResult
{
  rootOutput: Float32Array;
  outputs: Record<NodeId, Record<string, Tensor>>;
  stats: GpuRunStats;
}

function opNameOf ( ref: string ): string
{
  const at = ref.lastIndexOf( '@' );
  return at >= 0 ? ref.slice( 0, at ) : ref;
}

/**
 * 允许"无 dispatch 直接透传"的组合原语（输出 = 子链末端）。
 * 除这些之外的 op 若没有 GPU 实现，必须**报错**——静默透传会把算错的数当对的交出去。
 */
const PASSTHROUGH_OPS = new Set( [
  'Seq', 'Repeat', 'If', 'Map', 'Bus', 'Gate', 'Memory', 'AdaLn', 'CrossAttn', 'Rollout', 'Cache',
] );

function firstOut ( v: Record<string, Tensor> | undefined ): Tensor | null
{
  if ( !v ) return null;
  for ( const k of Object.keys( v ) ) return v[ k ];
  return null;
}

function firstOutName ( ref: string ): string
{
  const def = builtinRegistry.get( ref );
  const list = def?.contract.io?.out ?? [];
  const first = list[ 0 ];
  if ( typeof first === 'string' && first.trim() !== '' ) return first;
  if ( first && typeof first === 'object' && first.name && first.name.trim() !== '' ) return first.name;
  return 'out';
}

/** 从 artifact 建立 nodeId → pass 序号，用于驱动 `endPass()`。 */
function passIndexMap ( artifact: Artifact ): Map<NodeId, number>
{
  const m = new Map<NodeId, number>();
  for ( const pass of artifact.passes )
    for ( const d of pass.dispatches ) if ( !m.has( d.nodeId ) ) m.set( d.nodeId, pass.index );
  return m;
}

/**
 * 在 GPU 上执行一个已编译的 artifact。
 * 注意：`artifact` 提供 **pass 边界**；`ir` 提供拓扑序；函数本身不做任何静态分析。
 */
export async function runGpu (
  model: Model,
  ir: InferResult,
  artifact: Artifact,
  binding: { tensors: TensorTable; symbols?: Record<string, number> },
  rt: GpuRuntime,
  opts: GpuRunOptions,
): Promise<GpuRunResult>
{
  ensureBuiltinOps();

  const symbols = binding.symbols ?? rt.symbols;
  const order = ir.graph.order.length > 0 ? ir.graph.order : Object.keys( model.nodes ).sort();
  const passOf = passIndexMap( artifact );
  const incoming = ir.graph.incoming ?? {};

  const outputs: Record<NodeId, Record<string, Tensor>> = {};
  const passthrough: string[] = [];

  let currentPass = -1;

  for ( const id of order )
  {
    const node = model.nodes[ id ];
    if ( !node ) throw new Error( `runGpu: 节点表缺少 ${ id }` );
    const op = opNameOf( node.op );

    const impl = opts.impls.get( op );
    const contract = builtinRegistry.get( node.op )?.contract;

    // ---- 端口解析：与 CPU 执行器、与 infer 完全同一套优先级 ----
    const bindRaw = node.props?.[ 'bind' ];
    const bind: Record<string, string> = {};
    if ( bindRaw && typeof bindRaw === 'object' && !Array.isArray( bindRaw ) )
      for ( const k of Object.keys( bindRaw as Record<string, unknown> ) )
      {
        const v = ( bindRaw as Record<string, unknown> )[ k ];
        if ( typeof v === 'string' ) bind[ k ] = v;
      }

    const params: Record<string, GPUBuffer> = {};
    for ( const k of Object.keys( bind ) ) params[ k ] = rt.weight( bind[ k ] );

    const ports = ( contract?.io?.in ?? [] ).map( ( s, i ) =>
      typeof s === 'string' ? ( s.trim() !== '' ? s : `in${ i }` )
        : ( s.name && s.name.trim() !== '' ? s.name : `in${ i }` ) );

    const ins: Record<string, Tensor | null> = {};
    const children = node.children ?? [];
    for ( let k = 0; k < ports.length; k++ )
    {
      const port = ports[ k ];
      let v: Tensor | null = null;
      if ( node.slot && Object.prototype.hasOwnProperty.call( node.slot, port ) )
        v = firstOut( outputs[ node.slot[ port ] ] );
      else
      {
        const edge = ( incoming[ id ] ?? [] ).find( ( e ) => e.toPort === port );
        if ( edge ) v = outputs[ edge.from ]?.[ edge.fromPort ] ?? firstOut( outputs[ edge.from ] );
        else if ( k < children.length ) v = firstOut( outputs[ children[ k ] ] );
      }
      if ( v === null && bind[ port ] !== undefined && rt.hasWeight( bind[ port ] ) )
        v = { buffer: params[ port ], offset: 0, length: rt.lengthOf( bind[ port ] ), shape: [] };
      ins[ port ] = v;
    }

    const childOutputs: Tensor[] = [];
    for ( const c of children )
    {
      const o = firstOut( outputs[ c ] );
      if ( o ) childOutputs.push( o );
    }

    // ---- pass 边界：artifact 说该断就断 ----
    const wantPass = passOf.get( id );
    if ( wantPass !== undefined && wantPass !== currentPass )
    {
      if ( currentPass >= 0 ) rt.endPass();
      currentPass = wantPass;
    }

    if ( !impl )
    {
      // 没有 GPU 实现的节点：**只有已知的透传组合原语**允许直接透传；
      // 其余一律报错——静默透传会把"算错的数"当成"对的数"交出去。
      if ( !PASSTHROUGH_OPS.has( op ) )
        throw new Error( `runGpu: op ${ node.op } 没有 GPU 实现（节点 ${ id }）` );
      passthrough.push( id );
      const fallback = ins[ ports[ 0 ] ] ?? childOutputs[ childOutputs.length - 1 ] ?? null;
      if ( !fallback ) throw new Error( `runGpu: op ${ node.op } 无 GPU 实现且没有可透传的输入（节点 ${ id }）` );
      outputs[ id ] = { [ firstOutName( node.op ) ]: fallback };
      continue;
    }

    const outs = impl( { nodeId: id, op, props: node.props, ins, params, childOutputs, symbols }, rt );
    if ( !outs ) throw new Error( `runGpu: ${ node.op } 的实现没有返回输出` );
    outputs[ id ] = outs;
  }

  rt.submit();

  const rootId = ir.graph.root ?? order[ order.length - 1 ];
  const rootTensor = firstOut( outputs[ rootId ] );
  if ( !rootTensor ) throw new Error( `runGpu: 根节点 ${ rootId } 没有输出` );

  const rootOutput = opts.readbackRoot === false ? new Float32Array( 0 ) : await rt.readback( rootTensor );

  return {
    rootOutput,
    outputs,
    stats: {
      dispatches: rt.dispatches,
      passBreaks: rt.passBreaks,
      plannedPasses: artifact.passes.length,
      passthrough,
    },
  };
}
