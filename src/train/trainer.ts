import type { GpuContext } from '../gpu/device';
import { Arena } from '../gpu/arena';
import
{
  alignTo,
  createStorageBuffer,
  createUniform,
  readbackF32,
  StructPacker,
  tensorResource,
  writeF32,
  writeU32,
  type Tensor,
} from '../gpu/buffer';
import { createComputePipeline, dispatch } from '../gpu/pipeline';
import { headDim, type GPTConfig } from '../model/config';
import type { GPTWeights, LayerWeights, LinearWeights } from '../model/init';

// --- 前向算子 ---
import embeddingWgsl from '../gpu/kernels/embedding/embedding.wgsl?raw';
import vecAddWgsl from '../gpu/kernels/misc/vec_add.wgsl?raw';
import layernormWgsl from '../gpu/kernels/norm/layernorm.wgsl?raw';
import gemmNtWgsl from '../gpu/kernels/gemm/gemm_nt.wgsl?raw';
import ropeWgsl from '../gpu/kernels/rope/rope.wgsl?raw';
import attentionWgsl from '../gpu/kernels/attention/attention.wgsl?raw';
import geluWgsl from '../gpu/kernels/activation/gelu.wgsl?raw';
// --- 反向算子 ---
import gemmNnWgsl from '../gpu/kernels/gemm/gemm_nn.wgsl?raw';
import gemmTnWgsl from '../gpu/kernels/gemm/gemm_tn.wgsl?raw';
import geluBwdWgsl from '../gpu/kernels/activation/gelu_bwd.wgsl?raw';
import ropeBwdWgsl from '../gpu/kernels/rope/rope_bwd.wgsl?raw';
import lnStatsWgsl from '../gpu/kernels/norm/ln_stats.wgsl?raw';
import lnDxWgsl from '../gpu/kernels/norm/ln_dx.wgsl?raw';
import lnDwdbWgsl from '../gpu/kernels/norm/ln_dwdb.wgsl?raw';
import sumRowsWgsl from '../gpu/kernels/misc/sum_rows.wgsl?raw';
import ceSoftmaxBwdWgsl from '../gpu/kernels/misc/ce_softmax_bwd.wgsl?raw';
import embeddingBwdWgsl from '../gpu/kernels/embedding/embedding_bwd.wgsl?raw';
import adamwWgsl from '../gpu/kernels/misc/adamw.wgsl?raw';
import attnBwdSoftmaxWgsl from '../gpu/kernels/attention/attn_bwd_softmax.wgsl?raw';
import attnBwdDqWgsl from '../gpu/kernels/attention/attn_bwd_dq.wgsl?raw';
import attnBwdDkdvWgsl from '../gpu/kernels/attention/attn_bwd_dkdv.wgsl?raw';

const EPS = 1e-5;
const ROPE_BASE = 10000;
/** 必须与 attention.wgsl / attn_bwd_*.wgsl 的 MAX_T 一致。 */
const MAX_T = 512;

type Res = GPUBuffer | Tensor;

export interface AdamwOptions
{
  lr: number;
  b1: number;
  b2: number;
  eps: number;
  wd: number;
}

interface Param
{
  name: string;
  buffer: GPUBuffer;
  grad: GPUBuffer;
  m: GPUBuffer;
  v: GPUBuffer;
  length: number;
  uniform: GPUBuffer;
}

interface LayerCache
{
  /** 进入该层的 x（ln1 的输入，也是残差加法的左操作数） */
  xIn: Tensor;
  ln1: Tensor;
  /** RoPE 之后的 q/k */
  qr: Tensor;
  kr: Tensor;
  /** 注意力用的 v（未经 RoPE） */
  v: Tensor;
  att: Tensor;
  xa: Tensor;
  ln2: Tensor;
  fc: Tensor;
  act: Tensor;
}

interface ForwardCache
{
  layers: LayerCache[];
  finalX: Tensor;
  lnf: Tensor;
  logits: Tensor;
}

/**
 * tiny-GPT 训练器：缓存前向激活 → 反向链式求导 → AdamW 更新。
 *
 * 与推理类 TinyGpt 完全独立（不改动已验证的前向行为）。权重、梯度、优化器状态
 * 均为独立 GPUBuffer；激活与反向临时张量走 Arena（按尺寸池化复用）。
 */
export class Trainer
{
  readonly config: GPTConfig;
  readonly maxBatch: number;

  private readonly device: GPUDevice;
  private readonly arena: Arena;
  private readonly params = new Map<string, Param>();
  private readonly dummyBias: GPUBuffer;
  private readonly inputTokens: GPUBuffer;
  private readonly inputPositions: GPUBuffer;
  private readonly targetsBuffer: GPUBuffer;
  /** backward 的逐行 loss 权重（SFT mask / DPO 系数），每次 backward 前写入。 */
  private readonly lossWeightsBuffer: GPUBuffer;
  private readonly uniformCache = new Map<string, GPUBuffer>();

  private readonly pEmbed: GPUComputePipeline;
  private readonly pAdd: GPUComputePipeline;
  private readonly pNorm: GPUComputePipeline;
  private readonly pGemmNt: GPUComputePipeline;
  private readonly pRope: GPUComputePipeline;
  private readonly pAttn: GPUComputePipeline;
  private readonly pGelu: GPUComputePipeline;
  private readonly pGemmNn: GPUComputePipeline;
  private readonly pGemmTn: GPUComputePipeline;
  private readonly pGeluBwd: GPUComputePipeline;
  private readonly pRopeBwd: GPUComputePipeline;
  private readonly pLnStats: GPUComputePipeline;
  private readonly pLnDx: GPUComputePipeline;
  private readonly pLnDwdb: GPUComputePipeline;
  private readonly pSumRows: GPUComputePipeline;
  private readonly pCe: GPUComputePipeline;
  private readonly pEmbedBwd: GPUComputePipeline;
  private readonly pAdamw: GPUComputePipeline;
  private readonly pAttnSm: GPUComputePipeline;
  private readonly pAttnDq: GPUComputePipeline;
  private readonly pAttnDkdv: GPUComputePipeline;

  private cache: ForwardCache | null = null;
  private M = 0;
  private B = 0;
  private T = 0;
  private t = 0;

  constructor ( gpu: GpuContext, config: GPTConfig, weights: GPTWeights, maxBatch = 4 )
  {
    const device = gpu.device;
    this.device = device;
    this.config = config;
    this.maxBatch = maxBatch;

    const C = config.nEmbd;
    const V = config.vocabSize;
    const M = config.blockSize * maxBatch;
    const floatsPerRow = C * 26 + V;
    const needed = alignTo( M * floatsPerRow * 4 * 2 + ( 1 << 21 ), 256 );
    this.arena = new Arena( device, needed );

    this.inputTokens = device.createBuffer( {
      label: 'train-tokens',
      size: alignTo( M * 4 ),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    } );
    this.inputPositions = device.createBuffer( {
      label: 'train-positions',
      size: alignTo( M * 4 ),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    } );
    this.targetsBuffer = device.createBuffer( {
      label: 'train-targets',
      size: alignTo( M * 4 ),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    } );
    this.lossWeightsBuffer = device.createBuffer( {
      label: 'train-loss-weights',
      size: alignTo( M * 4 ),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    } );
    this.dummyBias = device.createBuffer( {
      label: 'train-dummy-bias',
      size: alignTo( 64 * 4 ),
      usage: GPUBufferUsage.STORAGE,
    } );

    this.pEmbed = createComputePipeline( device, embeddingWgsl, 'main', 'embedding' );
    this.pAdd = createComputePipeline( device, vecAddWgsl, 'main', 'vec_add' );
    this.pNorm = createComputePipeline( device, layernormWgsl, 'main', 'layernorm' );
    this.pGemmNt = createComputePipeline( device, gemmNtWgsl, 'main', 'gemm_nt' );
    this.pRope = createComputePipeline( device, ropeWgsl, 'main', 'rope' );
    this.pAttn = createComputePipeline( device, attentionWgsl, 'main', 'attention' );
    this.pGelu = createComputePipeline( device, geluWgsl, 'main', 'gelu' );
    this.pGemmNn = createComputePipeline( device, gemmNnWgsl, 'main', 'gemm_nn' );
    this.pGemmTn = createComputePipeline( device, gemmTnWgsl, 'main', 'gemm_tn' );
    this.pGeluBwd = createComputePipeline( device, geluBwdWgsl, 'main', 'gelu_bwd' );
    this.pRopeBwd = createComputePipeline( device, ropeBwdWgsl, 'main', 'rope_bwd' );
    this.pLnStats = createComputePipeline( device, lnStatsWgsl, 'main', 'ln_stats' );
    this.pLnDx = createComputePipeline( device, lnDxWgsl, 'main', 'ln_dx' );
    this.pLnDwdb = createComputePipeline( device, lnDwdbWgsl, 'main', 'ln_dwdb' );
    this.pSumRows = createComputePipeline( device, sumRowsWgsl, 'main', 'sum_rows' );
    this.pCe = createComputePipeline( device, ceSoftmaxBwdWgsl, 'main', 'ce_softmax_bwd' );
    this.pEmbedBwd = createComputePipeline( device, embeddingBwdWgsl, 'main', 'embedding_bwd' );
    this.pAdamw = createComputePipeline( device, adamwWgsl, 'main', 'adamw' );
    this.pAttnSm = createComputePipeline( device, attnBwdSoftmaxWgsl, 'main', 'attn_bwd_softmax' );
    this.pAttnDq = createComputePipeline( device, attnBwdDqWgsl, 'main', 'attn_bwd_dq' );
    this.pAttnDkdv = createComputePipeline( device, attnBwdDkdvWgsl, 'main', 'attn_bwd_dkdv' );

    this.uploadWeights( weights );
    this.clearOptimizerState();
  }

  // ---------------------------------------------------------------- 参数管理

  async readParam ( name: string ): Promise<Float32Array>
  {
    const p = this.param( name );
    return readbackF32( this.device, p.buffer, 0, p.length );
  }

  writeParam ( name: string, data: Float32Array ): void
  {
    const p = this.param( name );
    if ( data.length !== p.length ) throw new Error( `writeParam ${ name }: 长度 ${ data.length } != ${ p.length }` );
    writeF32( this.device, p.buffer, 0, data );
  }

  /** 读回 AdamW 优化器状态（m/v），供 checkpoint 保存。 */
  async readOptimizerState ( name: string ): Promise<{ m: Float32Array; v: Float32Array }>
  {
    const p = this.param( name );
    const m = await readbackF32( this.device, p.m, 0, p.length );
    const v = await readbackF32( this.device, p.v, 0, p.length );
    return { m, v };
  }

  /** 写入 AdamW 优化器状态，供 checkpoint 加载后继续训练（heat-up 无需重复）。 */
  writeOptimizerState ( name: string, m: Float32Array, v: Float32Array ): void
  {
    const p = this.param( name );
    if ( m.length !== p.length || v.length !== p.length ) throw new Error( `writeOptimizerState ${ name }: 长度不匹配` );
    writeF32( this.device, p.m, 0, m );
    writeF32( this.device, p.v, 0, v );
  }

  /** 覆盖 AdamW 步数计数（checkpoint 恢复 bias correction 所需）。 */
  setStepCount ( t: number ): void
  {
    if ( t < 0 ) throw new Error( `setStepCount: t=${ t }` );
    this.t = t;
  }

  /** 所有注册参数的名字（checkpoint 遍历用）。 */
  paramNames (): string[]
  {
    return [ ...this.params.keys() ];
  }

  async readGrad ( name: string ): Promise<Float32Array>
  {
    const p = this.param( name );
    return readbackF32( this.device, p.grad, 0, p.length );
  }

  /** 把训练后的 GPU 参数读回成 CPU 侧 GPTWeights，供推理/采样复用。 */
  async exportWeights (): Promise<GPTWeights>
  {
    const optional = async ( name: string ): Promise<Float32Array | null> =>
      this.params.has( name ) ? this.readParam( name ) : null;
    const linear = async ( prefix: string ): Promise<LinearWeights> => ( {
      w: await this.readParam( `${ prefix }.w` ),
      b: await optional( `${ prefix }.b` ),
    } );

    const wte = await this.readParam( 'wte' );
    const wpe = await this.readParam( 'wpe' );
    const layers: LayerWeights[] = [];
    for ( let i = 0; i < this.config.nLayer; i++ )
    {
      const p = `L${ i }`;
      layers.push( {
        ln1W: await this.readParam( `${ p }.ln1W` ),
        ln1B: await this.readParam( `${ p }.ln1B` ),
        ln2W: await this.readParam( `${ p }.ln2W` ),
        ln2B: await this.readParam( `${ p }.ln2B` ),
        wq: await linear( `${ p }.wq` ),
        wk: await linear( `${ p }.wk` ),
        wv: await linear( `${ p }.wv` ),
        attnProj: await linear( `${ p }.attnProj` ),
        fc: await linear( `${ p }.fc` ),
        mlpProj: await linear( `${ p }.mlpProj` ),
      } );
    }

    return {
      wte,
      wpe,
      layers,
      lnFW: await this.readParam( 'lnFW' ),
      lnFB: await this.readParam( 'lnFB' ),
      // tie_word_embeddings：lm_head 与 wte 共享
      lmHead: { w: wte, b: await optional( 'lmHead.b' ) },
    };
  }

  /** 模型总参数量。 */
  paramCount (): number
  {
    let n = 0;
    for ( const p of this.params.values() ) n += p.length;
    return n;
  }

  /** 当前前向缓存占用的 arena 字节数（激活缓存，不含权重/梯度/优化器状态）。 */
  get arenaUsedBytes (): number
  {
    return this.arena.usedBytes;
  }

  /** 全局梯度 L2 范数（在所有参数梯度缓冲上归约）。 */
  async gradNorm (): Promise<number>
  {
    let sq = 0;
    for ( const p of this.params.values() )
    {
      const g = await readbackF32( this.device, p.grad, 0, p.length );
      for ( let i = 0; i < g.length; i++ ) sq += g[ i ] * g[ i ];
    }
    return Math.sqrt( sq );
  }

  // ---------------------------------------------------------------- 前向

  /** 执行一次完整前向并缓存激活（供 backward 使用）。 */
  forward ( tokens: Uint32Array, B: number, T: number ): void
  {
    const cfg = this.config;
    const C = cfg.nEmbd;
    const H = cfg.nHead;
    const D = headDim( cfg );
    const V = cfg.vocabSize;
    const M = B * T;

    if ( T > MAX_T ) throw new Error( `T=${ T } 超过 attention 上限 MAX_T=${ MAX_T }` );
    if ( M > cfg.blockSize * this.maxBatch )
    {
      throw new Error( `B*T=${ M } 超过 maxBatch*blockSize=${ cfg.blockSize * this.maxBatch }` );
    }
    if ( tokens.length !== M ) throw new Error( `tokens 长度 ${ tokens.length } != B*T=${ M }` );

    this.M = M;
    this.B = B;
    this.T = T;
    this.arena.reset();

    writeU32( this.device, this.inputTokens, 0, tokens );
    const positions = new Uint32Array( M );
    for ( let i = 0; i < M; i++ ) positions[ i ] = i % T;
    writeU32( this.device, this.inputPositions, 0, positions );

    const nMC = Math.ceil( ( M * C ) / 64 );
    const uEmbed = this.uniform( `embed:${ M }:${ C }`, ( p ) =>
    {
      p.u32( M );
      p.u32( C );
      p.u32( 0 );
      p.u32( 0 );
    } );
    const uNorm = this.uniform( `ln:${ M }:${ C }:${ cfg.bias ? 1 : 0 }`, ( p ) =>
    {
      p.u32( M );
      p.u32( C );
      p.f32( EPS );
      p.u32( cfg.bias ? 1 : 0 );
    } );
    const uRope = this.uniform( `rope:${ M }:${ H }:${ D }:${ T }`, ( p ) =>
    {
      p.u32( M );
      p.u32( H );
      p.u32( D );
      p.u32( T );
      p.f32( ROPE_BASE );
      p.f32( 0 );
      p.f32( 0 );
      p.f32( 0 );
    } );
    const uAttn = this.uniform( `attn:${ B }:${ T }:${ H }:${ D }`, ( p ) =>
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
    const uGelu = this.uniform( `gelu:${ M * 4 * C }`, ( p ) =>
    {
      p.u32( M * 4 * C );
      p.u32( 0 );
      p.u32( 0 );
      p.u32( 0 );
    } );

    const tokEmb = this.arena.allocFloat32( M * C, [ M, C ] );
    const posEmb = this.arena.allocFloat32( M * C, [ M, C ] );
    this.run( this.pEmbed, [ this.buf( 'wte' ), this.inputTokens, tokEmb, uEmbed ], nMC );
    this.run( this.pEmbed, [ this.buf( 'wpe' ), this.inputPositions, posEmb, uEmbed ], nMC );

    let x = this.arena.allocFloat32( M * C, [ M, C ] );
    this.run( this.pAdd, [ tokEmb, posEmb, x ], nMC );

    const layers: LayerCache[] = [];

    for ( let i = 0; i < cfg.nLayer; i++ )
    {
      const pre = `L${ i }`;
      const xIn = x;

      const ln1 = this.arena.allocFloat32( M * C, [ M, C ] );
      this.run( this.pNorm, [ x, this.buf( `${ pre }.ln1W` ), this.biasBuf( `${ pre }.ln1B` ), ln1, uNorm ], M );

      const q = this.arena.allocFloat32( M * C, [ M, C ] );
      const k = this.arena.allocFloat32( M * C, [ M, C ] );
      const v = this.arena.allocFloat32( M * C, [ M, C ] );
      this.gemmNt( ln1, this.buf( `${ pre }.wq.w` ), q, this.biasBuf( `${ pre }.wq.b` ), M, C, C, cfg.bias );
      this.gemmNt( ln1, this.buf( `${ pre }.wk.w` ), k, this.biasBuf( `${ pre }.wk.b` ), M, C, C, cfg.bias );
      this.gemmNt( ln1, this.buf( `${ pre }.wv.w` ), v, this.biasBuf( `${ pre }.wv.b` ), M, C, C, cfg.bias );

      const qr = this.arena.allocFloat32( M * C, [ M, C ] );
      const kr = this.arena.allocFloat32( M * C, [ M, C ] );
      const nRope = Math.ceil( ( M * H * ( D / 2 ) ) / 64 );
      this.run( this.pRope, [ q, qr, uRope ], nRope );
      this.run( this.pRope, [ k, kr, uRope ], nRope );

      const att = this.arena.allocFloat32( M * C, [ M, C ] );
      this.run( this.pAttn, [ qr, kr, v, att, uAttn ], [ T, H, B ] );

      const proj = this.arena.allocFloat32( M * C, [ M, C ] );
      this.gemmNt( att, this.buf( `${ pre }.attnProj.w` ), proj, this.biasBuf( `${ pre }.attnProj.b` ), M, C, C, cfg.bias );

      const xa = this.arena.allocFloat32( M * C, [ M, C ] );
      this.run( this.pAdd, [ x, proj, xa ], nMC );

      const ln2 = this.arena.allocFloat32( M * C, [ M, C ] );
      this.run( this.pNorm, [ xa, this.buf( `${ pre }.ln2W` ), this.biasBuf( `${ pre }.ln2B` ), ln2, uNorm ], M );

      const fc = this.arena.allocFloat32( M * 4 * C, [ M, 4 * C ] );
      this.gemmNt( ln2, this.buf( `${ pre }.fc.w` ), fc, this.biasBuf( `${ pre }.fc.b` ), M, 4 * C, C, cfg.bias );

      const act = this.arena.allocFloat32( M * 4 * C, [ M, 4 * C ] );
      this.run( this.pGelu, [ fc, act, uGelu ], Math.ceil( ( M * 4 * C ) / 64 ) );

      const mp = this.arena.allocFloat32( M * C, [ M, C ] );
      this.gemmNt( act, this.buf( `${ pre }.mlpProj.w` ), mp, this.biasBuf( `${ pre }.mlpProj.b` ), M, C, 4 * C, cfg.bias );

      const xb = this.arena.allocFloat32( M * C, [ M, C ] );
      this.run( this.pAdd, [ xa, mp, xb ], nMC );

      layers.push( { xIn, ln1, qr, kr, v, att, xa, ln2, fc, act } );
      x = xb;
    }

    const lnf = this.arena.allocFloat32( M * C, [ M, C ] );
    this.run( this.pNorm, [ x, this.buf( 'lnFW' ), this.biasBuf( 'lnFB' ), lnf, uNorm ], M );

    const logits = this.arena.allocFloat32( M * V, [ M, V ] );
    this.gemmNt( lnf, this.buf( 'wte' ), logits, this.biasBuf( 'lmHead.b' ), M, V, C, cfg.bias );

    this.cache = { layers, finalX: x, lnf, logits };
  }

  /** 对当前缓存的 logits 计算平均交叉熵（CPU 侧读回）。
   *
   * weights：逐行 loss 权重（可选）。
   *   - 缺省：均匀 1/M，等价平均 CE（预训练语义）
   *   - SFT：有效行 1/N、masked 行 0 —— 返回 Σw·ce/Σw
   */
  async loss ( targets: Uint32Array, weights?: Float32Array ): Promise<number>
  {
    const r = await this.weightedLossParts( targets, weights );
    return r.loss;
  }

  /**
   * 加权 CE 的两个分量：loss = Σw·ce / Σw。
   * DPO 需要 Σw·ce 不除以 Σw 的形式（token logprob 之和），故拆开返回。
   * weights 缺省时 w ≡ 1/M，Σw=1。
   */
  async weightedLossParts ( targets: Uint32Array, weights?: Float32Array ): Promise<{ loss: number; weightedSum: number; sumW: number }>
  {
    if ( !this.cache ) throw new Error( 'loss: 需先调用 forward' );
    const M = this.M;
    const V = this.config.vocabSize;
    if ( targets.length !== M ) throw new Error( `targets 长度 ${ targets.length } != B*T=${ M }` );
    if ( weights !== undefined && weights.length !== M ) throw new Error( `weights 长度 ${ weights.length } != B*T=${ M }` );
    const logits = await readbackF32( this.device, this.cache.logits.buffer, 0, M * V );
    let weightedSum = 0;
    let sumW = 0;
    for ( let m = 0; m < M; m++ )
    {
      const w = weights ? weights[ m ] : 1 / M;
      if ( w === 0 ) continue;
      const base = m * V;
      let mx = -Infinity;
      for ( let v = 0; v < V; v++ ) mx = Math.max( mx, logits[ base + v ] );
      let sum = 0;
      for ( let v = 0; v < V; v++ ) sum += Math.exp( logits[ base + v ] - mx );
      weightedSum += w * ( mx + Math.log( sum ) - logits[ base + targets[ m ] ] );
      sumW += w;
    }
    return { loss: weightedSum / sumW, weightedSum, sumW };
  }

  /** 读回当前缓存的 logits（M×V，供 DPO 逐行 token logprob 等训练逻辑使用）。 */
  async readLogits (): Promise<Float32Array>
  {
    if ( !this.cache ) throw new Error( 'readLogits: 需先调用 forward' );
    const lg = this.cache.logits;
    return readbackF32( this.device, lg.buffer, lg.offset / 4, lg.length );
  }

  // ---------------------------------------------------------------- 反向

  /**
   * 执行一次完整反向，把所有参数梯度写入各自的 grad buffer。
   *
   * weights：逐行 loss 权重（可选，与 loss() 同语义）。
   *   - 缺省：均匀 1/M（预训练）
   *   - SFT：masked 行 0（prompt 不计梯度）
   *   - DPO：chosen/rejected 行传 ±β 系数 —— 把标量 loss 对 token-logprob
   *     之和的梯度折进 CE 反向，其余链式传播完全复用
   */
  backward ( targets: Uint32Array, weights?: Float32Array ): void
  {
    const cfg = this.config;
    const C = cfg.nEmbd;
    const H = cfg.nHead;
    const D = headDim( cfg );
    const V = cfg.vocabSize;
    const M = this.M;
    const B = this.B;
    const T = this.T;
    const hasBias = cfg.bias;

    if ( !this.cache ) throw new Error( 'backward: 需先调用 forward' );
    if ( targets.length !== M ) throw new Error( `targets 长度 ${ targets.length } != B*T=${ M }` );
    if ( weights !== undefined && weights.length !== M ) throw new Error( `weights 长度 ${ weights.length } != B*T=${ M }` );
    const cache = this.cache;

    // 默认均匀 1/M（预训练语义）；SFT/DPO 由调用方按行给出
    const w = weights ?? new Float32Array( M ).fill( 1 / M );
    writeF32( this.device, this.lossWeightsBuffer, 0, w );
    writeU32( this.device, this.targetsBuffer, 0, targets );
    // wpe 梯度仅有 embedding 一条路径（就地累加），必须先清零；wte 由 gemm_tn 覆盖写。
    this.clearBuffers( [ this.param( 'wpe' ).grad ] );

    // --- 交叉熵 ---
    const dlogits = this.arena.allocFloat32( M * V, [ M, V ] );
    const uCe = this.uniform( `ce:${ M }:${ V }`, ( p ) =>
    {
      p.u32( M );
      p.u32( V );
      p.u32( 0 );
      p.u32( 0 );
    } );
    this.run( this.pCe, [ cache.logits, this.targetsBuffer, this.lossWeightsBuffer, dlogits, uCe ], M );

    // --- lm_head（wte 共享）：dWte = dY^T @ lnf；dlnf = dY @ wte ---
    this.gemmTn( dlogits, cache.lnf, this.param( 'wte' ).grad, V, C, M );
    if ( hasBias ) this.sumRows( dlogits, this.param( 'lmHead.b' ).grad, M, V );
    let dx = this.gemmNn( dlogits, this.buf( 'wte' ), M, C, V );

    // --- 最终 LayerNorm ---
    const statsF = this.arena.allocFloat32( M * 4, [ M, 4 ] );
    this.lnStats( cache.finalX, dx, this.buf( 'lnFW' ), statsF, M, C );
    this.lnDwdb( cache.finalX, dx, statsF, this.param( 'lnFW' ).grad, this.param( 'lnFB' ).grad, M, C );
    dx = this.lnDx( cache.finalX, dx, this.buf( 'lnFW' ), statsF, M, C );

    for ( let i = cfg.nLayer - 1; i >= 0; i-- )
    {
      const L = cache.layers[ i ];
      const pre = `L${ i }`;

      // x = xa + mp：dmp = dx，进入 xa 的残差路径梯度也是 dx
      // mlpProj: mp = act @ mlpProj.w^T + b
      this.gemmTn( dx, L.act, this.param( `${ pre }.mlpProj.w` ).grad, C, 4 * C, M );
      if ( hasBias ) this.sumRows( dx, this.param( `${ pre }.mlpProj.b` ).grad, M, C );
      const dact = this.gemmNn( dx, this.buf( `${ pre }.mlpProj.w` ), M, 4 * C, C );

      const dfc = this.arena.allocFloat32( M * 4 * C, [ M, 4 * C ] );
      const uGelu = this.uniform( `gelu:${ M * 4 * C }`, ( p ) =>
      {
        p.u32( M * 4 * C );
        p.u32( 0 );
        p.u32( 0 );
        p.u32( 0 );
      } );
      this.run( this.pGeluBwd, [ L.fc, dact, dfc, uGelu ], Math.ceil( ( M * 4 * C ) / 64 ) );

      // fc: fc = ln2 @ fc.w^T + b
      this.gemmTn( dfc, L.ln2, this.param( `${ pre }.fc.w` ).grad, 4 * C, C, M );
      if ( hasBias ) this.sumRows( dfc, this.param( `${ pre }.fc.b` ).grad, M, 4 * C );
      const dln2 = this.gemmNn( dfc, this.buf( `${ pre }.fc.w` ), M, C, 4 * C );

      // ln2 反向
      const stats2 = this.arena.allocFloat32( M * 4, [ M, 4 ] );
      this.lnStats( L.xa, dln2, this.buf( `${ pre }.ln2W` ), stats2, M, C );
      this.lnDwdb( L.xa, dln2, stats2, this.param( `${ pre }.ln2W` ).grad, this.param( `${ pre }.ln2B` ).grad, M, C );
      const dxa2 = this.lnDx( L.xa, dln2, this.buf( `${ pre }.ln2W` ), stats2, M, C );

      // dxa = dx（残差加路径） + dxa2（ln2 路径）
      const dxa = this.arena.allocFloat32( M * C, [ M, C ] );
      this.vecAdd( dx, dxa2, dxa, M * C );

      // attnProj: proj = att @ attnProj.w^T + b
      this.gemmTn( dxa, L.att, this.param( `${ pre }.attnProj.w` ).grad, C, C, M );
      if ( hasBias ) this.sumRows( dxa, this.param( `${ pre }.attnProj.b` ).grad, M, C );
      const datt = this.gemmNn( dxa, this.buf( `${ pre }.attnProj.w` ), M, C, C );

      // 注意力反向
      const uAttn = this.attnUniform( B, T, H, D );
      const ps = this.arena.allocFloat32( B * H * T * T, [ B, H, T, T ] );
      const ds = this.arena.allocFloat32( B * H * T * T, [ B, H, T, T ] );
      this.run( this.pAttnSm, [ L.qr, L.kr, L.v, datt, ps, ds, uAttn ], [ T, H, B ] );

      const dqr = this.arena.allocFloat32( M * C, [ M, C ] );
      this.run( this.pAttnDq, [ ds, L.kr, dqr, uAttn ], [ T, H, B ] );
      const dkr = this.arena.allocFloat32( M * C, [ M, C ] );
      const dv = this.arena.allocFloat32( M * C, [ M, C ] );
      this.run( this.pAttnDkdv, [ ds, ps, L.qr, datt, dkr, dv, uAttn ], [ T, H, B ] );

      // RoPE 反向
      const uRope = this.ropeUniform( M, H, D, T );
      const dq = this.arena.allocFloat32( M * C, [ M, C ] );
      const dk = this.arena.allocFloat32( M * C, [ M, C ] );
      const nRope = Math.ceil( ( M * H * ( D / 2 ) ) / 64 );
      this.run( this.pRopeBwd, [ dqr, dq, uRope ], nRope );
      this.run( this.pRopeBwd, [ dkr, dk, uRope ], nRope );

      // q/k/v 线性层
      this.gemmTn( dq, L.ln1, this.param( `${ pre }.wq.w` ).grad, C, C, M );
      this.gemmTn( dk, L.ln1, this.param( `${ pre }.wk.w` ).grad, C, C, M );
      this.gemmTn( dv, L.ln1, this.param( `${ pre }.wv.w` ).grad, C, C, M );
      if ( hasBias )
      {
        this.sumRows( dq, this.param( `${ pre }.wq.b` ).grad, M, C );
        this.sumRows( dk, this.param( `${ pre }.wk.b` ).grad, M, C );
        this.sumRows( dv, this.param( `${ pre }.wv.b` ).grad, M, C );
      }
      const dln1a = this.gemmNn( dq, this.buf( `${ pre }.wq.w` ), M, C, C );
      const dln1b = this.gemmNn( dk, this.buf( `${ pre }.wk.w` ), M, C, C );
      const dln1c = this.gemmNn( dv, this.buf( `${ pre }.wv.w` ), M, C, C );
      const dln1ab = this.arena.allocFloat32( M * C, [ M, C ] );
      this.vecAdd( dln1a, dln1b, dln1ab, M * C );
      const dln1 = this.arena.allocFloat32( M * C, [ M, C ] );
      this.vecAdd( dln1ab, dln1c, dln1, M * C );

      // ln1 反向
      const stats1 = this.arena.allocFloat32( M * 4, [ M, 4 ] );
      this.lnStats( L.xIn, dln1, this.buf( `${ pre }.ln1W` ), stats1, M, C );
      this.lnDwdb( L.xIn, dln1, stats1, this.param( `${ pre }.ln1W` ).grad, this.param( `${ pre }.ln1B` ).grad, M, C );
      const dxprev1 = this.lnDx( L.xIn, dln1, this.buf( `${ pre }.ln1W` ), stats1, M, C );

      // 进入 xIn 的梯度 = 残差加路径(dxa) + ln1 路径(dxprev1)
      const dxnext = this.arena.allocFloat32( M * C, [ M, C ] );
      this.vecAdd( dxa, dxprev1, dxnext, M * C );
      dx = dxnext;
    }

    // --- 词嵌入反向：dx（初始 x 的梯度）经恒等加法分别流入 tokEmb / posEmb ---
    const uWte = this.uniform( `embbwd:${ M }:${ C }:${ V }`, ( p ) =>
    {
      p.u32( M );
      p.u32( C );
      p.u32( V );
      p.u32( 0 );
    } );
    this.run( this.pEmbedBwd, [ this.inputTokens, dx, this.param( 'wte' ).grad, uWte ], V );

    const bs = cfg.blockSize;
    const uWpe = this.uniform( `embbwd:${ M }:${ C }:${ bs }`, ( p ) =>
    {
      p.u32( M );
      p.u32( C );
      p.u32( bs );
      p.u32( 0 );
    } );
    this.run( this.pEmbedBwd, [ this.inputPositions, dx, this.param( 'wpe' ).grad, uWpe ], bs );
  }

  // ---------------------------------------------------------------- 优化器

  /** AdamW 单步（in-place 更新参数，并推进步数 t）。 */
  step ( opts: AdamwOptions ): void
  {
    this.t += 1;
    const b1t = Math.pow( opts.b1, this.t );
    const b2t = Math.pow( opts.b2, this.t );
    for ( const p of this.params.values() )
    {
      this.writeUniform( p.uniform, ( pk ) =>
      {
        pk.u32( p.length );
        pk.u32( 0 );
        pk.u32( 0 );
        pk.u32( 0 );
        pk.f32( opts.lr );
        pk.f32( opts.b1 );
        pk.f32( opts.b2 );
        pk.f32( opts.eps );
        pk.f32( opts.wd );
        pk.f32( b1t );
        pk.f32( b2t );
        pk.f32( 0 );
      } );
      this.run(
        this.pAdamw,
        [ this.buf( p.name ), this.param( p.name ).grad, this.param( p.name ).m, this.param( p.name ).v, p.uniform ],
        Math.ceil( p.length / 64 ),
      );
    }
  }

  get stepCount (): number
  {
    return this.t;
  }

  // ---------------------------------------------------------------- 内部工具

  private param ( name: string ): Param
  {
    const p = this.params.get( name );
    if ( !p ) throw new Error( `缺少参数: ${ name }` );
    return p;
  }

  private buf ( name: string ): GPUBuffer
  {
    return this.param( name ).buffer;
  }

  private biasBuf ( name: string ): Res
  {
    return this.params.get( name )?.buffer ?? this.dummyBias;
  }

  private alloc ( n: number, shape?: number[] ): Tensor
  {
    return this.arena.allocFloat32( n, shape );
  }

  // --- 各算子封装（Res 可为 GPUBuffer 或 {buffer,offset,size}） ---

  private vecAdd ( a: Res, b: Res, out: Res, n: number ): void
  {
    this.run( this.pAdd, [ a, b, out ], Math.ceil( n / 64 ) );
  }

  private gemmNt ( A: Res, B: Res, C: Res, bias: Res, M: number, N: number, K: number, hasBias: boolean ): void
  {
    this.gemm( this.pGemmNt, A, B, C, bias, M, N, K, hasBias );
  }

  private gemmNn ( A: Res, B: Res, M: number, N: number, K: number ): Tensor
  {
    const C = this.alloc( M * N, [ M, N ] );
    this.gemm( this.pGemmNn, A, B, C, this.dummyBias, M, N, K, false );
    return C;
  }

  /** 权重梯度：dW[out,in] = dY[rows,out]^T @ X[rows,in]。M=out、N=in、K=rows。 */
  private gemmTn ( A: Res, B: Res, C: Res, M: number, N: number, K: number ): void
  {
    this.gemm( this.pGemmTn, A, B, C, this.dummyBias, M, N, K, false );
  }

  private gemm (
    pipeline: GPUComputePipeline,
    A: Res,
    B: Res,
    C: Res,
    bias: Res,
    M: number,
    N: number,
    K: number,
    hasBias: boolean,
  ): void
  {
    const u = this.uniform( `gemm:${ M }:${ N }:${ K }:${ hasBias ? 1 : 0 }`, ( p ) =>
    {
      p.u32( M );
      p.u32( N );
      p.u32( K );
      p.u32( hasBias ? 1 : 0 );
    } );
    this.run( pipeline, [ A, B, C, bias, u ], [ Math.ceil( M / 64 ), Math.ceil( N / 64 ), 1 ] );
  }

  private sumRows ( x: Res, out: Res, rows: number, cols: number ): void
  {
    const u = this.uniform( `sumrows:${ rows }:${ cols }`, ( p ) =>
    {
      p.u32( rows );
      p.u32( cols );
      p.u32( 0 );
      p.u32( 0 );
    } );
    this.run( this.pSumRows, [ x, out, u ], cols );
  }

  private lnStats ( x: Res, dy: Res, w: Res, stats: Res, rows: number, D: number ): void
  {
    const u = this.uniform( `lnstats:${ rows }:${ D }`, ( p ) =>
    {
      p.u32( rows );
      p.u32( D );
      p.f32( EPS );
      p.u32( 0 );
    } );
    this.run( this.pLnStats, [ x, dy, w, stats, u ], rows );
  }

  private lnDx ( x: Res, dy: Res, w: Res, stats: Res, rows: number, D: number ): Tensor
  {
    const dx = this.alloc( rows * D, [ rows, D ] );
    const u = this.uniform( `lndx:${ rows }:${ D }`, ( p ) =>
    {
      p.u32( rows );
      p.u32( D );
      p.u32( 0 );
      p.u32( 0 );
    } );
    this.run( this.pLnDx, [ x, dy, w, stats, dx, u ], Math.ceil( ( rows * D ) / 64 ) );
    return dx;
  }

  private lnDwdb ( x: Res, dy: Res, stats: Res, dw: Res, db: Res, rows: number, D: number ): void
  {
    const u = this.uniform( `lndwdb:${ rows }:${ D }`, ( p ) =>
    {
      p.u32( rows );
      p.u32( D );
      p.u32( 0 );
      p.u32( 0 );
    } );
    this.run( this.pLnDwdb, [ x, dy, stats, dw, db, u ], D );
  }

  private attnUniform ( B: number, T: number, H: number, D: number ): GPUBuffer
  {
    return this.uniform( `attnbwd:${ B }:${ T }:${ H }:${ D }`, ( p ) =>
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
  }

  private ropeUniform ( rows: number, H: number, D: number, T: number ): GPUBuffer
  {
    return this.uniform( `ropebwd:${ rows }:${ H }:${ D }:${ T }`, ( p ) =>
    {
      p.u32( rows );
      p.u32( H );
      p.u32( D );
      p.u32( T );
      p.f32( ROPE_BASE );
      p.f32( 0 );
      p.f32( 0 );
      p.f32( 0 );
    } );
  }

  private run ( pipeline: GPUComputePipeline, resources: Res[], workgroups: number | readonly number[] ): void
  {
    const entries: GPUBindGroupEntry[] = resources.map( ( resource, binding ) => ( {
      binding,
      resource: 'buffer' in resource ? tensorResource( resource ) : { buffer: resource },
    } ) );
    const bindGroup = this.device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries } );
    dispatch( this.device, pipeline, bindGroup, workgroups );
  }

  private uniform ( key: string, build: ( p: StructPacker ) => void ): GPUBuffer
  {
    let buffer = this.uniformCache.get( key );
    if ( !buffer )
    {
      const packer = new StructPacker();
      build( packer );
      buffer = createUniform( this.device, packer.bytes(), key );
      this.uniformCache.set( key, buffer );
    }
    return buffer;
  }

  private writeUniform ( buffer: GPUBuffer, build: ( p: StructPacker ) => void ): void
  {
    const packer = new StructPacker();
    build( packer );
    const bytes = packer.bytes();
    this.device.queue.writeBuffer( buffer, 0, bytes.buffer as ArrayBuffer, bytes.byteOffset, bytes.byteLength );
  }

  private clearBuffers ( buffers: GPUBuffer[] ): void
  {
    const encoder = this.device.createCommandEncoder();
    for ( const b of buffers ) encoder.clearBuffer( b );
    this.device.queue.submit( [ encoder.finish() ] );
  }

  private register ( name: string, data: Float32Array ): void
  {
    const n = data.length;
    const buffer = createStorageBuffer( this.device, n * 4, name );
    writeF32( this.device, buffer, 0, data );
    const grad = createStorageBuffer( this.device, n * 4, `${ name }.grad` );
    const m = createStorageBuffer( this.device, n * 4, `${ name }.m` );
    const v = createStorageBuffer( this.device, n * 4, `${ name }.v` );
    const uniform = createUniform( this.device, new Uint8Array( 64 ), `${ name }.adamw` );
    this.params.set( name, { name, buffer, grad, m, v, length: n, uniform } );
  }

  private clearOptimizerState (): void
  {
    const encoder = this.device.createCommandEncoder();
    for ( const p of this.params.values() )
    {
      encoder.clearBuffer( p.m );
      encoder.clearBuffer( p.v );
    }
    this.device.queue.submit( [ encoder.finish() ] );
  }

  private uploadLinear ( prefix: string, w: Float32Array, b: Float32Array | null ): void
  {
    this.register( `${ prefix }.w`, w );
    if ( b ) this.register( `${ prefix }.b`, b );
  }

  private uploadWeights ( weights: GPTWeights ): void
  {
    // 注意：lmHead.w 与 wte 共享；参数只注册一次 wte，反向时两条路径的梯度都汇入 grad(wte)。
    this.register( 'wte', weights.wte );
    this.register( 'wpe', weights.wpe );
    this.register( 'lnFW', weights.lnFW );
    this.register( 'lnFB', weights.lnFB );
    if ( weights.lmHead.b ) this.register( 'lmHead.b', weights.lmHead.b );

    weights.layers.forEach( ( L, i ) =>
    {
      const p = `L${ i }`;
      this.register( `${ p }.ln1W`, L.ln1W );
      this.register( `${ p }.ln1B`, L.ln1B );
      this.register( `${ p }.ln2W`, L.ln2W );
      this.register( `${ p }.ln2B`, L.ln2B );
      this.uploadLinear( `${ p }.wq`, L.wq.w, L.wq.b );
      this.uploadLinear( `${ p }.wk`, L.wk.w, L.wk.b );
      this.uploadLinear( `${ p }.wv`, L.wv.w, L.wv.b );
      this.uploadLinear( `${ p }.attnProj`, L.attnProj.w, L.attnProj.b );
      this.uploadLinear( `${ p }.fc`, L.fc.w, L.fc.b );
      this.uploadLinear( `${ p }.mlpProj`, L.mlpProj.w, L.mlpProj.b );
    } );
  }
}