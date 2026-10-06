/**
 * 可联邦的小模型引擎。
 *
 * 接口 TrainEngine 是「训练引擎」的抽象：只要能把模型表达成
 * 一组扁平命名的 f32 参数，并支持「本地训几步 / 在指定数据上评估 / 采样」，
 * 就能接入公共训练网络。当前实现：
 *
 *   - TinyMlpEngine —— 纯 JS 字符级语言模型（embedding + 单隐层 MLP），
 *     零依赖、任何设备可跑、可在非安全上下文（局域网 http）下工作。
 *   - gpu-tinygpt —— 由工程内 WebGPU Trainer 适配（接口已预留，见 roadmap）。
 *
 * 全部方法都是 async：WebGPU 引擎的 forward/backward 需要 await 回读，
 * 接口从一开始就按异步设计，避免将来推倒重来。
 *
 * 数学约定（与 LLM 语义对齐）：
 *   给定字符序列，用 ctx 长度的窗口预测下一个字符
 *     x = concat(emb[t_{p-ctx+1..p}])   # ctx × embDim
 *     h = gelu(x · W1 + b1)             # hidden
 *     y = h · W2 + b2                   # vocabSize（logits）
 *     L = CrossEntropy(y, t_{p+1})
 *   激活用 tanh 近似的 GELU；优化器用 AdamW；精度全程 f32。
 */
import type { MlpModelSpec, ModelSpec, NamedWeights } from './protocol';
import { mulberry32 } from './rng';
import { gpuIfReady, gpuLastError } from './capability';
import { GpuTinyGptEngine } from './gpu-engine';

export { mulberry32 };

export interface TrainBatchResult
{
  loss: number;
  tokens: number;
}

export interface TrainEngine
{
  readonly spec: ModelSpec;
  stepCount: number;
  paramNames(): string[];
  paramCount(): number;
  /** 深拷贝出当前权重（可直接上网传输）。GPU 引擎是异步回读，故为 async。 */
  getWeights(): Promise<NamedWeights>;
  setWeights( w: NamedWeights ): void;
  /** 从 ids 中随机抽 batchSize 个窗口做一步 AdamW；返回平均 loss 与消耗 token 数。 */
  trainBatch( ids: Uint32Array, batchSize: number, lr: number ): Promise<TrainBatchResult>;
  /** 确定性评估：从 start 起连续 count 个窗口的平均 CE（无随机性，可跨节点对拍）。 */
  evalAt( ids: Uint32Array, start: number, count: number ): Promise<number>;
  /** 从给定前缀贪心续写 n 个字符 id。 */
  sample( prefix: number[], n: number ): Promise<number[]>;
}

// ------------------------------------------------------------------ GELU

const GELU_C = Math.sqrt( 2 / Math.PI );
const GELU_A = 0.044715;

function gelu ( x: number ): number
{
  const u = GELU_C * ( x + GELU_A * x * x * x );
  return 0.5 * x * ( 1 + Math.tanh( u ) );
}

function geluGrad ( x: number ): number
{
  const u = GELU_C * ( x + GELU_A * x * x * x );
  const t = Math.tanh( u );
  return 0.5 * ( 1 + t ) + 0.5 * x * ( 1 - t * t ) * GELU_C * ( 1 + 3 * GELU_A * x * x );
}

// ------------------------------------------------------------------ 引擎实现

export class TinyMlpEngine implements TrainEngine
{
  readonly spec: MlpModelSpec;
  stepCount = 0;

  private readonly V: number;
  private readonly C: number;
  private readonly E: number;
  private readonly H: number;
  /** 窗口展平后的输入维度 = ctx × embDim */
  private readonly X: number;

  private readonly emb: Float32Array;
  private readonly w1: Float32Array;
  private readonly b1: Float32Array;
  private readonly w2: Float32Array;
  private readonly b2: Float32Array;

  private readonly mState: NamedWeights;
  private readonly vState: NamedWeights;

  private readonly rng: () => number;

  constructor ( spec: MlpModelSpec )
  {
    this.spec = spec;
    this.V = spec.vocabSize;
    this.C = spec.ctx;
    this.E = spec.embDim;
    this.H = spec.hidden;
    this.X = spec.ctx * spec.embDim;

    const init = mulberry32( spec.seed );
    // Xavier 均匀：U(-sqrt(6/(fanIn+fanOut)), +sqrt(6/(fanIn+fanOut)))
    const xavier = ( fanIn: number, fanOut: number ): Float32Array =>
    {
      const limit = Math.sqrt( 6 / ( fanIn + fanOut ) );
      const a = new Float32Array( fanIn * fanOut );
      for ( let i = 0; i < a.length; i++ ) a[ i ] = ( init() * 2 - 1 ) * limit;
      return a;
    };

    this.emb = xavier( this.V, this.E );
    this.w1 = xavier( this.X, this.H );
    this.b1 = new Float32Array( this.H );
    this.w2 = xavier( this.H, this.V );
    this.b2 = new Float32Array( this.V );

    this.mState = {
      emb: new Float32Array( this.emb.length ),
      w1: new Float32Array( this.w1.length ),
      b1: new Float32Array( this.b1.length ),
      w2: new Float32Array( this.w2.length ),
      b2: new Float32Array( this.b2.length ),
    };
    this.vState = {
      emb: new Float32Array( this.emb.length ),
      w1: new Float32Array( this.w1.length ),
      b1: new Float32Array( this.b1.length ),
      w2: new Float32Array( this.w2.length ),
      b2: new Float32Array( this.b2.length ),
    };
    this.rng = mulberry32( spec.seed ^ 0x9e3779b9 );
  }

  paramNames (): string[]
  {
    return [ 'emb', 'w1', 'b1', 'w2', 'b2' ];
  }

  paramCount (): number
  {
    return this.emb.length + this.w1.length + this.b1.length + this.w2.length + this.b2.length;
  }

  async getWeights (): Promise<NamedWeights>
  {
    return {
      emb: this.emb.slice(),
      w1: this.w1.slice(),
      b1: this.b1.slice(),
      w2: this.w2.slice(),
      b2: this.b2.slice(),
    };
  }

  setWeights ( w: NamedWeights ): void
  {
    const want: Array<[ string, number, Float32Array ]> = [
      [ 'emb', this.emb.length, this.emb ],
      [ 'w1', this.w1.length, this.w1 ],
      [ 'b1', this.b1.length, this.b1 ],
      [ 'w2', this.w2.length, this.w2 ],
      [ 'b2', this.b2.length, this.b2 ],
    ];
    for ( const [ name, len, dst ] of want )
    {
      const src = w[ name ];
      if ( !src ) throw new Error( `setWeights: 缺少参数 ${ name }` );
      if ( src.length !== len ) throw new Error( `setWeights: ${ name } 长度 ${ src.length } != ${ len }` );
      dst.set( src );
    }
  }

  /** 取窗口第 j 个位置对应的 token id；越界左侧补 0。 */
  private windowToken ( seq: ArrayLike<number>, p: number, j: number ): number
  {
    const idx = p - this.C + 1 + j;
    return idx < 0 ? 0 : seq[ idx ];
  }

  /**
   * 单个窗口前向。
   * tokBase/xBase/hBase 是各缓冲区的起始下标；logitBase 单独给，
   * 因为 logits 是 [窗口数][vocabSize] 布局，步长与 hidden 维度不同。
   */
  private forwardWindow (
    seq: ArrayLike<number>,
    p: number,
    x: Float32Array,
    xBase: number,
    pre: Float32Array,
    h: Float32Array,
    logits: Float32Array,
    hBase: number,
    logitBase: number,
  ): void
  {
    const { C, E, H, V, X } = this;
    for ( let j = 0; j < C; j++ )
    {
      const e = this.windowToken( seq, p, j ) * E;
      for ( let r = 0; r < E; r++ ) x[ xBase + j * E + r ] = this.emb[ e + r ];
    }
    for ( let k = 0; k < H; k++ )
    {
      let s = this.b1[ k ];
      for ( let i = 0; i < X; i++ ) s += x[ xBase + i ] * this.w1[ i * H + k ];
      pre[ hBase + k ] = s;
      h[ hBase + k ] = gelu( s );
    }
    for ( let o = 0; o < V; o++ )
    {
      let s = this.b2[ o ];
      for ( let k = 0; k < H; k++ ) s += h[ hBase + k ] * this.w2[ k * V + o ];
      logits[ logitBase + o ] = s;
    }
  }

  /** 单窗口 CE（logits 已算出）。 */
  private crossEntropy ( logits: Float32Array, base: number, target: number ): number
  {
    const V = this.V;
    let mx = -Infinity;
    for ( let o = 0; o < V; o++ ) mx = Math.max( mx, logits[ base + o ] );
    let se = 0;
    for ( let o = 0; o < V; o++ ) se += Math.exp( logits[ base + o ] - mx );
    return mx + Math.log( se ) - logits[ base + target ];
  }

  async evalAt ( ids: Uint32Array, start: number, count: number ): Promise<number>
  {
    const { C, H, V, X } = this;
    if ( ids.length < C + 2 ) throw new Error( 'evalAt: 数据不足以构造窗口' );
    const x = new Float32Array( X );
    const pre = new Float32Array( H );
    const h = new Float32Array( H );
    const logits = new Float32Array( V );

    const limit = Math.min( count, ids.length - C - 1 - start );
    if ( limit <= 0 ) throw new Error( 'evalAt: 探针窗口越界' );
    let total = 0;
    for ( let i = 0; i < limit; i++ )
    {
      const p = start + i;
      this.forwardWindow( ids, p, x, 0, pre, h, logits, 0, 0 );
      total += this.crossEntropy( logits, 0, ids[ p + 1 ] );
    }
    return total / limit;
  }

  async trainBatch ( ids: Uint32Array, batchSize: number, lr: number ): Promise<TrainBatchResult>
  {
    const { C, E, H, V, X } = this;
    const span = ids.length - C - 1;
    if ( span <= 2 ) throw new Error( `trainBatch: 本地数据过短（${ ids.length } 字符）` );
    const B = Math.max( 1, batchSize );

    const x = new Float32Array( B * X );
    const pre = new Float32Array( B * H );
    const h = new Float32Array( B * H );
    const logits = new Float32Array( B * V );
    const starts = new Int32Array( B );
    const targets = new Int32Array( B );

    let lossSum = 0;
    for ( let b = 0; b < B; b++ )
    {
      const p = C - 1 + Math.floor( this.rng() * span );
      starts[ b ] = p;
      targets[ b ] = ids[ p + 1 ];
      this.forwardWindow( ids, p, x, b * X, pre, h, logits, b * H, b * V );
      lossSum += this.crossEntropy( logits, b * V, targets[ b ] );
    }

    // --- 反向：梯度累加器 ---
    const gEmb = new Float32Array( this.V * E );
    const gW1 = new Float32Array( X * H );
    const gB1 = new Float32Array( H );
    const gW2 = new Float32Array( H * V );
    const gB2 = new Float32Array( V );
    const dy = new Float32Array( V );
    const dh = new Float32Array( H );
    const dx = new Float32Array( X );

    for ( let b = 0; b < B; b++ )
    {
      const lb = b * V;
      let mx = -Infinity;
      for ( let o = 0; o < V; o++ ) mx = Math.max( mx, logits[ lb + o ] );
      let se = 0;
      for ( let o = 0; o < V; o++ ) se += Math.exp( logits[ lb + o ] - mx );
      for ( let o = 0; o < V; o++ ) dy[ o ] = Math.exp( logits[ lb + o ] - mx ) / se;
      dy[ targets[ b ] ] -= 1;

      const hb = b * H;
      const xb = b * X;
      for ( let k = 0; k < H; k++ )
      {
        let s = 0;
        const row = k * V;
        for ( let o = 0; o < V; o++ ) s += dy[ o ] * this.w2[ row + o ];
        dh[ k ] = s * geluGrad( pre[ hb + k ] );
      }
      for ( let k = 0; k < H; k++ )
      {
        const hv = h[ hb + k ];
        const row = k * V;
        for ( let o = 0; o < V; o++ ) gW2[ row + o ] += hv * dy[ o ];
      }
      for ( let o = 0; o < V; o++ ) gB2[ o ] += dy[ o ];

      for ( let i = 0; i < X; i++ )
      {
        let s = 0;
        const row = i * H;
        for ( let k = 0; k < H; k++ ) s += dh[ k ] * this.w1[ row + k ];
        dx[ i ] = s;
      }
      for ( let i = 0; i < X; i++ )
      {
        const xv = x[ xb + i ];
        const row = i * H;
        for ( let k = 0; k < H; k++ ) gW1[ row + k ] += xv * dh[ k ];
      }
      for ( let k = 0; k < H; k++ ) gB1[ k ] += dh[ k ];

      for ( let j = 0; j < C; j++ )
      {
        const t = this.windowToken( ids, starts[ b ], j ) * E;
        for ( let r = 0; r < E; r++ ) gEmb[ t + r ] += dx[ j * E + r ];
      }
    }

    const inv = 1 / B;
    this.adamw( 'emb', this.emb, gEmb, inv, lr );
    this.adamw( 'w1', this.w1, gW1, inv, lr );
    this.adamw( 'b1', this.b1, gB1, inv, lr );
    this.adamw( 'w2', this.w2, gW2, inv, lr );
    this.adamw( 'b2', this.b2, gB2, inv, lr );
    this.stepCount += 1;

    return { loss: lossSum / B, tokens: B * ( C + 1 ) };
  }

  private adamw ( name: string, param: Float32Array, grad: Float32Array, invBatch: number, lr: number ): void
  {
    const m = this.mState[ name ];
    const v = this.vState[ name ];
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    const wd = 1e-4;
    const t = this.stepCount + 1;
    const bc1 = 1 - Math.pow( b1, t );
    const bc2 = 1 - Math.pow( b2, t );
    for ( let i = 0; i < param.length; i++ )
    {
      const g = grad[ i ] * invBatch;
      m[ i ] = b1 * m[ i ] + ( 1 - b1 ) * g;
      v[ i ] = b2 * v[ i ] + ( 1 - b2 ) * g * g;
      const mh = m[ i ] / bc1;
      const vh = v[ i ] / bc2;
      param[ i ] -= lr * ( mh / ( Math.sqrt( vh ) + eps ) + wd * param[ i ] );
    }
  }

  async sample ( prefix: number[], n: number ): Promise<number[]>
  {
    const { H, V, X } = this;
    const seq = prefix.length > 0 ? prefix.slice() : [ 0 ];
    const x = new Float32Array( X );
    const pre = new Float32Array( H );
    const h = new Float32Array( H );
    const logits = new Float32Array( V );
    const out: number[] = [];

    for ( let s = 0; s < n; s++ )
    {
      const p = seq.length - 1;
      this.forwardWindow( seq, p, x, 0, pre, h, logits, 0, 0 );
      let best = 0;
      for ( let o = 1; o < V; o++ ) if ( logits[ o ] > logits[ best ] ) best = o;
      seq.push( best );
      out.push( best );
    }
    return out;
  }
}

/** 引擎工厂。房间清单里写死 engine，各节点据此构造同一个模型。 */
export function createEngine ( spec: ModelSpec ): TrainEngine
{
  switch ( spec.engine )
  {
    case 'mlp':
      return new TinyMlpEngine( spec );
    case 'gpu-tinygpt':
    {
      const gpu = gpuIfReady();
      if ( !gpu )
      {
        throw new Error(
          'WebGPU 尚未就绪，无法构造 gpu-tinygpt 引擎：' +
          ( gpuLastError() ?? '还未探测设备能力（页面启动时应先跑一次能力探测）' ),
        );
      }
      return new GpuTinyGptEngine( gpu, spec );
    }
    default:
      throw new Error( `未知引擎：${ String( ( spec as { engine?: unknown } ).engine ) }` );
  }
}
