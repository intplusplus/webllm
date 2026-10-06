/**
 * WebGPU tiny-GPT 训练引擎：把工程内已有的 `Trainer` 适配到 `TrainEngine` 接口。
 *
 * 这是「M-N1」那一步 —— 之前联邦网络跑的是纯 JS 小模型（MLP 语言模型），
 * 这个适配器一接上，房间里训的就是真正的 Transformer（自研 WGSL 前向 + 反向 + AdamW，
 * 见 src/gpu/kernels/*.wgsl 与 src/train/trainer.ts）。
 *
 * 与 MLP 引擎的差异（都写进 manifest，所有节点必须一致）：
 *   - 上下文叫 blockSize，激活是 B×T 的二维形状（MLP 是一条条独立窗口）
 *   - 一次前向的批大小受 maxBatch 约束（决定激活 arena 大小）
 *   - 权重读回是 GPU→CPU 的异步回读，所以 getWeights() 是 async
 *
 * 硬件前提（不满足就退回 CPU 引擎，见 capability.ts）：
 *   1. 页面处于安全上下文（https 或 localhost）
 *   2. navigator.gpu 存在且能拿到适配器
 */
import type { GpuContext } from '../gpu/device';
import { initWeights } from '../model/init';
import type { GPTConfig } from '../model/config';
import { Trainer } from '../train/trainer';
import { mulberry32 } from './rng';
import type { TrainBatchResult, TrainEngine } from './engine';
import type { GpuModelSpec, NamedWeights } from './protocol';

export class GpuTinyGptEngine implements TrainEngine
{
  readonly spec: GpuModelSpec;

  private readonly trainer: Trainer;
  private readonly config: GPTConfig;
  private readonly B: number;
  private readonly rng: () => number;

  constructor ( gpu: GpuContext, spec: GpuModelSpec )
  {
    this.spec = spec;
    this.B = Math.max( 1, Math.floor( spec.maxBatch ) );
    this.config = {
      vocabSize: spec.vocabSize,
      blockSize: spec.blockSize,
      nLayer: spec.nLayer,
      nHead: spec.nHead,
      nEmbd: spec.nEmbd,
      bias: spec.bias,
    };
    this.trainer = new Trainer( gpu, this.config, initWeights( this.config, spec.seed ), this.B );
    this.rng = mulberry32( spec.seed ^ 0x5f3759df );
  }

  get stepCount (): number
  {
    return this.trainer.stepCount;
  }

  paramNames (): string[]
  {
    return this.trainer.paramNames();
  }

  paramCount (): number
  {
    return this.trainer.paramCount();
  }

  /** GPU→CPU 异步回读全部参数。 */
  async getWeights (): Promise<NamedWeights>
  {
    const out: NamedWeights = {};
    for ( const name of this.trainer.paramNames() )
    {
      out[ name ] = await this.trainer.readParam( name );
    }
    return out;
  }

  setWeights ( w: NamedWeights ): void
  {
    for ( const name of this.trainer.paramNames() )
    {
      const src = w[ name ];
      if ( !src ) throw new Error( `setWeights: 缺少参数 ${ name }` );
      this.trainer.writeParam( name, src );
    }
  }

  /** 抽 B 条 blockSize 长度的连续片段，一次前向 + 反向 + AdamW。 */
  async trainBatch ( ids: Uint32Array, batchSize: number, lr: number ): Promise<TrainBatchResult>
  {
    const T = this.config.blockSize;
    const B = Math.max( 1, Math.min( Math.floor( batchSize ), this.B ) );
    const span = ids.length - T - 1;
    if ( span <= 1 ) throw new Error( `trainBatch: 本地数据过短（${ ids.length } 字符，至少需要 ${ T + 2 }）` );

    const tokens = new Uint32Array( B * T );
    const targets = new Uint32Array( B * T );
    for ( let b = 0; b < B; b++ )
    {
      const off = Math.floor( this.rng() * span );
      for ( let i = 0; i < T; i++ )
      {
        tokens[ b * T + i ] = ids[ off + i ];
        targets[ b * T + i ] = ids[ off + i + 1 ];
      }
    }

    this.trainer.forward( tokens, B, T );
    const loss = await this.trainer.loss( targets );
    this.trainer.backward( targets );
    this.trainer.step( { lr, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 } );
    return { loss, tokens: B * T };
  }

  /**
   * 确定性评估：从 start 起连续 count 个「输入片段」，每个片段内部预测下一个字符。
   * 按 maxBatch 分批走一次前向，避免激活 arena 爆掉。
   */
  async evalAt ( ids: Uint32Array, start: number, count: number ): Promise<number>
  {
    const T = this.config.blockSize;
    const limit = Math.min( count, ids.length - T - 1 - start );
    if ( limit <= 0 ) throw new Error( `evalAt: 探针窗口越界（start=${ start }，可用 ${ ids.length }）` );

    const tokens = new Uint32Array( this.B * T );
    const targets = new Uint32Array( this.B * T );
    let total = 0;
    let done = 0;
    for ( let off = 0; off < limit; off += this.B )
    {
      const n = Math.min( this.B, limit - off );
      for ( let b = 0; b < n; b++ )
      {
        const p = start + off + b;
        for ( let i = 0; i < T; i++ )
        {
          tokens[ b * T + i ] = ids[ p + i ];
          targets[ b * T + i ] = ids[ p + i + 1 ];
        }
      }
      this.trainer.forward( tokens.subarray( 0, n * T ), n, T );
      total += ( await this.trainer.loss( targets.subarray( 0, n * T ) ) ) * n;
      done += n;
    }
    return total / done;
  }

  /** 贪心续写：一次前向只走一条序列。 */
  async sample ( prefix: number[], n: number ): Promise<number[]>
  {
    const T = this.config.blockSize;
    const V = this.config.vocabSize;
    const seq = prefix.length > 0 ? prefix.slice() : [ 0 ];
    const out: number[] = [];

    for ( let s = 0; s < n; s++ )
    {
      const win = seq.slice( Math.max( 0, seq.length - T ) );
      const tokens = Uint32Array.from( win );
      this.trainer.forward( tokens, 1, win.length );
      const logits = await this.trainer.readLogits();
      const base = ( win.length - 1 ) * V;
      let best = 0;
      for ( let v = 1; v < V; v++ ) if ( logits[ base + v ] > logits[ base + best ] ) best = v;
      seq.push( best );
      out.push( best );
    }
    return out;
  }
}
