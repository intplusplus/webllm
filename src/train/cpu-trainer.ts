/**
 * 纯 CPU 训练器 —— 无 WebGPU 环境的兜底，也是 Spec IR 全链路的第一个"真实消费者"。
 *
 * 与 GPU Trainer（./trainer.ts）API 同构：训练台的三阶段管线
 * （预训练 / SFT / DPO / checkpoint / 对话）不改调用方式即可切换实现。
 *
 * 实现立场：
 *  - 前向/反向全部走 Spec IR 的 infer → plan → emit → run / backward，
 *    与 verify:ir 自检套件（54 项，含 GRAD-V7 加权损失回归）同一套代码路径。
 *  - 加权损失语义（SFT 掩码 / DPO 系数）由 IR CrossEntropy 的可选 `weight` 端口承载：
 *    loss = Σw·ce/Σw，与 GPU 版 `weightedLossParts` 严格对齐。
 *  - 优化器是自带的可序列化 AdamW（m/v/stepCount 可导出导入），
 *    公式与 grad.ts `createAdamW` / GPU kernel 同款；不做全局裁剪（与 GPU AdamwOptions 一致）。
 *  - artifact 按 (B,T) 缓存：IR 产物是符号形状特化的，(B,T) 一变必须重编译；
 *    缓存让「编译一次、N 步复用」（ENG-V1）在训练循环里成立。
 *  - tie（lmHead.w ↔ wte）依赖 buildGptIr 的**同一 Float32Array 引用**：
 *    writeParam 一律就地写（set，不换引用），别名自动同步；
 *    优化器状态按张量身份归并到规范名（与 backward 的归并规则同源）。
 */

import type { InferResult, Model } from '../ir/types';
import type { Artifact } from '../ir/emit';
import { emit } from '../ir/emit';
import { run, type CpuImplRegistry, type RunBinding } from '../ir/exec';
import { plan, type BackendCapability } from '../ir/plan';
import { infer } from '../ir/infer';
import { builtinCpuImpls } from '../ir/cpu-impls';
import { builtinCpuGrads } from '../ir/cpu-grads';
import { asF32, backward, type GradRegistry } from '../ir/grad';
import type { TensorValue } from '../ir/binding';
import { buildGptIr, bindBatch, type GptIr } from '../ir/tinygpt-ir';
import {
  attachCrossEntropy,
  attachCrossEntropyWeighted,
  setTargets,
  setLossWeights,
} from '../ir/train';
import type { GPTConfig } from '../model/config';
import type { GPTWeights, LinearWeights, LayerWeights } from '../model/init';

/** 与 GPU Trainer 的 AdamwOptions 同构（不含 clip：GPU kernel 也不做全局裁剪）。 */
export interface AdamwOptions
{
  lr: number;
  b1: number;
  b2: number;
  eps: number;
  wd: number;
}

/**
 * GPU Trainer 与 CpuTrainer 的公共调用面。
 * GPU Trainer 结构上满足本接口（方法签名一致），app.ts 据此零成本切换。
 */
export interface TrainCore
{
  readonly config: GPTConfig;
  readonly stepCount: number;
  paramNames (): string[];
  forward ( tokens: Uint32Array, B: number, T: number ): void;
  loss ( targets: Uint32Array, weights?: Float32Array ): Promise<number>;
  backward ( targets: Uint32Array, weights?: Float32Array ): void;
  step ( opts: AdamwOptions ): void;
  readLogits (): Promise<Float32Array>;
  readParam ( name: string ): Promise<Float32Array>;
  writeParam ( name: string, data: Float32Array ): void;
  readOptimizerState ( name: string ): Promise<{ m: Float32Array; v: Float32Array }>;
  writeOptimizerState ( name: string, m: Float32Array, v: Float32Array ): void;
  exportWeights (): Promise<GPTWeights>;
  setStepCount ( t: number ): void;
}

interface Compiled
{
  model: Model;
  ir: InferResult;
  artifact: Artifact;
  binding: RunBinding;
}

function cpuBackend (): BackendCapability
{
  return {
    backend: 'cpu',
    maxBufferSize: 512 * 1024 * 1024,
    maxBindGroups: 4,
    maxWorkgroupsPerDimension: 65535,
    supportsF16: false,
    supportsSubgroups: false,
  };
}

export class CpuTrainer implements TrainCore
{
  readonly config: GPTConfig;

  private readonly gpt: GptIr;
  private readonly modelPlain: Model;   // root = head（纯前向，出 logits）
  private readonly modelLoss: Model;    // root = CrossEntropy（mean）
  private readonly modelLossW: Model;   // root = CrossEntropy + weight 端口
  private readonly impls: CpuImplRegistry;
  private readonly grads: GradRegistry;

  /** (B,T) → 编译产物。IR 产物形状特化，符号变了必须重编。 */
  private readonly fwdCache = new Map<string, Compiled>();
  private readonly bwdPlainCache = new Map<string, Compiled>();
  private readonly bwdWeightedCache = new Map<string, Compiled>();

  /** tie 别名 → 规范名（按张量身份；与 backward 的归并同源，键集与 paramNames 对齐）。 */
  private readonly canonical = new Map<string, string>();

  /**
   * 参数名映射：GPU 契约名（`L0.wq.w`）↔ IR 张量表名（`layers.0.wq.w`）。
   * GPU Trainer 的 checkpoint/paramNames 用前者，buildGptIr 的表用后者 ——
   * CpuTrainer 对外一律说 GPU 名（checkpoint 互通），对内一律查 IR 表。
   */
  private readonly gpuToIr = new Map<string, string>();
  private readonly irToGpu = new Map<string, string>();

  /** AdamW 状态：键为规范参数名（与 paramNames 一致，checkpoint 往返无损）。 */
  private readonly mState = new Map<string, Float32Array>();
  private readonly vState = new Map<string, Float32Array>();
  private stepCounter = 0;

  /** 参数名单（与 GPU uploadWeights 的注册集逐字对齐，checkpoint 互通的前提）。 */
  private readonly paramList: string[];

  // 前向缓存（GPU 版 cache 的 CPU 等价物）
  private lastTokens: Uint32Array | null = null;
  private lastB = 0;
  private lastT = 0;
  private lastLogits: Float32Array | null = null;
  private lastGrad: Record<string, Float32Array> | null = null;

  constructor ( config: GPTConfig, weights: GPTWeights )
  {
    this.config = config;
    this.gpt = buildGptIr( config, weights );
    this.modelPlain = this.gpt.model;
    this.modelLoss = attachCrossEntropy( this.gpt.model );
    this.modelLossW = attachCrossEntropyWeighted( this.modelLoss );
    this.impls = builtinCpuImpls();
    this.grads = builtinCpuGrads();

    // 参数名单：镜像 GPU uploadWeights 的注册顺序与命名（lmHead.w 不注册，tie 走 wte）
    // 同时建 GPU 名 ↔ IR 表名的双向映射。
    const names: string[] = [];
    const has = ( gpuName: string, irName: string ): void =>
    {
      names.push( gpuName );
      this.gpuToIr.set( gpuName, irName );
      this.irToGpu.set( irName, gpuName );
    };
    has( 'wte', 'wte' );
    has( 'wpe', 'wpe' );
    has( 'lnFW', 'lnFW' );
    if ( config.bias ) has( 'lnFB', 'lnFB' );
    if ( config.bias && weights.lmHead.b ) has( 'lmHead.b', 'lmHead.b' );
    weights.layers.forEach( ( _, i ) =>
    {
      const g = `L${ i }`;
      const ir = `layers.${ i }`;
      has( `${ g }.ln1W`, `${ ir }.ln1W` );
      if ( config.bias ) has( `${ g }.ln1B`, `${ ir }.ln1B` );
      has( `${ g }.ln2W`, `${ ir }.ln2W` );
      if ( config.bias ) has( `${ g }.ln2B`, `${ ir }.ln2B` );
      for ( const k of [ 'wq', 'wk', 'wv', 'attnProj', 'fc', 'mlpProj' ] )
      {
        has( `${ g }.${ k }.w`, `${ ir }.${ k }.w` );
        if ( config.bias ) has( `${ g }.${ k }.b`, `${ ir }.${ k }.b` );
      }
    } );
    this.paramList = names;

    // tie 归并：按张量身份（与 grad.ts backward 的规则同源）。
    // wte 先入表（buildGptIr 的 set 顺序）⇒ lmHead.w → wte，优化器状态不重复。
    const byData = new Map<Float32Array, string>();
    for ( const name of this.gpt.tensors.names() )
    {
      const t = this.gpt.tensors.get( name );
      if ( !t || !( t.data instanceof Float32Array ) ) continue;
      const prev = byData.get( t.data );
      if ( prev === undefined ) byData.set( t.data, name );
      else this.canonical.set( name, prev );
    }
  }

  // ------------------------------------------------------------ 编译

  private compile ( model: Model, B: number, T: number ): Compiled
  {
    const ir = infer( model, { symbols: { B, T } } );
    const errs = ir.diags.filter( ( d ) => d.level === 'error' );
    if ( errs.length > 0 )
      throw new Error( `CpuTrainer 编译失败：${ errs.map( ( d ) => d.code ).join( ',' ) }` );
    const artifact = emit( model, ir, plan( model, ir, { backends: [ cpuBackend() ] } ) );
    return { model, ir, artifact, binding: { tensors: this.gpt.tensors, symbols: { B, T } } };
  }

  private cached ( cache: Map<string, Compiled>, model: Model, B: number, T: number ): Compiled
  {
    const key = `${ B }x${ T }`;
    let c = cache.get( key );
    if ( !c ) { c = this.compile( model, B, T ); cache.set( key, c ); }
    return c;
  }

  // ------------------------------------------------------------ 前向 / 损失

  forward ( tokens: Uint32Array, B: number, T: number ): void
  {
    if ( tokens.length !== B * T )
      throw new Error( `CpuTrainer.forward: tokens 长度 ${ tokens.length } != B*T=${ B * T }` );
    const c = this.cached( this.fwdCache, this.modelPlain, B, T );
    bindBatch( this.gpt, tokens, B, T );
    const res = run( c.model, c.ir, c.artifact, c.binding, this.impls );
    // plain 图的根就是 head ⇒ rootOutput 即 logits
    this.lastLogits = asF32( res.rootOutput ).slice();
    this.lastTokens = tokens.slice();
    this.lastB = B;
    this.lastT = T;
  }

  /** 前向 + 读 logits 一步到位（DPO 的冻结 reference 副本用）。 */
  async readForward ( tokens: Uint32Array, B: number, T: number ): Promise<Float32Array>
  {
    this.forward( tokens, B, T );
    return this.readLogits();
  }

  async readLogits (): Promise<Float32Array>
  {
    if ( !this.lastLogits ) throw new Error( 'CpuTrainer.readLogits: 需先调用 forward' );
    return this.lastLogits.slice();
  }

  /** 平均 / 加权交叉熵（与 GPU weightedLossParts 语义一致）。 */
  async loss ( targets: Uint32Array, weights?: Float32Array ): Promise<number>
  {
    const r = await this.weightedLossParts( targets, weights );
    return r.loss;
  }

  /**
   * 加权 CE 的两个分量：loss = Σw·ce / Σw。
   * weights 缺省时 w ≡ 1/M（Σw=1，weightedSum 即平均 CE）。
   */
  async weightedLossParts ( targets: Uint32Array, weights?: Float32Array ): Promise<{ loss: number; weightedSum: number; sumW: number }>
  {
    if ( !this.lastLogits ) throw new Error( 'CpuTrainer.loss: 需先调用 forward' );
    const M = this.lastB * this.lastT;
    const V = this.config.vocabSize;
    if ( targets.length !== M ) throw new Error( `targets 长度 ${ targets.length } != B*T=${ M }` );
    if ( weights !== undefined && weights.length !== M ) throw new Error( `weights 长度 ${ weights.length } != B*T=${ M }` );
    const logits = this.lastLogits;
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
    if ( !( sumW > 0 ) ) throw new Error( 'CpuTrainer.loss: 行权重之和必须 > 0' );
    return { loss: weightedSum / sumW, weightedSum, sumW };
  }

  // ------------------------------------------------------------ 反向 / 更新

  backward ( targets: Uint32Array, weights?: Float32Array ): void
  {
    if ( !this.lastTokens ) throw new Error( 'CpuTrainer.backward: 需先调用 forward' );
    const B = this.lastB;
    const T = this.lastT;
    // 表里重绑 tokens/targets（幂等；forward 之后表可能被读路径动过，稳妥起见重写）
    bindBatch( this.gpt, this.lastTokens, B, T );
    setTargets( this.gpt.tensors, targets, B, T );

    let c: Compiled;
    if ( weights !== undefined )
    {
      setLossWeights( this.gpt.tensors, weights, B, T );
      c = this.cached( this.bwdWeightedCache, this.modelLossW, B, T );
    }
    else
    {
      c = this.cached( this.bwdPlainCache, this.modelLoss, B, T );
    }

    const b = backward( c.model, c.ir, c.artifact, c.binding, this.impls, this.grads );
    this.lastGrad = b.dParams;
  }

  step ( opts: AdamwOptions ): void
  {
    if ( !this.lastGrad ) throw new Error( 'CpuTrainer.step: 需先调用 backward' );
    const grads = this.lastGrad;
    this.lastGrad = null;
    this.stepCounter += 1;

    const b1t = Math.pow( opts.b1, this.stepCounter );
    const b2t = Math.pow( opts.b2, this.stepCounter );
    const bc1 = 1 - b1t;
    const bc2 = 1 - b2t;

    for ( const irKey0 of Object.keys( grads ).sort() )
    {
      // dParams 键是 IR 表名 → tie 归并（IR 名）→ 优化器状态用 GPU 契约名
      const irKey = this.canonical.get( irKey0 ) ?? irKey0;
      const g = grads[ irKey0 ];
      const tv = this.gpt.tensors.get( irKey );
      if ( !tv || !( tv.data instanceof Float32Array ) ) continue;
      const k = this.irToGpu.get( irKey ) ?? irKey;
      const p = tv.data;
      if ( p.length !== g.length ) continue;

      let mi = this.mState.get( k );
      let vi = this.vState.get( k );
      if ( !mi || mi.length !== g.length ) { mi = new Float32Array( g.length ); this.mState.set( k, mi ); }
      if ( !vi || vi.length !== g.length ) { vi = new Float32Array( g.length ); this.vState.set( k, vi ); }

      for ( let i = 0; i < p.length; i++ )
      {
        // 与 grad.ts createAdamW 同款：就地更新（tie 别名共享同一内存，天然同步）
        mi[ i ] = opts.b1 * mi[ i ] + ( 1 - opts.b1 ) * g[ i ];
        vi[ i ] = opts.b2 * vi[ i ] + ( 1 - opts.b2 ) * g[ i ] * g[ i ];
        const mhat = mi[ i ] / bc1;
        const vhat = vi[ i ] / bc2;
        p[ i ] = p[ i ] - opts.lr * ( mhat / ( Math.sqrt( vhat ) + opts.eps ) + opts.wd * p[ i ] );
      }
    }
  }

  get stepCount (): number
  {
    return this.stepCounter;
  }

  setStepCount ( t: number ): void
  {
    if ( t < 0 ) throw new Error( `setStepCount: t=${ t }` );
    this.stepCounter = t;
  }

  // ------------------------------------------------------------ 参数管理

  paramNames (): string[]
  {
    return [ ...this.paramList ];
  }

  paramCount (): number
  {
    let n = 0;
    for ( const name of this.paramList )
    {
      const t = this.requireParam( name );
      n += t.data.length;
    }
    return n;
  }

  async readParam ( name: string ): Promise<Float32Array>
  {
    const t = this.requireParam( name );
    return asF32( t ).slice();
  }

  /** 就地写（set 而非换引用）—— tie 别名（lmHead.w ↔ wte）必须同步，断引用即断 tie。 */
  writeParam ( name: string, data: Float32Array ): void
  {
    const t = this.requireParam( name );
    const p = asF32( t );
    if ( data.length !== p.length ) throw new Error( `writeParam ${ name }: 长度 ${ data.length } != ${ p.length }` );
    p.set( data );
  }

  async readOptimizerState ( name: string ): Promise<{ m: Float32Array; v: Float32Array }>
  {
    const t = this.requireParam( name );
    const len = t.data.length;
    const m = this.mState.get( name );
    const v = this.vState.get( name );
    return {
      m: m ? m.slice() : new Float32Array( len ),
      v: v ? v.slice() : new Float32Array( len ),
    };
  }

  writeOptimizerState ( name: string, m: Float32Array, v: Float32Array ): void
  {
    const t = this.requireParam( name );
    const len = t.data.length;
    if ( m.length !== len || v.length !== len ) throw new Error( `writeOptimizerState ${ name }: 长度不匹配` );
    this.mState.set( name, m.slice() );
    this.vState.set( name, v.slice() );
  }

  /** 把当前参数读回成 GPTWeights（lmHead.w 与 wte 同引用，tie 语义保持）。 */
  async exportWeights (): Promise<GPTWeights>
  {
    const optional = ( name: string ): Float32Array | null =>
      this.paramList.includes( name ) ? this.readParamSync( name ) : null;
    const linear = ( prefix: string ): LinearWeights => ( {
      w: this.readParamSync( `${ prefix }.w` ),
      b: optional( `${ prefix }.b` ),
    } );

    const wte = this.readParamSync( 'wte' );
    const layers: LayerWeights[] = [];
    for ( let i = 0; i < this.config.nLayer; i++ )
    {
      const p = `L${ i }`;
      layers.push( {
        ln1W: this.readParamSync( `${ p }.ln1W` ),
        ln1B: this.readParamSync( `${ p }.ln1B` ),
        ln2W: this.readParamSync( `${ p }.ln2W` ),
        ln2B: this.readParamSync( `${ p }.ln2B` ),
        wq: linear( `${ p }.wq` ),
        wk: linear( `${ p }.wk` ),
        wv: linear( `${ p }.wv` ),
        attnProj: linear( `${ p }.attnProj` ),
        fc: linear( `${ p }.fc` ),
        mlpProj: linear( `${ p }.mlpProj` ),
      } );
    }

    return {
      wte,
      wpe: this.readParamSync( 'wpe' ),
      layers,
      lnFW: this.readParamSync( 'lnFW' ),
      lnFB: this.paramList.includes( 'lnFB' ) ? this.readParamSync( 'lnFB' ) : new Float32Array( 0 ),
      lmHead: { w: wte, b: optional( 'lmHead.b' ) },
    };
  }

  // ------------------------------------------------------------ 内部

  private requireParam ( name: string ): TensorValue
  {
    // 对外（GPU 契约名）→ 对内（IR 表名）；已是 IR 名也能直查
    const irName = this.gpuToIr.get( name ) ?? name;
    const t = this.gpt.tensors.get( irName );
    if ( !t ) throw new Error( `CpuTrainer: 未知参数 ${ name }` );
    return t;
  }

  private readParamSync ( name: string ): Float32Array
  {
    const t = this.requireParam( name );
    if ( !( t.data instanceof Float32Array ) ) throw new Error( `CpuTrainer: 参数 ${ name } 不是 f32` );
    return t.data.slice();
  }
}
