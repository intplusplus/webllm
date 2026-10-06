import type { GpuContext } from '../gpu/device';
import { Arena } from '../gpu/arena';
import {
    alignTo,
    createStorageBuffer,
    createUniform,
    readbackBytes,
    readbackF32,
    StructPacker,
    tensorResource,
    writeF16,
    writeU32,
    type Tensor,
} from '../gpu/buffer';
import { CommandBatch, createBatchProfile, createComputePipeline, dispatch, type BatchProfile } from '../gpu/pipeline';
import { qwenHeadDim, qwenKvDim, type QwenConfig } from './qwen-config';

import embeddingF16Wgsl from '../gpu/kernels/embedding_f16.wgsl?raw';
import vecAddWgsl from '../gpu/kernels/vec_add.wgsl?raw';
import rmsnormWgsl from '../gpu/kernels/rmsnorm.wgsl?raw';
import gemmNtF16FromF32Wgsl from '../gpu/kernels/gemm_nt_f16_from_f32.wgsl?raw';
import gemmGemvF16Wgsl from '../gpu/kernels/gemm_gemv_f16.wgsl?raw';
import gemmGemvSplitF16Wgsl from '../gpu/kernels/gemm_gemv_split_f16.wgsl?raw';
import gemmGemvSplitReduceWgsl from '../gpu/kernels/gemm_gemv_split_reduce.wgsl?raw';
import ropeHalfWgsl from '../gpu/kernels/rope_half.wgsl?raw';
import attentionGqaWgsl from '../gpu/kernels/attention_gqa.wgsl?raw';
import attentionGqaCacheWgsl from '../gpu/kernels/attention_gqa_cache.wgsl?raw';
import kvStoreWgsl from '../gpu/kernels/kv_store.wgsl?raw';
import siluMulWgsl from '../gpu/kernels/silu_mul.wgsl?raw';

/** 必须与 attention_gqa.wgsl 中的 MAX_T 一致。 */
const MAX_T = 512;

/**
 * split-K 的调度目标：一次投影至少要排满约 24 个 workgroup（Vega 8~11 CU 的
 * 2~3 倍覆盖，留出延迟隐藏空间）才不浪费算力。低于该值就把 K 维切段并行。
 */
const GEMV_SPLIT_TARGET_WG = 24;
/** K 段数上限：段太短会放大归约与调度开销，8 已足够填满核显。 */
const GEMV_SPLIT_MAX = 8;

/**
 * 选 split-K 的段数 S ∈ {2,4,8}（返回 1 表示不切）。
 *
 * 原则：在满足 K % S == 0 且 (K/S) % 2 == 0（u32 边界）的前提下，
 * 取能让 workgroup 总数达到 GEMV_SPLIT_TARGET_WG 的最小 S；
 * 若所有档位都达不到（N 特别小，如 k/v_proj 只有 1 个 workgroup），
 * 取最大的可用 S——并行度仍是纯收益。
 *
 * 真实模型的调度结果：
 *   k/v_proj (N=128, 1 wg)  → S=8 → 8 wg
 *   q/o_proj (N=896, 4 wg)  → S=8 → 32 wg
 *   down_proj (N=896, 4 wg) → S=8 → 32 wg
 *   gate/up (N=4864, 19 wg) → S=2 → 38 wg
 *   lm_head (N=151936, 594 wg) → 不切（并行度已足够）
 */
function planGemvSplit ( wgN: number, K: number ): number
{
    if ( wgN >= GEMV_SPLIT_TARGET_WG ) return 1;
    let s = 1;
    for ( const cand of [ 2, 4, GEMV_SPLIT_MAX ] )
    {
        if ( K % cand === 0 && ( K / cand ) % 2 === 0 )
        {
            s = cand;
            if ( wgN * cand >= GEMV_SPLIT_TARGET_WG ) break;
        }
    }
    return s;
}

/**
 * Qwen2 的权重集合（CPU 侧 fp32，供参考实现与上传共用）。
 * 布局沿用 HF：Linear 权重为 [out, in] 行主序。
 */
export interface QwenLayerWeights
{
    /** [C] */
    inputNorm: Float32Array;
    /** [C, C] */
    qW: Float32Array;
    /** [C] */
    qB: Float32Array;
    /** [nKv*D, C] */
    kW: Float32Array;
    /** [nKv*D] */
    kB: Float32Array;
    /** [nKv*D, C] */
    vW: Float32Array;
    /** [nKv*D] */
    vB: Float32Array;
    /** [C, C]，无 bias */
    oW: Float32Array;
    /** [C] */
    postNorm: Float32Array;
    /** [inter, C] */
    gateW: Float32Array;
    /** [inter, C] */
    upW: Float32Array;
    /** [C, inter] */
    downW: Float32Array;
}

export interface QwenWeights
{
    /** [vocab, C]，同时充当 lm_head（tied） */
    embed: Float32Array;
    /** [C] */
    finalNorm: Float32Array;
    layers: QwenLayerWeights[];
}

/**
 * Qwen2 的 GPU 前向图（仅推理）。
 *
 * 权重常驻显存：矩阵权重以 f16 存放（Qwen2.5 原生 fp16，省一半显存），
 * norm/bias 用 f32（体积极小）。激活由 Arena 分配。
 * 与 TinyGpt 完全独立，不改动已验证的训练/推理路径。
 */
export class QwenGpt
{
    readonly config: QwenConfig;
    readonly maxBatch: number;

    /** 最近一次 forward 中每层 MLP 残差加完后的 hidden [M, C]，供对拍定位偏差层。 */
    readonly layerOutputs: Tensor[] = [];

    private readonly device: GPUDevice;
    private readonly arena: Arena;
    private readonly f16Weights = new Map<string, GPUBuffer>();
    private readonly f32Weights = new Map<string, GPUBuffer>();
    private readonly uniformCache = new Map<string, GPUBuffer>();
    private readonly dummyBias: GPUBuffer;
    private readonly inputTokens: GPUBuffer;
    /** 每层一份 KV cache，布局为扁平 [blockSize, nKvHead*D]。 */
    private readonly kCache: GPUBuffer[] = [];
    private readonly vCache: GPUBuffer[] = [];
    private cacheLen = 0;

    private readonly pEmbed: GPUComputePipeline;
    private readonly pAdd: GPUComputePipeline;
    private readonly pNorm: GPUComputePipeline;
    private readonly pGemmF16: GPUComputePipeline;
    private readonly pGemmGemv: GPUComputePipeline;
    private readonly pGemvSplit: GPUComputePipeline;
    private readonly pGemvReduce: GPUComputePipeline;
    private readonly pRope: GPUComputePipeline;
    private readonly pAttn: GPUComputePipeline;
    private readonly pAttnCache: GPUComputePipeline;
    private readonly pKvStore: GPUComputePipeline;
    private readonly pSiluMul: GPUComputePipeline;

    /**
     * 激活中的 batch（runForward 期间非空）：所有 kernel dispatch 进同一个
     * encoder，pass 边界表达读写依赖，最后只 submit 一次。
     */
    private batch: CommandBatch | null = null;
    /** 一次性剖析（timestamp-query）：若非空，下一次 runForward 逐 pass 计时。 */
    private pendingProfile: BatchProfile | null = null;

    constructor ( gpu: GpuContext, config: QwenConfig, maxBatch = 1 )
    {
        const device = gpu.device;
        this.device = device;
        this.config = config;
        this.maxBatch = maxBatch;

        const M = config.blockSize * maxBatch;
        const C = config.nEmbd;
        const kv = qwenKvDim( config );
        // 每行激活粗估：n1/q/qr/att/proj/xa/n2/d/xb 各 C，k/v/kr 各 kv，g/u/act 各 inter，
        // 再加 logits（vocab）。仅作容量提示，Arena 实际按需分配。
        const floatsPerRow = C * 9 + kv * 3 + config.intermediate * 3 + config.vocabSize;
        this.arena = new Arena( device, alignTo( M * floatsPerRow * 4 + ( 1 << 21 ), 256 ) );

        this.inputTokens = device.createBuffer( {
            label: 'qwen-tokens',
            size: alignTo( M * 4 ),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } );
        this.dummyBias = device.createBuffer( { label: 'qwen-dummy-bias', size: 16, usage: GPUBufferUsage.STORAGE } );

        this.pEmbed = createComputePipeline( device, embeddingF16Wgsl, 'main', 'embedding_f16' );
        this.pAdd = createComputePipeline( device, vecAddWgsl, 'main', 'vec_add' );
        this.pNorm = createComputePipeline( device, rmsnormWgsl, 'main', 'rmsnorm' );
        this.pGemmF16 = createComputePipeline( device, gemmNtF16FromF32Wgsl, 'main', 'gemm_nt_f16_from_f32' );
        this.pGemmGemv = createComputePipeline( device, gemmGemvF16Wgsl, 'main', 'gemm_gemv_f16' );
        this.pGemvSplit = createComputePipeline( device, gemmGemvSplitF16Wgsl, 'main', 'gemm_gemv_split_f16' );
        this.pGemvReduce = createComputePipeline( device, gemmGemvSplitReduceWgsl, 'main', 'gemm_gemv_split_reduce' );
        this.pRope = createComputePipeline( device, ropeHalfWgsl, 'main', 'rope_half' );
        this.pAttn = createComputePipeline( device, attentionGqaWgsl, 'main', 'attention_gqa' );
        this.pAttnCache = createComputePipeline( device, attentionGqaCacheWgsl, 'main', 'attention_gqa_cache' );
        this.pKvStore = createComputePipeline( device, kvStoreWgsl, 'main', 'kv_store' );
        this.pSiluMul = createComputePipeline( device, siluMulWgsl, 'main', 'silu_mul' );

        const cacheFloats = config.blockSize * kv;
        for ( let i = 0; i < config.nLayer; i++ )
        {
            this.kCache.push( createStorageBuffer( device, cacheFloats * 4, `L${ i }.kCache` ) );
            this.vCache.push( createStorageBuffer( device, cacheFloats * 4, `L${ i }.vCache` ) );
        }
    }

    // ------------------------------------------------------------ KV cache 状态

    get cacheLength (): number
    {
        return this.cacheLen;
    }

    /** 清空 KV cache（开始一段新的对话）。 */
    resetCache (): void
    {
        this.cacheLen = 0;
    }

    /** 已分配 KV cache 的字节数。 */
    get cacheBytes (): number
    {
        let total = 0;
        for ( const b of this.kCache ) total += b.size;
        for ( const b of this.vCache ) total += b.size;
        return total;
    }

    get arenaUsedBytes (): number
    {
        return this.arena.usedBytes;
    }

    /** 已上传的权重显存占用（字节）。 */
    get weightBytes (): number
    {
        let total = 0;
        for ( const b of this.f16Weights.values() ) total += b.size;
        for ( const b of this.f32Weights.values() ) total += b.size;
        return total;
    }

    // ------------------------------------------------------------ 权重上传

    /** 分配一个 f16 权重 buffer（length 为元素个数）。 */
    allocF16 ( name: string, length: number ): void
    {
        if ( this.f16Weights.has( name ) ) throw new Error( `权重已存在: ${ name }` );
        this.f16Weights.set( name, createStorageBuffer( this.device, length * 2, name ) );
    }

    /** 向已分配的 f16 权重 buffer 写入一段（用于大张量分块上传）。 */
    writeF16Chunk ( name: string, offsetElements: number, data: Float32Array ): void
    {
        const buffer = this.f16Weights.get( name );
        if ( !buffer ) throw new Error( `缺少 f16 权重: ${ name }` );
        writeF16( this.device, buffer, offsetElements, data );
    }

    /** 一次性上传一个 f16 权重张量。 */
    uploadF16 ( name: string, data: Float32Array ): void
    {
        this.allocF16( name, data.length );
        this.writeF16Chunk( name, 0, data );
    }

    /** 上传一个 f32 权重张量（norm / bias）。 */
    uploadF32 ( name: string, data: Float32Array ): void
    {
        if ( this.f32Weights.has( name ) ) throw new Error( `权重已存在: ${ name }` );
        const buffer = createStorageBuffer( this.device, data.length * 4, name );
        this.device.queue.writeBuffer( buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength );
        this.f32Weights.set( name, buffer );
    }

    /** 上传一层 Qwen 权重（供流式加载复用）。 */
    uploadLayer ( index: number, L: QwenLayerWeights ): void
    {
        const p = `L${ index }`;
        this.uploadF32( `${ p }.inNorm`, L.inputNorm );
        this.uploadF16( `${ p }.qW`, L.qW );
        this.uploadF32( `${ p }.qB`, L.qB );
        this.uploadF16( `${ p }.kW`, L.kW );
        this.uploadF32( `${ p }.kB`, L.kB );
        this.uploadF16( `${ p }.vW`, L.vW );
        this.uploadF32( `${ p }.vB`, L.vB );
        this.uploadF16( `${ p }.oW`, L.oW );
        this.uploadF32( `${ p }.postNorm`, L.postNorm );
        this.uploadF16( `${ p }.gateW`, L.gateW );
        this.uploadF16( `${ p }.upW`, L.upW );
        this.uploadF16( `${ p }.downW`, L.downW );
    }

    /** 上传整个权重集合（小模型 / 测试用）。 */
    uploadWeights ( weights: QwenWeights ): void
    {
        this.uploadF16( 'embed', weights.embed );
        this.uploadF32( 'finalNorm', weights.finalNorm );
        weights.layers.forEach( ( L, i ) => this.uploadLayer( i, L ) );
    }

    // ------------------------------------------------------------ 前向

    /**
     * 无 KV cache 的完整前向（B 个序列、每个 T 长），返回 logits 张量 [B*T, vocab]。
     * 用于对拍与批量评分；自回归生成请走 prefill / decodeStep。
     */
    forward ( tokens: Uint32Array, B: number, T: number ): Tensor
    {
        return this.runForward( tokens, B * T, B, T, 0, false );
    }

    /** 预填充：一次性处理整段 prompt 并写入 KV cache，返回最后一行对应的 logits [T, vocab]。 */
    prefill ( tokens: Uint32Array ): Tensor
    {
        if ( this.cacheLen !== 0 ) throw new Error( `prefill 前请先 resetCache()（当前 cacheLen=${ this.cacheLen }）` );
        if ( tokens.length === 0 ) throw new Error( 'prefill: tokens 不能为空' );
        return this.runForward( tokens, tokens.length, 1, tokens.length, 0, true );
    }

    /** 增量解码一步：只处理 1 个 token，复用 KV cache，返回 logits [1, vocab]。 */
    decodeStep ( token: number ): Tensor
    {
        if ( this.cacheLen === 0 ) throw new Error( 'decodeStep 前请先调用 prefill()' );
        return this.runForward( Uint32Array.of( token ), 1, 1, 1, this.cacheLen, true );
    }

    /**
     * 统一前向：useCache=false 走无 cache 路径（k/v 仅在本次上下文内可见）；
     * useCache=true 走 KV cache 路径（k/v 写入 cache，注意力读取整段 cache）。
     */
    private runForward (
        tokens: Uint32Array,
        M: number,
        B: number,
        T: number,
        pastLen: number,
        useCache: boolean,
    ): Tensor
    {
        const cfg = this.config;
        const C = cfg.nEmbd;
        const H = cfg.nHead;
        const HI = qwenKvDim( cfg );
        const D = qwenHeadDim( cfg );
        const I = cfg.intermediate;
        const V = cfg.vocabSize;
        const eps = cfg.rmsEps;

        if ( T > MAX_T ) throw new Error( `T=${ T } 超过 attention 上限 MAX_T=${ MAX_T }` );
        if ( useCache && pastLen + T > cfg.blockSize )
        {
            throw new Error( `上下文长度 ${ pastLen + T } 超过 blockSize=${ cfg.blockSize }` );
        }
        if ( !useCache && T > cfg.blockSize ) throw new Error( `T=${ T } 超过 blockSize=${ cfg.blockSize }` );
        if ( tokens.length !== M ) throw new Error( `tokens 长度 ${ tokens.length } != M=${ M }` );

        this.arena.reset();
        this.layerOutputs.length = 0;
        writeU32( this.device, this.inputTokens, 0, tokens );

        // 位置：row % T + pastLen。无 cache 时 pastLen=0 且 T 为每条序列的长度；
        // 有 cache 时 T=M、pastLen 为已有上下文长度，于是绝对位置连续。
        const uEmbed = this.uniform( `embed:${ M }:${ C }`, ( p ) => p.u32( M ).u32( C ).u32( 0 ).u32( 0 ) );
        const uNorm = this.uniform( `rms:${ M }:${ C }:${ eps }`, ( p ) => p.u32( M ).u32( C ).f32( eps ).u32( 0 ) );
        const uRopeQ = this.uniform( `ropeq:${ M }:${ H }:${ D }:${ T }:${ pastLen }:${ cfg.ropeTheta }`, ( p ) =>
            p.u32( M ).u32( H ).u32( D ).u32( T ).f32( cfg.ropeTheta ).u32( pastLen ).u32( 0 ).u32( 0 ) );
        const uRopeK = this.uniform( `ropek:${ M }:${ cfg.nKvHead }:${ D }:${ T }:${ pastLen }:${ cfg.ropeTheta }`, ( p ) =>
            p.u32( M ).u32( cfg.nKvHead ).u32( D ).u32( T ).f32( cfg.ropeTheta ).u32( pastLen ).u32( 0 ).u32( 0 ) );
        const uAttn = this.uniform( `gqa:${ B }:${ T }:${ H }:${ cfg.nKvHead }:${ D }`, ( p ) =>
            p.u32( B ).u32( T ).u32( H ).u32( cfg.nKvHead ).u32( D ).f32( 1 / Math.sqrt( D ) ).f32( 0 ).f32( 0 ) );
        const uAttnCache = this.uniform( `gqac:${ M }:${ H }:${ cfg.nKvHead }:${ D }:${ pastLen }`, ( p ) =>
            p.u32( M ).u32( H ).u32( cfg.nKvHead ).u32( D ).f32( 1 / Math.sqrt( D ) ).u32( pastLen ).u32( 0 ).u32( 0 ) );
        const uKvStore = this.uniform( `kv:${ M }:${ HI }:${ pastLen }`, ( p ) =>
            p.u32( M ).u32( HI ).u32( pastLen ).u32( 0 ) );
        const uSilu = this.uniform( `silu:${ M * I }`, ( p ) => p.u32( M * I ).u32( 0 ).u32( 0 ).u32( 0 ) );

        const nMC = Math.ceil( ( M * C ) / 64 );
        const nKvStore = Math.ceil( ( M * HI ) / 64 );

        // 全部 kernel 装进同一个 command encoder、只 submit 一次（省掉每 dispatch
        // ~110µs 的 CPU 提交开销）；pass 边界即读写依赖屏障，划分原则：
        // 同一 pass 内的 dispatch 之间不得存在对同一 buffer 的 RAW / WAR / WAW。
        // 每层 12 个 pass / 17 次 dispatch（cache 路径），24 层 + 头尾 ≈ 291 pass。
        this.batch = new CommandBatch( this.device );
        if ( this.pendingProfile ) this.batch.enableProfile( this.pendingProfile );
        try
        {
            // --- 嵌入（无位置嵌入，位置信息由 RoPE 注入） ---
            let x = this.arena.allocFloat32( M * C, [ M, C ] );
            this.run( this.pEmbed, [ this.wf16( 'embed' ), this.inputTokens, tensorResource( x ), uEmbed ], Math.ceil( ( M * C ) / 64 ) );
            this.passBreak(); // x 写 → 各层 norm 读

            const nRopeQ = Math.ceil( ( M * H * ( D / 2 ) ) / 64 );
            const nRopeK = Math.ceil( ( M * cfg.nKvHead * ( D / 2 ) ) / 64 );

            for ( let i = 0; i < cfg.nLayer; i++ )
            {
                const p = `L${ i }`;

                // --- attention 子层 ---
                const n1 = this.arena.allocFloat32( M * C, [ M, C ] );
                this.run( this.pNorm, [ tensorResource( x ), this.f32( `${ p }.inNorm` ), tensorResource( n1 ), uNorm ], M );
                this.passBreak(); // n1 写 → q/k/v 读

                // q/k/v 都只读 n1、写各自的输出，可同 pass 并行
                const q = this.arena.allocFloat32( M * C, [ M, C ] );
                const k = this.arena.allocFloat32( M * HI, [ M, HI ] );
                const v = this.arena.allocFloat32( M * HI, [ M, HI ] );
                this.gemm( n1, `${ p }.qW`, q, `${ p }.qB`, M, C, C );
                this.gemm( n1, `${ p }.kW`, k, `${ p }.kB`, M, HI, C );
                this.gemm( n1, `${ p }.vW`, v, `${ p }.vB`, M, HI, C );
                this.passBreak(); // q/k 写 → rope 读

                // q→qr 与 k→kr 无相互依赖，同 pass
                const qr = this.arena.allocFloat32( M * C, [ M, C ] );
                const kr = this.arena.allocFloat32( M * HI, [ M, HI ] );
                this.run( this.pRope, [ tensorResource( q ), tensorResource( qr ), uRopeQ ], nRopeQ );
                this.run( this.pRope, [ tensorResource( k ), tensorResource( kr ), uRopeK ], nRopeK );
                this.passBreak(); // kr 写 → cache 读

                const att = this.arena.allocFloat32( M * C, [ M, C ] );
                if ( useCache )
                {
                    // RoPE 后写入 cache，再让注意力读整段 cache（cache 写 → attn 读必须隔 pass）
                    this.run( this.pKvStore, [ tensorResource( kr ), { buffer: this.kCache[ i ], size: this.kCache[ i ].size }, uKvStore ], nKvStore );
                    this.run( this.pKvStore, [ tensorResource( v ), { buffer: this.vCache[ i ], size: this.vCache[ i ].size }, uKvStore ], nKvStore );
                    this.passBreak();
                    this.run(
                        this.pAttnCache,
                        [
                            tensorResource( qr ),
                            { buffer: this.kCache[ i ], size: this.kCache[ i ].size },
                            { buffer: this.vCache[ i ], size: this.vCache[ i ].size },
                            tensorResource( att ),
                            uAttnCache,
                        ],
                        [ M, H ],
                    );
                    this.passBreak(); // att 写 → o 读
                }
                else
                {
                    this.run( this.pAttn, [ tensorResource( qr ), tensorResource( kr ), tensorResource( v ), tensorResource( att ), uAttn ], [ T, H, B ] );
                    this.passBreak();
                }

                const proj = this.arena.allocFloat32( M * C, [ M, C ] );
                this.gemm( att, `${ p }.oW`, proj, null, M, C, C );
                this.passBreak(); // proj 写 → add 读

                const xa = this.arena.allocFloat32( M * C, [ M, C ] );
                this.run( this.pAdd, [ tensorResource( x ), tensorResource( proj ), tensorResource( xa ) ], nMC );
                this.passBreak(); // xa 写 → norm2 读

                // --- MLP 子层（SwiGLU） ---
                const n2 = this.arena.allocFloat32( M * C, [ M, C ] );
                this.run( this.pNorm, [ tensorResource( xa ), this.f32( `${ p }.postNorm` ), tensorResource( n2 ), uNorm ], M );
                this.passBreak(); // n2 写 → gate/up 读

                const gate = this.arena.allocFloat32( M * I, [ M, I ] );
                const up = this.arena.allocFloat32( M * I, [ M, I ] );
                this.gemm( n2, `${ p }.gateW`, gate, null, M, I, C );
                this.gemm( n2, `${ p }.upW`, up, null, M, I, C );
                this.passBreak(); // gate/up 写 → silu 读

                const act = this.arena.allocFloat32( M * I, [ M, I ] );
                this.run( this.pSiluMul, [ tensorResource( gate ), tensorResource( up ), tensorResource( act ), uSilu ], Math.ceil( ( M * I ) / 64 ) );
                this.passBreak(); // act 写 → down 读

                const d = this.arena.allocFloat32( M * C, [ M, C ] );
                this.gemm( act, `${ p }.downW`, d, null, M, C, I );
                this.passBreak(); // d 写 → add2 读

                const xb = this.arena.allocFloat32( M * C, [ M, C ] );
                this.run( this.pAdd, [ tensorResource( xa ), tensorResource( d ), tensorResource( xb ) ], nMC );
                this.passBreak(); // xb 写 → 下一层 norm 读

                x = xb;
                this.layerOutputs.push( xb );
            }

            const lnf = this.arena.allocFloat32( M * C, [ M, C ] );
            this.run( this.pNorm, [ tensorResource( x ), this.f32( 'finalNorm' ), tensorResource( lnf ), uNorm ], M );
            this.passBreak(); // lnf 写 → lm_head 读

            const logits = this.arena.allocFloat32( M * V, [ M, V ] );
            this.gemm( lnf, 'embed', logits, null, M, V, C );
            if ( useCache ) this.cacheLen = pastLen + M;
            return logits;
        }
        finally
        {
            const batch = this.batch;
            this.batch = null;
            batch.submit();
            this.pendingProfile = null;
        }
    }

    /**
     * 给下一次 forward/prefill/decodeStep 开启逐 pass 剖析（timestamp-query）。
     * 返回的句柄在该次前向 submit 后可用 readBatchProfile 读回每 pass 耗时；
     * 用完调用方负责 destroy() 资源。需要 timestamp-query feature。
     */
    profileNextForward ( maxPasses = 512 ): BatchProfile
    {
        if ( this.pendingProfile ) throw new Error( '上一次剖析尚未消费' );
        this.pendingProfile = createBatchProfile( this.device, maxPasses );
        return this.pendingProfile;
    }

    async readTensor ( t: Tensor ): Promise<Float32Array>
    {
        return readbackF32( this.device, t.buffer, t.offset / 4, t.length );
    }

    /** 只读回张量的一段（生成时只需最后一行 logits，省掉整块回传）。 */
    async readTensorSlice ( t: Tensor, elementOffset: number, elementCount: number ): Promise<Float32Array>
    {
        return readbackF32( this.device, t.buffer, t.offset / 4 + elementOffset, elementCount );
    }

    /** 读回某个 f16 权重的原始字节（用于校验加载/上传是否正确）。 */
    async readWeightF16Bytes ( name: string, elementOffset: number, elementCount: number ): Promise<ArrayBuffer>
    {
        const buffer = this.f16Weights.get( name );
        if ( !buffer ) throw new Error( `缺少 f16 权重: ${ name }` );
        return readbackBytes( this.device, buffer, elementOffset * 2, elementCount * 2 );
    }

    // ------------------------------------------------------------ 内部工具

    private run (
        pipeline: GPUComputePipeline,
        resources: ( GPUBuffer | GPUBufferBinding )[],
        workgroups: number | readonly number[],
    ): void
    {
        const entries: GPUBindGroupEntry[] = resources.map( ( resource, binding ) => ( { binding, resource } ) );
        const bindGroup = this.device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries } );
        if ( this.batch )
        {
            this.batch.dispatch( pipeline, bindGroup, workgroups );
            return;
        }
        dispatch( this.device, pipeline, bindGroup, workgroups );
    }

    /**
     * pass 边界（隐式内存屏障）：之前 pass 的写入对之后的 dispatch 可见。
     * 同一 pass 内的 dispatch 之间对同一 buffer 的读写冲突行为未定义，
     * 因此每个「写 → 读」依赖处都必须调用。
     */
    private passBreak (): void
    {
        this.batch?.endPass();
    }

    /**
     * C = A @ W^T (+ bias)，A/C 为 f32、W 为 f16。
     * M == 1（解码）时走 GEMV kernel，避免通用 GEMM 沿 M 分块造成的填充浪费；
     * 该 kernel 把权重按 u32（2 个 f16）读取，要求 K 为偶数，否则退回通用 GEMM。
     * 解码投影的 workgroup 数不足时（如 k/v_proj 只有 1 个）再按 K 维 split-K 并行，
     * 由 gemm_gemv_split_reduce 归约部分和。
     */
    private gemm ( A: Tensor, weight: string, C: Tensor, bias: string | null, M: number, N: number, K: number ): void
    {
        const hasBias = bias !== null;
        const u = this.uniform( `gemm:${ M }:${ N }:${ K }:${ hasBias ? 1 : 0 }`, ( p ) =>
            p.u32( M ).u32( N ).u32( K ).u32( hasBias ? 1 : 0 ) );
        const biasBuffer = bias ? this.f32( bias ) : this.dummyBias;

        if ( M === 1 && K % 2 === 0 )
        {
            const wgN = Math.ceil( N / 256 );
            const split = planGemvSplit( wgN, K );
            if ( split > 1 )
            {
                // split-K：S 段并行算部分和 → reduce 归约（bias 只在归约端加一次）。
                // partial 由 Arena 分配，两次 dispatch 之间按提交顺序复用，天然安全。
                const partial = this.arena.allocFloat32( split * N, [ split, N ] );
                const us = this.uniform( `gemvsplit:${ N }:${ K }:${ split }:${ hasBias ? 1 : 0 }`, ( p ) =>
                    p.u32( N ).u32( K ).u32( split ).u32( hasBias ? 1 : 0 ) );
                this.run(
                    this.pGemvSplit,
                    [ tensorResource( A ), this.wf16( weight ), tensorResource( partial ), us ],
                    [ wgN, split ],
                );
                this.run(
                    this.pGemvReduce,
                    [ tensorResource( partial ), tensorResource( C ), biasBuffer, us ],
                    wgN,
                );
                return;
            }

            this.run(
                this.pGemmGemv,
                [ tensorResource( A ), this.wf16( weight ), tensorResource( C ), biasBuffer, u ],
                wgN,
            );
            return;
        }

        this.run(
            this.pGemmF16,
            [ tensorResource( A ), this.wf16( weight ), tensorResource( C ), biasBuffer, u ],
            [ Math.ceil( M / 64 ), Math.ceil( N / 64 ), 1 ],
        );
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

    private wf16 ( name: string ): GPUBuffer
    {
        const buffer = this.f16Weights.get( name );
        if ( !buffer ) throw new Error( `缺少 f16 权重: ${ name }` );
        return buffer;
    }

    private f32 ( name: string ): GPUBuffer
    {
        const buffer = this.f32Weights.get( name );
        if ( !buffer ) throw new Error( `缺少 f32 权重: ${ name }` );
        return buffer;
    }
}
