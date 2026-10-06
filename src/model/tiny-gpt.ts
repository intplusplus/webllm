import type { GpuContext } from '../gpu/device';
import { Arena } from '../gpu/arena';
import {
    alignTo,
    createStorageBuffer,
    createUniform,
    readbackF32,
    StructPacker,
    tensorResource,
    writeF16,
    writeU32,
    type Tensor,
} from '../gpu/buffer';
import { createComputePipeline, dispatch } from '../gpu/pipeline';
import { headDim, type GPTConfig } from './config';
import type { GPTWeights, LinearWeights } from './init';

import embeddingWgsl from '../gpu/kernels/embedding.wgsl?raw';
import vecAddWgsl from '../gpu/kernels/vec_add.wgsl?raw';
import layernormWgsl from '../gpu/kernels/layernorm.wgsl?raw';
import addLayernormWgsl from '../gpu/kernels/add_layernorm.wgsl?raw';
import gemmNtWgsl from '../gpu/kernels/gemm_nt.wgsl?raw';
import gemmNtF16FromF32Wgsl from '../gpu/kernels/gemm_nt_f16_from_f32.wgsl?raw';
import ropeWgsl from '../gpu/kernels/rope.wgsl?raw';
import attentionWgsl from '../gpu/kernels/attention.wgsl?raw';
import geluWgsl from '../gpu/kernels/gelu.wgsl?raw';

const EPS = 1e-5;
const ROPE_BASE = 10000;
/** 必须与 attention.wgsl 中的 MAX_T 一致。 */
const MAX_T = 512;

/**
 * tiny-GPT 的 GPU 前向图：把各 WGSL 算子按 GPT-2 结构串起来。
 * 权重常驻 buffer；激活由 Arena 分配为独立 buffer（按尺寸池化复用）。
 */
export class TinyGpt
{
    readonly config: GPTConfig;
    readonly maxBatch: number;
    readonly useF16Gemm: boolean;

    private readonly device: GPUDevice;
    private readonly arena: Arena;
    private readonly weightBuffers = new Map<string, GPUBuffer>();
    private readonly f16Weights = new Map<string, GPUBuffer>();
    private readonly uniformCache = new Map<string, GPUBuffer>();
    private readonly dummyBias: GPUBuffer;
    private readonly inputTokens: GPUBuffer;
    private readonly inputPositions: GPUBuffer;

    private readonly pEmbed: GPUComputePipeline;
    private readonly pAdd: GPUComputePipeline;
    private readonly pNorm: GPUComputePipeline;
    private readonly pAddNorm: GPUComputePipeline;
    private readonly pGemm: GPUComputePipeline;
    private readonly pGemmF16FromF32: GPUComputePipeline | null;
    private readonly pRope: GPUComputePipeline;
    private readonly pAttn: GPUComputePipeline;
    private readonly pGelu: GPUComputePipeline;

    constructor ( gpu: GpuContext, config: GPTConfig, weights: GPTWeights, maxBatch = 4, useF16Gemm = false )
    {
        const device = gpu.device;
        this.device = device;
        this.config = config;
        this.maxBatch = maxBatch;
        this.useF16Gemm = useF16Gemm;

        const M = config.blockSize * maxBatch;
        const C = config.nEmbd;
        // 每行激活粗估：tok/pos/x/ln1/q/k/v/qr/kr/att/proj/ln2/(fc+act=8C)/mp/lnf + logits
        const floatsPerRow = C * 26 + config.vocabSize;
        const needed = alignTo( M * floatsPerRow * 4 * 2 + ( 1 << 21 ), 256 );
        this.arena = new Arena( device, needed );

        this.inputTokens = device.createBuffer( {
            label: 'input-tokens',
            size: alignTo( M * 4 ),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } );
        this.inputPositions = device.createBuffer( {
            label: 'input-positions',
            size: alignTo( M * 4 ),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } );
        this.dummyBias = device.createBuffer( { label: 'dummy-bias', size: 4, usage: GPUBufferUsage.STORAGE } );

        this.pEmbed = createComputePipeline( device, embeddingWgsl, 'main', 'embedding' );
        this.pAdd = createComputePipeline( device, vecAddWgsl, 'main', 'vec_add' );
        this.pNorm = createComputePipeline( device, layernormWgsl, 'main', 'layernorm' );
        this.pAddNorm = createComputePipeline( device, addLayernormWgsl, 'main', 'add_layernorm' );
        this.pGemm = createComputePipeline( device, gemmNtWgsl, 'main', 'gemm_nt' );
        this.pGemmF16FromF32 = this.useF16Gemm ? createComputePipeline( device, gemmNtF16FromF32Wgsl, 'main', 'gemm_nt_f16_from_f32' ) : null;
        this.pRope = createComputePipeline( device, ropeWgsl, 'main', 'rope' );
        this.pAttn = createComputePipeline( device, attentionWgsl, 'main', 'attention' );
        this.pGelu = createComputePipeline( device, geluWgsl, 'main', 'gelu' );

        this.uploadWeights( weights );
    }

    get arenaUsedBytes (): number
    {
        return this.arena.usedBytes;
    }

    /** 执行一次完整前向（异步提交），返回 logits 张量。 */
    forward ( tokens: Uint32Array, B: number, T: number ): Tensor
    {
        const cfg = this.config;
        const C = cfg.nEmbd;
        const H = cfg.nHead;
        const D = headDim( cfg );
        const V = cfg.vocabSize;
        const M = B * T;

        if ( T > MAX_T ) throw new Error( `T=${ T } 超过 attention 上限 MAX_T=${ MAX_T }` );
        if ( B * T > cfg.blockSize * this.maxBatch )
        {
            throw new Error( `B*T=${ B * T } 超过 maxBatch*blockSize=${ cfg.blockSize * this.maxBatch }` );
        }
        if ( tokens.length !== M ) throw new Error( `tokens 长度 ${ tokens.length } != B*T=${ M }` );

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

        // --- 嵌入 + 位置嵌入 ---
        const tokEmb = this.arena.allocFloat32( M * C, [ M, C ] );
        const posEmb = this.arena.allocFloat32( M * C, [ M, C ] );
        this.run( this.pEmbed, [ this.w( 'wte' ), this.inputTokens, tensorResource( tokEmb ), uEmbed ], nMC );
        this.run( this.pEmbed, [ this.w( 'wpe' ), this.inputPositions, tensorResource( posEmb ), uEmbed ], nMC );

        let x = this.arena.allocFloat32( M * C, [ M, C ] );
        this.run( this.pAdd, [ tensorResource( tokEmb ), tensorResource( posEmb ), tensorResource( x ) ], nMC );

        for ( let i = 0; i < cfg.nLayer; i++ )
        {
            const p = `L${ i }`;

            // attention 子层
            const ln1 = this.arena.allocFloat32( M * C, [ M, C ] );
            this.run( this.pNorm, [ tensorResource( x ), this.w( `${ p }.ln1W` ), this.biasBuf( `${ p }.ln1B` ), tensorResource( ln1 ), uNorm ], M );

            const q = this.arena.allocFloat32( M * C, [ M, C ] );
            const k = this.arena.allocFloat32( M * C, [ M, C ] );
            const v = this.arena.allocFloat32( M * C, [ M, C ] );
            this.gemm( ln1, `${ p }.wq.w`, q, `${ p }.wq.b`, M, C, C, cfg.bias );
            this.gemm( ln1, `${ p }.wk.w`, k, `${ p }.wk.b`, M, C, C, cfg.bias );
            this.gemm( ln1, `${ p }.wv.w`, v, `${ p }.wv.b`, M, C, C, cfg.bias );

            const qr = this.arena.allocFloat32( M * C, [ M, C ] );
            const kr = this.arena.allocFloat32( M * C, [ M, C ] );
            const nRope = Math.ceil( ( M * H * ( D / 2 ) ) / 64 );
            this.run( this.pRope, [ tensorResource( q ), tensorResource( qr ), uRope ], nRope );
            this.run( this.pRope, [ tensorResource( k ), tensorResource( kr ), uRope ], nRope );

            const att = this.arena.allocFloat32( M * C, [ M, C ] );
            this.run( this.pAttn, [ tensorResource( qr ), tensorResource( kr ), tensorResource( v ), tensorResource( att ), uAttn ], [ T, H, B ] );

            const proj = this.arena.allocFloat32( M * C, [ M, C ] );
            this.gemm( att, `${ p }.attnProj.w`, proj, `${ p }.attnProj.b`, M, C, C, cfg.bias );

            // mlp 子层：xa = x + proj，ln2 = LN(xa)，两者融合为一个 kernel。
            const xa = this.arena.allocFloat32( M * C, [ M, C ] );
            const ln2 = this.arena.allocFloat32( M * C, [ M, C ] );
            this.run(
                this.pAddNorm,
                [
                    tensorResource( x ),
                    tensorResource( proj ),
                    this.w( `${ p }.ln2W` ),
                    this.biasBuf( `${ p }.ln2B` ),
                    tensorResource( ln2 ),
                    tensorResource( xa ),
                    uNorm,
                ],
                M,
            );

            const fc = this.arena.allocFloat32( M * 4 * C, [ M, 4 * C ] );
            this.gemm( ln2, `${ p }.fc.w`, fc, `${ p }.fc.b`, M, 4 * C, C, cfg.bias );

            const act = this.arena.allocFloat32( M * 4 * C, [ M, 4 * C ] );
            this.run( this.pGelu, [ tensorResource( fc ), tensorResource( act ), uGelu ], Math.ceil( ( M * 4 * C ) / 64 ) );

            const mp = this.arena.allocFloat32( M * C, [ M, C ] );
            this.gemm( act, `${ p }.mlpProj.w`, mp, `${ p }.mlpProj.b`, M, C, 4 * C, cfg.bias );

            const xb = this.arena.allocFloat32( M * C, [ M, C ] );
            this.run( this.pAdd, [ tensorResource( xa ), tensorResource( mp ), tensorResource( xb ) ], nMC );

            x = xb;
        }

        const lnf = this.arena.allocFloat32( M * C, [ M, C ] );
        this.run( this.pNorm, [ tensorResource( x ), this.w( 'lnFW' ), this.w( 'lnFB' ), tensorResource( lnf ), uNorm ], M );

        const logits = this.arena.allocFloat32( M * V, [ M, V ] );
        this.gemm( lnf, 'wte', logits, 'lmHead.b', M, V, C, cfg.bias );
        return logits;
    }

    async readTensor ( t: Tensor ): Promise<Float32Array>
    {
        return readbackF32( this.device, t.buffer, t.offset / 4, t.length );
    }

    private run (
        pipeline: GPUComputePipeline,
        resources: ( GPUBuffer | GPUBufferBinding )[],
        workgroups: number | readonly number[],
    ): void
    {
        const entries: GPUBindGroupEntry[] = resources.map( ( resource, binding ) => ( { binding, resource } ) );
        const bindGroup = this.device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries } );
        dispatch( this.device, pipeline, bindGroup, workgroups );
    }

    /** C = A @ weight^T (+ bias)。fp32 与 f16 GEMM 两条路径都由这里分发。 */
    private gemm (
        A: Tensor,
        weight: string,
        C: Tensor,
        bias: string | null,
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
        const biasBuffer = bias ? this.biasBuf( bias ) : this.dummyBias;
        const workgroups: [ number, number, number ] = [ Math.ceil( M / 64 ), Math.ceil( N / 64 ), 1 ];

        if ( !this.useF16Gemm )
        {
            this.run( this.pGemm, [ tensorResource( A ), this.w( weight ), tensorResource( C ), biasBuffer, u ], workgroups );
            return;
        }
        if ( !this.pGemmF16FromF32 ) throw new Error( 'f16 GEMM 未初始化' );
        // A 仍为 fp32 activation；f16 GEMM 在装载到 shared 时转换，避免额外 cast kernel/dispatch。
        this.run( this.pGemmF16FromF32, [ tensorResource( A ), this.wf16( weight ), tensorResource( C ), biasBuffer, u ], workgroups );
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

    private w ( name: string ): GPUBuffer
    {
        const buffer = this.weightBuffers.get( name );
        if ( !buffer ) throw new Error( `缺少权重 buffer: ${ name }` );
        return buffer;
    }

    private biasBuf ( name: string ): GPUBuffer
    {
        return this.weightBuffers.get( name ) ?? this.dummyBias;
    }

    private wf16 ( name: string ): GPUBuffer
    {
        const buffer = this.f16Weights.get( name );
        if ( !buffer ) throw new Error( `缺少 f16 权重 buffer: ${ name }` );
        return buffer;
    }

    private upload ( name: string, data: Float32Array ): void
    {
        const buffer = this.device.createBuffer( {
            label: name,
            size: Math.max( alignTo( data.length * 4 ), 4 ),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } );
        this.device.queue.writeBuffer(
            buffer,
            0,
            data.buffer as ArrayBuffer,
            data.byteOffset,
            data.byteLength,
        );
        this.weightBuffers.set( name, buffer );
    }

    private uploadF16 ( name: string, data: Float32Array ): void
    {
        const buffer = createStorageBuffer( this.device, data.length * 2, `${ name }.f16` );
        writeF16( this.device, buffer, 0, data );
        this.f16Weights.set( name, buffer );
    }

    private uploadLinear ( prefix: string, lw: LinearWeights ): void
    {
        this.upload( `${ prefix }.w`, lw.w );
        if ( lw.b ) this.upload( `${ prefix }.b`, lw.b );
        if ( this.useF16Gemm ) this.uploadF16( `${ prefix }.w`, lw.w );
    }

    private uploadWeights ( weights: GPTWeights ): void
    {
        this.upload( 'wte', weights.wte );
        this.upload( 'wpe', weights.wpe );
        this.upload( 'lnFW', weights.lnFW );
        this.upload( 'lnFB', weights.lnFB );
        if ( weights.lmHead.b ) this.upload( 'lmHead.b', weights.lmHead.b );
        if ( this.useF16Gemm ) this.uploadF16( 'wte', weights.wte );

        weights.layers.forEach( ( L, i ) =>
        {
            const p = `L${ i }`;
            this.upload( `${ p }.ln1W`, L.ln1W );
            this.upload( `${ p }.ln1B`, L.ln1B );
            this.upload( `${ p }.ln2W`, L.ln2W );
            this.upload( `${ p }.ln2B`, L.ln2B );
            this.uploadLinear( `${ p }.wq`, L.wq );
            this.uploadLinear( `${ p }.wk`, L.wk );
            this.uploadLinear( `${ p }.wv`, L.wv );
            this.uploadLinear( `${ p }.attnProj`, L.attnProj );
            this.uploadLinear( `${ p }.fc`, L.fc );
            this.uploadLinear( `${ p }.mlpProj`, L.mlpProj );
        } );
    }
}