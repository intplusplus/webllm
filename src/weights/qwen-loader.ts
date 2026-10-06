/**
 * 从本地 safetensors 加载 Qwen2.5-0.5B-Instruct-GPTQ-Int4 权重到 GPU。
 *
 * 真实格式（已在 M4 前置探测中逐字节确认）：
 *   - 每层 7 个投影为 GPTQ int4：qweight I32 / qzeros I32（存 zero-1）/ scales F16 / g_idx I32（恒等）
 *   - q/k/v 有 F16 bias；o_proj 与 MLP 无 bias
 *   - input/post_attention layernorm 权重为 F16
 *   - embed_tokens F16 [vocab, C]；无独立 lm_head（tie_word_embeddings）
 *
 * 显存策略：矩阵权重反量化后统一按 f16 上传（Qwen2.5 原生 fp16，精度无损且体积减半）；
 * norm/bias 用 f32（体积可忽略）。embed 分块上传，避免一次性分配 544MB 的 f32 副本。
 */
import type { GpuContext } from '../gpu/device';
import {
    f16BitsToF32,
    parseSafetensors,
    readTensorBytes,
    readTensorF32,
    requireTensor,
    type SafetensorFile,
} from './safetensors';
import { dequantizeGptqInt4 } from './quant';
import type { QwenConfig } from '../model/qwen-config';
import { qwenKvDim } from '../model/qwen-config';
import type { QwenGpt, QwenLayerWeights } from '../model/qwen';

export const QWEN_LOCAL_DIR = '/models/qwen2.5-0.5b-int4/';

export interface QwenLoadOptions
{
    url?: string;
    /** 每层加载完成时回调，便于展示进度。 */
    onProgress?: ( done: number, total: number, label: string ) => void;
    /** 保留该层的 CPU fp32 权重（用于校验反量化/上传），默认不保留。 */
    keepLayer?: number;
    /** 保留 embed 的 [start, start+rows) 行 fp32（用于校验），默认不保留。 */
    keepEmbedRows?: { start: number; rows: number };
}

export interface QwenLoadResult
{
    /** 上传的 GPU 权重字节数。 */
    gpuBytes: number;
    /** 处理的 tensor 数量。 */
    tensorCount: number;
    /** safetensors 文件字节数。 */
    fileBytes: number;
    /** keepLayer 指定时，保留的该层 fp32 权重。 */
    layer?: QwenLayerWeights;
    /** keepEmbedRows 指定时，保留的若干行 fp32 权重。 */
    embedRows?: Float32Array;
}

/** 读取 GPTQ int4 三件套并反量化为 f32 [out, in]。 */
function dequantGptqLinear (
    file: SafetensorFile,
    prefix: string,
    inFeatures: number,
    outFeatures: number,
    groupSize: number,
): Float32Array
{
    const qweight = readTensorBytes( file, `${ prefix }.qweight` );
    const qzeros = readTensorBytes( file, `${ prefix }.qzeros` );
    const scales = readTensorF32( file, `${ prefix }.scales` );
    return dequantizeGptqInt4( { qweight, qzeros, scales, inFeatures, outFeatures, groupSize } );
}

/** 从 F16 张量里只读 [rowStart, rowStart+rowCount) 行，避免整体展开。 */
function readF16Rows (
    file: SafetensorFile,
    name: string,
    rowStart: number,
    rowCount: number,
    rowLength: number,
): Float32Array
{
    const entry = requireTensor( file, name );
    if ( entry.dtype !== 'F16' )
    {
        throw new Error( `qwen-loader: ${ name } 期望 F16，实际 ${ entry.dtype }` );
    }
    const expected = entry.shape[ 0 ] * rowLength;
    if ( entry.shape.length !== 2 || expected !== entry.shape[ 0 ] * entry.shape[ 1 ] )
    {
        throw new Error( `qwen-loader: ${ name } 形状 ${ entry.shape } 与 rowLength=${ rowLength } 不符` );
    }
    const byteStart = file.dataStart + entry.data_offsets[ 0 ] + rowStart * rowLength * 2;
    const view = new DataView( file.buffer, byteStart, rowCount * rowLength * 2 );
    const out = new Float32Array( rowCount * rowLength );
    for ( let i = 0; i < out.length; i++ ) out[ i ] = f16BitsToF32( view.getUint16( i * 2, true ) );
    return out;
}

/** embed 分块上传的每块行数（8192 行 ≈ 29MB f32 临时副本）。 */
const EMBED_CHUNK_ROWS = 8192;

/**
 * 加载真实 Qwen2.5-0.5B int4 权重并上传到 QwenGpt。
 * 逐层加载，每层上传后即丢弃 CPU 侧 fp32 副本，控制峰值内存。
 */
export async function loadQwenWeights (
    _gpu: GpuContext,
    model: QwenGpt,
    options: QwenLoadOptions = {},
): Promise<QwenLoadResult>
{
    const config: QwenConfig = model.config;
    const C = config.nEmbd;
    const HI = qwenKvDim( config );
    const I = config.intermediate;
    const url = options.url ?? `${ QWEN_LOCAL_DIR }model.safetensors`;

    const res = await fetch( url );
    if ( !res.ok ) throw new Error( `qwen-loader: 加载失败 HTTP ${ res.status } ${ url }` );
    const raw = await res.arrayBuffer();
    const file = parseSafetensors( raw );
    let tensorCount = 0;

    // --- embed_tokens（F16，同时充当 tied lm_head） ---
    const embedEntry = requireTensor( file, 'model.embed_tokens.weight' );
    if ( embedEntry.shape.join( ',' ) !== `${ config.vocabSize },${ C }` )
    {
        throw new Error( `qwen-loader: embed_tokens 形状 [${ embedEntry.shape }]，期望 [${ config.vocabSize },${ C }]` );
    }
    model.allocF16( 'embed', config.vocabSize * C );
    for ( let start = 0; start < config.vocabSize; start += EMBED_CHUNK_ROWS )
    {
        const rows = Math.min( EMBED_CHUNK_ROWS, config.vocabSize - start );
        const chunk = readF16Rows( file, 'model.embed_tokens.weight', start, rows, C );
        model.writeF16Chunk( 'embed', start * C, chunk );
    }
    tensorCount += 1;
    const embedRows = options.keepEmbedRows
        ? readF16Rows( file, 'model.embed_tokens.weight', options.keepEmbedRows.start, options.keepEmbedRows.rows, C )
        : undefined;

    // --- 最终 RMSNorm ---
    model.uploadF32( 'finalNorm', readTensorF32( file, 'model.norm.weight' ) );
    tensorCount += 1;

    // --- 逐层 ---
    let keptLayer: QwenLayerWeights | undefined;
    for ( let i = 0; i < config.nLayer; i++ )
    {
        const p = `model.layers.${ i }`;
        const inDim = ( proj: string ) => p + '.' + proj;

        const layer: QwenLayerWeights = {
            inputNorm: readTensorF32( file, `${ p }.input_layernorm.weight` ),
            qW: dequantGptqLinear( file, inDim( 'self_attn.q_proj' ), C, C, 128 ),
            qB: readTensorF32( file, `${ p }.self_attn.q_proj.bias` ),
            kW: dequantGptqLinear( file, inDim( 'self_attn.k_proj' ), C, HI, 128 ),
            kB: readTensorF32( file, `${ p }.self_attn.k_proj.bias` ),
            vW: dequantGptqLinear( file, inDim( 'self_attn.v_proj' ), C, HI, 128 ),
            vB: readTensorF32( file, `${ p }.self_attn.v_proj.bias` ),
            oW: dequantGptqLinear( file, inDim( 'self_attn.o_proj' ), C, C, 128 ),
            postNorm: readTensorF32( file, `${ p }.post_attention_layernorm.weight` ),
            gateW: dequantGptqLinear( file, inDim( 'mlp.gate_proj' ), C, I, 128 ),
            upW: dequantGptqLinear( file, inDim( 'mlp.up_proj' ), C, I, 128 ),
            downW: dequantGptqLinear( file, inDim( 'mlp.down_proj' ), I, C, 128 ),
        };
        tensorCount += 12;

        model.uploadLayer( i, layer );
        if ( options.keepLayer === i ) keptLayer = layer;
        options.onProgress?.( i + 1, config.nLayer, `layer ${ i + 1 }/${ config.nLayer }` );
    }

    return {
        gpuBytes: model.weightBytes,
        tensorCount,
        fileBytes: raw.byteLength,
        layer: keptLayer,
        embedRows,
    };
}
