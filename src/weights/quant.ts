/**
 * GPTQ int4 反量化（对齐 AutoGPTQ / exllama 的真实打包格式）。
 *
 * 已在 Qwen2.5-0.5B-Instruct-GPTQ-Int4 上实测确认（config.json: bits=4, group_size=128,
 * sym=true, desc_act=false）：
 *   qweight I32 [in/8, out]       每个 int32 装 8 个 input 通道的 int4，低 4 位对应最小 input 索引
 *   qzeros  I32 [in/group, out/8] 每个 int32 装 8 个 output 通道的 zero，实测 nibble = 7
 *   scales  F16 [in/group, out]
 *   g_idx   I32 [in]              desc_act=false 时恒等于 i / group_size，可忽略
 *
 * exllama 约定：qzeros 存的是「真实 zero point - 1」，故反量化须用
 *   real = (q - (qzeros_nibble + 1)) * scale
 * 对称 4bit 的真实 zero point 是 2^(bits-1) = 8，存储值即 7。
 */

export interface GptqInt4Params
{
    /** GPTQ qweight 原始字节（I32，[inFeatures/packFactor, outFeatures] 行主序）。 */
    qweight: Uint8Array;
    /** GPTQ qzeros 原始字节（I32，[inFeatures/groupSize, outFeatures/packFactor]）；对称量化未存时可传 null。 */
    qzeros?: Uint8Array | null;
    /** 已解码为 f32 的 scales，[inFeatures/groupSize, outFeatures]。 */
    scales: Float32Array;
    inFeatures: number;
    outFeatures: number;
    /** 量化分组大小，Qwen int4 为 128。 */
    groupSize?: number;
    /** 量化位宽，当前支持 4。 */
    bits?: number;
}

/** 把 Uint8Array 包成 DataView，避免 byteOffset 未按 4 字节对齐时的构造异常。 */
function asView ( bytes: Uint8Array ): DataView
{
    return new DataView( bytes.buffer, bytes.byteOffset, bytes.byteLength );
}

/**
 * 反量化 GPTQ int4 权重，输出 [outFeatures, inFeatures] 行主序 fp32，
 * 与 Linear 的 w[out, in] 布局一致，可直接上传给 GEMM。
 */
export function dequantizeGptqInt4 ( params: GptqInt4Params ): Float32Array
{
    const groupSize = params.groupSize ?? 128;
    const bits = params.bits ?? 4;
    if ( bits !== 4 ) throw new Error( `GPTQ: 当前仅支持 4bit，收到 bits=${ bits }` );
    if ( groupSize <= 0 ) throw new Error( 'GPTQ: groupSize 必须为正' );
    if ( params.inFeatures <= 0 || params.outFeatures <= 0 ) throw new Error( 'GPTQ: in/out 必须为正' );

    const inF = params.inFeatures;
    const outF = params.outFeatures;
    const packFactor = 32 / bits; // 每个 int32 装 8 个 int4
    const mask = ( 1 << bits ) - 1;
    const packRows = Math.ceil( inF / packFactor );
    const packCols = Math.ceil( outF / packFactor );
    const groups = Math.ceil( inF / groupSize );

    if ( params.qweight.byteLength < packRows * outF * 4 )
    {
        throw new Error( `GPTQ: qweight 字节不足，需要 ${ packRows * outF * 4 }，实际 ${ params.qweight.byteLength }` );
    }
    if ( params.scales.length < groups * outF )
    {
        throw new Error( `GPTQ: scales 数量不足，需要 ${ groups * outF }，实际 ${ params.scales.length }` );
    }

    const qwView = asView( params.qweight );
    const qzView = params.qzeros ? asView( params.qzeros ) : null;
    if ( qzView && qzView.byteLength < groups * packCols * 4 )
    {
        throw new Error( `GPTQ: qzeros 字节不足，需要 ${ groups * packCols * 4 }，实际 ${ qzView.byteLength }` );
    }
    /** 对称量化且未存 qzeros 时的真实 zero point。 */
    const symmetricZero = 1 << ( bits - 1 );

    if ( groupSize % packFactor !== 0 )
    {
        throw new Error( `GPTQ: groupSize=${ groupSize } 必须是 packFactor=${ packFactor } 的整数倍` );
    }

    const out = new Float32Array( outF * inF );

    for ( let packRow = 0; packRow < packRows; packRow++ )
    {
        const rowBase = packRow * packFactor;
        // 同一 packRow 内的 input 下标都落在同一个量化组，故 zero 与 scale 可提到 k 循环之外
        const g = ( rowBase / groupSize ) | 0;
        const qwRowBase = packRow * outF;

        for ( let o = 0; o < outF; o++ )
        {
            const word = qwView.getUint32( ( qwRowBase + o ) * 4, true );
            const scale = params.scales[ g * outF + o ];

            let zero: number;
            if ( qzView )
            {
                const zWord = qzView.getUint32( ( g * packCols + ( ( o / packFactor ) | 0 ) ) * 4, true );
                zero = ( ( zWord >>> ( bits * ( o % packFactor ) ) ) & mask ) + 1;
            }
            else
            {
                zero = symmetricZero;
            }

            const outIndex = o * inF;
            for ( let k = 0; k < packFactor; k++ )
            {
                const i = rowBase + k;
                if ( i >= inF ) break;
                const q = ( word >>> ( bits * k ) ) & mask;
                out[ outIndex + i ] = ( q - zero ) * scale;
            }
        }
    }

    return out;
}