import type { GpuContext } from '../../gpu/device';
import { f16BitsToF32, parseSafetensorsHeader, type SafetensorEntry } from '../../weights/safetensors';
import { dequantizeGptqInt4 } from '../../weights/quant';

/** 权重已下载到本地 public/models 下，由 Vite 静态服务提供（支持 Range）。 */
const LOCAL_DIR = '/models/qwen2.5-0.5b-int4/';
const WEIGHT_FILE = `${ LOCAL_DIR }model.safetensors`;

/** 用 HTTP Range 只取指定字节区间，避免把 459MB 全量读进内存。 */
async function fetchRange ( url: string, start: number, end: number ): Promise<Uint8Array>
{
    const res = await fetch( url, { headers: { Range: `bytes=${ start }-${ end }` } } );
    if ( !res.ok ) throw new Error( `HTTP ${ res.status } ${ res.statusText } ${ url }` );
    const buf = await res.arrayBuffer();
    // 服务器若忽略 Range 返回 200 全量，这里自行截取所需区间。
    if ( res.status === 200 && buf.byteLength > end - start + 1 )
    {
        return new Uint8Array( buf, start, end - start + 1 );
    }
    return new Uint8Array( buf );
}

/** 读取一个 tensor 的原始字节。 */
async function fetchTensorBytes ( entry: SafetensorEntry, dataStart: number ): Promise<Uint8Array>
{
    const start = dataStart + entry.data_offsets[ 0 ];
    const end = dataStart + entry.data_offsets[ 1 ] - 1;
    return fetchRange( WEIGHT_FILE, start, end );
}

/**
 * M4 前置：核对本地 Qwen2.5-0.5B-Instruct-GPTQ-Int4 的真实张量格式。
 *
 * 覆盖：safetensors header 解析、Qwen2 张量命名/形状、GPTQ int4（I32 打包 + qzeros 存
 * zero-1 + F16 scales + 恒等 g_idx）实际反量化，并统计权重分布是否合理。
 */
export async function testQwenWeightFormat ( _gpu: GpuContext ): Promise<string>
{
    // 1) header
    const lenBytes = await fetchRange( WEIGHT_FILE, 0, 7 );
    const headerLen = new DataView( lenBytes.buffer, lenBytes.byteOffset, 8 ).getUint32( 0, true );
    if ( headerLen <= 0 || headerLen > 64 * 1024 * 1024 )
    {
        throw new Error( `header 长度异常：${ headerLen }` );
    }
    const headerBytes = await fetchRange( WEIGHT_FILE, 8, 8 + headerLen - 1 );
    const header = parseSafetensorsHeader( headerBytes );
    const dataStart = 8 + headerLen;
    const names = Object.keys( header );

    const fail = ( msg: string ): never =>
    {
        throw new Error( `${ msg }；实际 tensors=${ names.length } dtype分布=${ dtypeText }` );
    };
    const dtypes = new Map<string, number>();
    for ( const name of names )
    {
        const dt = header[ name ].dtype;
        dtypes.set( dt, ( dtypes.get( dt ) ?? 0 ) + 1 );
    }
    const dtypeText = [ ...dtypes.entries() ].map( ( [ k, v ] ) => `${ k }=${ v }` ).join( ' ' );

    // 2) Qwen2ForCausalLM 结构核对（config.json: 24 层 / 14 heads / 2 kv heads / hidden 896）
    if ( header[ 'model.embed_tokens.weight' ]?.shape.join( ',' ) !== '151936,896' )
    {
        fail( 'embed_tokens 形状不符' );
    }
    if ( header[ 'lm_head.weight' ] )
    {
        fail( 'tie_word_embeddings=true，不应存在独立 lm_head.weight' );
    }
    for ( const suffix of [ 'input_layernorm.weight', 'post_attention_layernorm.weight', 'self_attn.q_proj.bias', 'self_attn.q_proj.qweight', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias', 'self_attn.o_proj.qweight', 'mlp.gate_proj.qweight', 'mlp.up_proj.qweight', 'mlp.down_proj.qweight', 'mlp.gate_proj.scales' ] )
    {
        if ( !header[ `model.layers.0.${ suffix }` ] ) fail( `缺少张量 model.layers.0.${ suffix }` );
    }
    if ( !names.includes( 'model.norm.weight' ) ) fail( '缺少 model.norm.weight' );
    for ( let l = 0; l < 24; l++ )
    {
        if ( !header[ `model.layers.${ l }.self_attn.q_proj.qweight` ] ) fail( `缺少第 ${ l } 层权重` );
    }

    // 3) 取 layer0 k_proj（in=896, out=128）做真实反量化
    const p = 'model.layers.0.self_attn.k_proj';
    const qwEntry = header[ `${ p }.qweight` ];
    const qzEntry = header[ `${ p }.qzeros` ];
    const scEntry = header[ `${ p }.scales` ];
    const giEntry = header[ 'model.layers.0.mlp.gate_proj.g_idx' ];
    if ( !qwEntry || !qzEntry || !scEntry || !giEntry ) fail( '缺少 GPTQ int4 相关张量' );

    const inFeatures = 896;
    const outFeatures = 128;
    if ( qwEntry.dtype !== 'I32' || qwEntry.shape.join( ',' ) !== `${ inFeatures / 8 },${ outFeatures }` )
    {
        fail( `qweight 期望 I32[${ inFeatures / 8 },${ outFeatures }]，实际 ${ qwEntry.dtype }[${ qwEntry.shape }]` );
    }
    if ( qzEntry.shape.join( ',' ) !== `${ inFeatures / 128 },${ outFeatures / 8 }` )
    {
        fail( `qzeros 期望 [${ inFeatures / 128 },${ outFeatures / 8 }]，实际 [${ qzEntry.shape }]` );
    }
    if ( scEntry.dtype !== 'F16' || scEntry.shape.join( ',' ) !== `${ inFeatures / 128 },${ outFeatures }` )
    {
        fail( `scales 期望 F16[${ inFeatures / 128 },${ outFeatures }]，实际 ${ scEntry.dtype }[${ scEntry.shape }]` );
    }

    const [ qwBytes, qzBytes, scBytes, giBytes ] = await Promise.all( [
        fetchTensorBytes( qwEntry, dataStart ),
        fetchTensorBytes( qzEntry, dataStart ),
        fetchTensorBytes( scEntry, dataStart ),
        fetchTensorBytes( giEntry, dataStart ),
    ] );

    // qzeros 存的是「真实 zero - 1」：对称 4bit 的 zero=8 应存成 7
    const qzView = new DataView( qzBytes.buffer, qzBytes.byteOffset, qzBytes.byteLength );
    const storedZeros = new Set<number>();
    for ( let i = 0; i < qzBytes.byteLength / 4; i++ )
    {
        const word = qzView.getUint32( i * 4, true );
        for ( let k = 0; k < 8; k++ ) storedZeros.add( ( word >>> ( 4 * k ) ) & 0x0f );
    }
    if ( storedZeros.size !== 1 || !storedZeros.has( 7 ) )
    {
        fail( `qzeros nibble 期望恒为 7（zero=8-1），实际 ${ [ ...storedZeros ].sort() }` );
    }

    // desc_act=false ⇒ g_idx 为恒等映射 i/128
    const giView = new DataView( giBytes.buffer, giBytes.byteOffset, giBytes.byteLength );
    for ( let i = 0; i < giBytes.byteLength / 4; i++ )
    {
        if ( giView.getInt32( i * 4, true ) !== ( ( i / 128 ) | 0 ) ) fail( `g_idx[${ i }] 非恒等映射` );
    }

    const scView = new DataView( scBytes.buffer, scBytes.byteOffset, scBytes.byteLength );
    const scales = new Float32Array( scBytes.byteLength / 2 );
    for ( let i = 0; i < scales.length; i++ ) scales[ i ] = f16BitsToF32( scView.getUint16( i * 2, true ) );
    let maxScale = 0;
    for ( const s of scales ) maxScale = Math.max( maxScale, Math.abs( s ) );

    const deq = dequantizeGptqInt4( {
        qweight: qwBytes,
        qzeros: qzBytes,
        scales,
        inFeatures,
        outFeatures,
        groupSize: 128,
    } );

    let sum = 0;
    let sumSq = 0;
    let maxAbs = 0;
    for ( const w of deq )
    {
        if ( !Number.isFinite( w ) ) fail( '反量化结果出现 NaN/Inf' );
        sum += w;
        sumSq += w * w;
        maxAbs = Math.max( maxAbs, Math.abs( w ) );
    }
    const mean = sum / deq.length;
    const std = Math.sqrt( Math.max( 0, sumSq / deq.length - mean * mean ) );
    // 与 int4 量化范围自洽：|w| ≤ 8 * max(scale)
    if ( maxAbs > 8 * maxScale * 1.0001 )
    {
        fail( `反量化越界：maxAbs=${ maxAbs } > 8*maxScale=${ 8 * maxScale }` );
    }
    if ( std < 1e-3 || std > 1e-1 )
    {
        fail( `反量化分布异常：std=${ std.toExponential( 2 ) }（期望 ~1e-2）` );
    }

    const cfg = await ( await fetch( `${ LOCAL_DIR }config.json` ) ).json() as Record<string, unknown>;
    const q = cfg.quantization_config as Record<string, unknown>;

    return [
        `本地文件 headerLen=${ headerLen } tensors=${ names.length }（${ dtypeText }）`,
        `结构：embed[151936,896] F16、无独立 lm_head（tied）、24 层、hidden=896、kv_heads=2`,
        `GPTQ：bits=${ q.bits } group_size=${ q.group_size } sym=${ q.sym } desc_act=${ q.desc_act }；qweight I32[112,128] → qzeros I32[7,16]（存储 zero 恒为 7 = 8-1）→ scales F16[7,128]；g_idx 恒等`,
        `layer0.k_proj 反量化：n=${ deq.length } mean=${ mean.toExponential( 2 ) } std=${ std.toExponential( 2 ) } maxAbs=${ maxAbs.toExponential( 2 ) }（≤8*maxScale=${ ( 8 * maxScale ).toExponential( 2 ) }，尺度自洽）`,
    ].join( '；' );
}
