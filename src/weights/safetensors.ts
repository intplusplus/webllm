/**
 * 最小 Safetensors 解析器。
 *
 * Safetensors 布局：8 字节小端 header 长度 + JSON header + 原始 tensor bytes。
 * 这里只做 M4 第一阶段需要的能力：解析 header、定位 tensor bytes、把常见 dtype 读回 f32。
 */

export type SafetensorDtype =
    | 'F64'
    | 'F32'
    | 'F16'
    | 'BF16'
    | 'I64'
    | 'I32'
    | 'I16'
    | 'I8'
    | 'U8'
    | 'BOOL';

export interface SafetensorEntry
{
    dtype: SafetensorDtype;
    shape: number[];
    data_offsets: [ number, number ];
}

export interface SafetensorFile
{
    buffer: ArrayBuffer;
    dataStart: number;
    header: Record<string, SafetensorEntry>;
}

const ENTRY_KEY_SKIP = '__metadata__';

/**
 * 解析 safetensors 的 JSON header 字节，只做条目自洽校验（不校验整体文件是否越界）。
 * 这样即使只拿到 header（例如 HTTP Range 拉取），也能探测模型 tensor 结构。
 */
export function parseSafetensorsHeader ( headerBytes: Uint8Array ): Record<string, SafetensorEntry>
{
    const headerText = new TextDecoder( 'utf-8' ).decode( headerBytes );
    let parsed: unknown;
    try
    {
        parsed = JSON.parse( headerText );
    } catch ( err )
    {
        throw new Error( `safetensors: header JSON 解析失败：${ ( err as Error ).message }` );
    }
    if ( typeof parsed !== 'object' || parsed === null )
    {
        throw new Error( 'safetensors: header 不是 JSON object' );
    }

    const header: Record<string, SafetensorEntry> = {};
    for ( const [ name, raw ] of Object.entries( parsed as Record<string, unknown> ) )
    {
        if ( name === ENTRY_KEY_SKIP ) continue;
        if ( typeof raw !== 'object' || raw === null ) throw new Error( `safetensors: ${ name } 条目非法` );
        const item = raw as Record<string, unknown>;
        if ( typeof item.dtype !== 'string' ) throw new Error( `safetensors: ${ name } 缺少 dtype` );
        if ( !Array.isArray( item.shape ) ) throw new Error( `safetensors: ${ name } 缺少 shape` );
        if ( !Array.isArray( item.data_offsets ) || item.data_offsets.length !== 2 )
        {
            throw new Error( `safetensors: ${ name } data_offsets 非法` );
        }
        const [ start, end ] = item.data_offsets.map( Number );
        if ( !Number.isFinite( start ) || !Number.isFinite( end ) || start < 0 || end < start )
        {
            throw new Error( `safetensors: ${ name } data_offsets 数值非法` );
        }
        header[ name ] = {
            dtype: item.dtype as SafetensorDtype,
            shape: item.shape.map( Number ),
            data_offsets: [ start, end ],
        };
    }

    return header;
}

/** 解析 safetensors ArrayBuffer，并做头部/偏移量越界校验。 */
export function parseSafetensors ( buffer: ArrayBuffer ): SafetensorFile
{
    if ( buffer.byteLength < 8 ) throw new Error( 'safetensors: buffer 过短' );

    const lenView = new DataView( buffer );
    const headerLen = lenView.getUint32( 0, true );
    if ( lenView.getUint32( 4, true ) !== 0 )
    {
        throw new Error( 'safetensors: header 长度超过 4GiB，当前不支持' );
    }
    const dataStart = 8 + headerLen;
    if ( headerLen <= 0 || dataStart > buffer.byteLength )
    {
        throw new Error( `safetensors: header 长度非法 headerLen=${ headerLen } bufferSize=${ buffer.byteLength }` );
    }

    const header = parseSafetensorsHeader( new Uint8Array( buffer, 8, headerLen ) );
    for ( const [ name, entry ] of Object.entries( header ) )
    {
        if ( dataStart + entry.data_offsets[ 1 ] > buffer.byteLength )
        {
            throw new Error( `safetensors: ${ name } 数据越界 end=${ entry.data_offsets[ 1 ] }` );
        }
    }

    return { buffer, dataStart, header };
}

/** 元素总数。空 shape 表示标量。 */
export function tensorCount ( entry: SafetensorEntry ): number
{
    return entry.shape.reduce( ( a, b ) => a * b, 1 );
}

/** 获取条目，不存在则抛错。 */
export function requireTensor ( file: SafetensorFile, name: string ): SafetensorEntry
{
    const entry = file.header[ name ];
    if ( !entry ) throw new Error( `safetensors: 缺少 tensor ${ name }` );
    return entry;
}

/** 校验 dtype/shape，便于提前发现模型文件不匹配。 */
export function expectTensor (
    file: SafetensorFile,
    name: string,
    dtype: SafetensorDtype,
    shape: readonly number[],
): SafetensorEntry
{
    const entry = requireTensor( file, name );
    if ( entry.dtype !== dtype ) throw new Error( `safetensors: ${ name } dtype=${ entry.dtype }，期望 ${ dtype }` );
    if ( entry.shape.length !== shape.length || entry.shape.some( ( v, i ) => v !== shape[ i ] ) )
    {
        throw new Error( `safetensors: ${ name } shape=[${ entry.shape }]，期望 [${ shape }]` );
    }
    return entry;
}

/** IEEE-754 binary16 位模式转 f32 数值。 */
export function f16BitsToF32 ( bits: number ): number
{
    const sign = ( bits & 0x8000 ) !== 0 ? -1 : 1;
    const exp = ( bits >>> 10 ) & 0x1f;
    const mant = bits & 0x03ff;
    if ( exp === 0 ) return sign * mant * Math.pow( 2, -24 );
    if ( exp === 31 ) return mant === 0 ? sign * Infinity : NaN;
    return sign * ( 1 + mant / 1024 ) * Math.pow( 2, exp - 15 );
}

/** 读取 tensor 原始 bytes（浅拷贝，避免 offset/内存对齐问题）。 */
export function readTensorBytes ( file: SafetensorFile, name: string ): Uint8Array
{
    const entry = requireTensor( file, name );
    const start = file.dataStart + entry.data_offsets[ 0 ];
    const length = tensorCount( entry ) * dtypeByteSize( entry.dtype );
    return new Uint8Array( file.buffer.slice( start, start + length ) );
}

/** 把 tensor 读回 Float32Array。当前支持 F32/F16/BF16/U8/I8。 */
export function readTensorF32 ( file: SafetensorFile, name: string ): Float32Array
{
    const entry = requireTensor( file, name );
    const count = tensorCount( entry );
    const start = file.dataStart + entry.data_offsets[ 0 ];
    const out = new Float32Array( count );
    const view = new DataView( file.buffer, start, entry.data_offsets[ 1 ] - entry.data_offsets[ 0 ] );
    const f32 = new DataView( new ArrayBuffer( 4 ) );

    for ( let i = 0; i < count; i++ )
    {
        switch ( entry.dtype )
        {
            case 'F32':
                out[ i ] = view.getFloat32( i * 4, true );
                break;
            case 'F16':
                out[ i ] = f16BitsToF32( view.getUint16( i * 2, true ) );
                break;
            case 'BF16':
                f32.setUint32( 0, view.getUint16( i * 2, true ) << 16, true );
                out[ i ] = f32.getFloat32( 0, true );
                break;
            case 'U8':
                out[ i ] = view.getUint8( i );
                break;
            case 'I8':
                out[ i ] = view.getInt8( i );
                break;
            default:
                throw new Error( `readTensorF32: 不支持的 dtype ${ entry.dtype }（${ name }）` );
        }
    }
    return out;
}

/** dtype 元素字节大小。 */
export function dtypeByteSize ( dtype: SafetensorDtype ): number
{
    switch ( dtype )
    {
        case 'F64':
        case 'I64':
            return 8;
        case 'F32':
        case 'I32':
            return 4;
        case 'F16':
        case 'BF16':
        case 'I16':
            return 2;
        case 'I8':
        case 'U8':
        case 'BOOL':
            return 1;
        default:
            throw new Error( `未知 safetensors dtype: ${ dtype }` );
    }
}
