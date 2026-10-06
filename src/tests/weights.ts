import type { GpuContext } from '../gpu/device';
import { checkTolerance } from '../reference/cpu-ref';
import { expectTensor, parseSafetensors, readTensorF32 } from '../weights/safetensors';
import { dequantizeGptqInt4 } from '../weights/quant';

/** 测试本地构造一个最小 safetensors 文件。 */
function buildSafetensors (): ArrayBuffer
{
  const bias = new Uint8Array( new Float32Array( [ 1, -2, 0.5 ] ).buffer );
  const half = new Uint8Array( [ 0x00, 0x3c, 0x00, 0xb8 ] ); // f16: 1, -0.5
  const bfloat = new Uint8Array( [ 0x80, 0x3f, 0x00, 0xc0 ] ); // bf16: 1, -2
  const tensors = [
    { name: 'bias', dtype: 'F32', shape: [ 3 ], data: bias },
    { name: 'half', dtype: 'F16', shape: [ 2 ], data: half },
    { name: 'bfloat', dtype: 'BF16', shape: [ 2 ], data: bfloat },
  ];

  const header: Record<string, unknown> = { __metadata__: { format: 'test' } };
  let offset = 0;
  for ( const t of tensors )
  {
    header[ t.name ] = { dtype: t.dtype, shape: t.shape, data_offsets: [ offset, offset + t.data.byteLength ] };
    offset += t.data.byteLength;
  }

  const headerBytes = new TextEncoder().encode( JSON.stringify( header ) );
  const total = 8 + headerBytes.byteLength + offset;
  const buffer = new ArrayBuffer( total );
  const view = new DataView( buffer );
  view.setUint32( 0, headerBytes.byteLength, true );
  view.setUint32( 4, 0, true );
  new Uint8Array( buffer, 8, headerBytes.byteLength ).set( headerBytes );

  let cursor = 8 + headerBytes.byteLength;
  for ( const t of tensors )
  {
    new Uint8Array( buffer, cursor, t.data.byteLength ).set( t.data );
    cursor += t.data.byteLength;
  }
  return buffer;
}

/** M4-1：safetensors header/bytes 解析 + GPTQ int4 反量化。 */
export async function testWeightsFormat ( _gpu: GpuContext ): Promise<string>
{
  const file = parseSafetensors( buildSafetensors() );
  expectTensor( file, 'bias', 'F32', [ 3 ] );
  expectTensor( file, 'half', 'F16', [ 2 ] );
  expectTensor( file, 'bfloat', 'BF16', [ 2 ] );
  const got = readTensorF32( file, 'bias' );
  const expected = Float32Array.from( [ 1, -2, 0.5 ] );
  const r1 = checkTolerance( got, expected, 1e-6, 1e-6 );
  if ( !r1.ok ) throw new Error( `safetensors F32 解析失败：maxAbs=${ r1.maxAbs }` );

  const gotHalf = readTensorF32( file, 'half' );
  const gotBf16 = readTensorF32( file, 'bfloat' );
  const rHalf = checkTolerance( gotHalf, Float32Array.from( [ 1, -0.5 ] ), 1e-6, 1e-6 );
  const rBf16 = checkTolerance( gotBf16, Float32Array.from( [ 1, -2 ] ), 1e-6, 1e-6 );
  if ( !rHalf.ok ) throw new Error( `safetensors F16 解析失败：maxAbs=${ rHalf.maxAbs }` );
  if ( !rBf16.ok ) throw new Error( `safetensors BF16 解析失败：maxAbs=${ rBf16.maxAbs }` );

  // --- GPTQ int4：按真实打包格式（qweight I32 [in/8,out]，qzeros 存 zero-1）构造数据 ---
  const inF = 16;
  const outF = 5;
  const groupSize = 8;
  const packFactor = 8;

  // q[o][i]：人工给定的 4bit 码字，覆盖 0..15 全量程。
  const q: number[][] = [];
  for ( let o = 0; o < outF; o++ )
  {
    q.push( Array.from( { length: inF }, ( _, i ) => ( o * 7 + i * 3 ) & 0x0f ) );
  }
  const scales = new Float32Array( ( inF / groupSize ) * outF );
  for ( let g = 0; g < inF / groupSize; g++ )
  {
    for ( let o = 0; o < outF; o++ ) scales[ g * outF + o ] = 0.01 * ( g + 1 ) * ( o + 1 );
  }

  // qweight：每个 int32 的低 4 位对应最小的 input 索引
  const qweightBytes = new Uint8Array( ( inF / packFactor ) * outF * 4 );
  const qwView = new DataView( qweightBytes.buffer );
  for ( let p = 0; p < inF / packFactor; p++ )
  {
    for ( let o = 0; o < outF; o++ )
    {
      let word = 0;
      for ( let k = 0; k < packFactor; k++ ) word |= ( q[ o ][ p * packFactor + k ] & 0x0f ) << ( 4 * k );
      qwView.setUint32( ( p * outF + o ) * 4, word >>> 0, true );
    }
  }
  // qzeros：对称 4bit 的真实 zero = 8，存储值 = 7
  const zeroCols = Math.ceil( outF / packFactor );
  const qzerosBytes = new Uint8Array( ( inF / groupSize ) * zeroCols * 4 );
  const qzView = new DataView( qzerosBytes.buffer );
  const storedZero = 7;
  for ( let g = 0; g < inF / groupSize; g++ )
  {
    for ( let oc = 0; oc < zeroCols; oc++ )
    {
      let word = 0;
      for ( let k = 0; k < packFactor; k++ ) word |= storedZero << ( 4 * k );
      qzView.setUint32( ( g * zeroCols + oc ) * 4, word >>> 0, true );
    }
  }

  const want = new Float32Array( outF * inF );
  for ( let o = 0; o < outF; o++ )
  {
    for ( let i = 0; i < inF; i++ )
    {
      const g = Math.floor( i / groupSize );
      want[ o * inF + i ] = ( q[ o ][ i ] - ( storedZero + 1 ) ) * scales[ g * outF + o ];
    }
  }

  const deq = dequantizeGptqInt4( {
    qweight: qweightBytes,
    qzeros: qzerosBytes,
    scales,
    inFeatures: inF,
    outFeatures: outF,
    groupSize,
  } );
  const r2 = checkTolerance( deq, want, 1e-6, 1e-6 );
  if ( !r2.ok ) throw new Error( `GPTQ int4 反量化失败：maxAbs=${ r2.maxAbs }` );

  // 未提供 qzeros 时退化为对称量化 zero = 2^(bits-1)
  const deqSym = dequantizeGptqInt4( {
    qweight: qweightBytes,
    scales,
    inFeatures: inF,
    outFeatures: outF,
    groupSize,
  } );
  const r3 = checkTolerance( deqSym, want, 1e-6, 1e-6 );
  if ( !r3.ok ) throw new Error( `GPTQ int4 对称路径失败：maxAbs=${ r3.maxAbs }` );

  return `safetensors F32/F16/BF16 maxAbs=0；GPTQ int4 (in=${ inF } out=${ outF } group=${ groupSize }) maxAbs=${ r2.maxAbs.toExponential( 2 ) }，对称退路 maxAbs=${ r3.maxAbs.toExponential( 2 ) }`;
}
