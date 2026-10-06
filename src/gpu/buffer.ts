/** GPUBuffer 创建、上传与回读的底层工具。 */

export const ALIGN = 256;

/** 向上对齐到 align 的整数倍。 */
export function alignTo ( n: number, align = ALIGN ): number
{
  return Math.ceil( n / align ) * align;
}

/** 指向 arena 中某个张量的切片（offset 为字节偏移）。 */
export interface Tensor
{
  buffer: GPUBuffer;
  offset: number;
  length: number;
  shape: number[];
}

export function createStorageBuffer (
  device: GPUDevice,
  byteLength: number,
  label?: string,
  usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE |
    GPUBufferUsage.COPY_DST |
    GPUBufferUsage.COPY_SRC,
): GPUBuffer
{
  return device.createBuffer( {
    label,
    size: Math.max( alignTo( byteLength ), 4 ),
    usage,
  } );
}

/** 把 fp32 数据写入 buffer 的指定元素偏移处。 */
export function writeF32 (
  device: GPUDevice,
  buffer: GPUBuffer,
  offsetElements: number,
  data: Float32Array,
): void
{
  device.queue.writeBuffer(
    buffer,
    offsetElements * 4,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength,
  );
}

/** f32→f16 位模式转换用的共享暂存区：避免每个元素都分配 ArrayBuffer/DataView。 */
const F32_SCRATCH = new DataView( new ArrayBuffer( 4 ) );

/** 数值按 IEEE-754 binary16 舍入到最近偶数，返回 16 位位模式。 */
function f32ToF16Bits ( value: number ): number
{
  const f = Math.fround( value );
  if ( Number.isNaN( f ) ) return 0x7e00;

  F32_SCRATCH.setFloat32( 0, f, true );
  const bits = F32_SCRATCH.getUint32( 0, true );

  const sign = ( bits & 0x80000000 ) >>> 16;
  const exp = ( bits & 0x7f800000 ) >>> 23;
  const mant = bits & 0x007fffff;

  if ( exp === 0xff ) return sign | ( mant !== 0 ? 0x7e00 : 0x7c00 );

  let e = exp - 127;
  if ( e > 15 ) return sign | 0x7c00;
  if ( e < -24 ) return sign;

  if ( e < -14 )
  {
    const m = mant | 0x00800000;
    const shift = -( e + 1 );
    let half = m >>> shift;
    const rem = m & ( ( 1 << shift ) - 1 );
    const halfway = 1 << ( shift - 1 );
    if ( rem > halfway || ( rem === halfway && ( half & 1 ) !== 0 ) ) half++;
    return sign | half;
  }

  let half = mant >>> 13;
  const rem = mant & 0x1fff;
  const halfway = 0x1000;
  if ( rem > halfway || ( rem === halfway && ( half & 1 ) !== 0 ) )
  {
    half++;
    if ( half === 0x400 )
    {
      half = 0;
      e++;
      if ( e > 15 ) return sign | 0x7c00;
    }
  }
  return sign | ( ( e + 15 ) << 10 ) | half;
}

/** 把 fp32 数组以 IEEE-754 binary16 写入 buffer 的指定元素偏移处。 */
export function writeF16 (
  device: GPUDevice,
  buffer: GPUBuffer,
  offsetElements: number,
  data: Float32Array,
): void
{
  const paddedLength = Math.ceil( data.length / 2 ) * 2;
  const words = new Uint16Array( paddedLength );
  for ( let i = 0; i < data.length; i++ ) words[ i ] = f32ToF16Bits( data[ i ] );
  device.queue.writeBuffer(
    buffer,
    offsetElements * 2,
    words.buffer as ArrayBuffer,
    words.byteOffset,
    words.byteLength,
  );
}

/** 从 buffer 读回 fp32 数据（同步等待 GPU 完成）。 */
export async function readbackF32 (
  device: GPUDevice,
  buffer: GPUBuffer,
  offsetElements: number,
  length: number,
): Promise<Float32Array>
{
  const byteLength = Math.max( length * 4, 4 );
  const staging = device.createBuffer( {
    size: alignTo( byteLength ),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  } );

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer( buffer, offsetElements * 4, staging, 0, byteLength );
  device.queue.submit( [ encoder.finish() ] );

  await staging.mapAsync( GPUMapMode.READ );
  const out = new Float32Array( staging.getMappedRange( 0, byteLength ).slice( 0 ) );
  staging.unmap();
  staging.destroy();
  return out;
}

/** 从 buffer 读回原始字节（同步等待 GPU 完成）。 */
export async function readbackBytes (
  device: GPUDevice,
  buffer: GPUBuffer,
  byteOffset: number,
  byteLength: number,
): Promise<ArrayBuffer>
{
  const size = Math.max( byteLength, 4 );
  const staging = device.createBuffer( {
    size: alignTo( size ),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  } );

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer( buffer, byteOffset, staging, 0, size );
  device.queue.submit( [ encoder.finish() ] );

  await staging.mapAsync( GPUMapMode.READ );
  const out = staging.getMappedRange( 0, size ).slice( 0 );
  staging.unmap();
  staging.destroy();
  return out;
}

/** 把 u32 数据写入 buffer 的指定元素偏移处（用于 token id 等）。 */
export function writeU32 (
  device: GPUDevice,
  buffer: GPUBuffer,
  offsetElements: number,
  data: Uint32Array,
): void
{
  device.queue.writeBuffer(
    buffer,
    offsetElements * 4,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength,
  );
}

/** 创建并写入一个 uniform buffer（大小向上对齐到 16 字节）。 */
export function createUniform (
  device: GPUDevice,
  data: ArrayBufferView,
  label?: string,
): GPUBuffer
{
  const size = Math.max( alignTo( data.byteLength, 16 ), 16 );
  const buffer = device.createBuffer( {
    label,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  } );
  device.queue.writeBuffer(
    buffer,
    0,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength,
  );
  return buffer;
}

/** 把 Arena 张量转成 bind group 的 resource entry。 */
export function tensorResource ( t: Tensor ): GPUBufferBinding
{
  return { buffer: t.buffer, offset: t.offset, size: t.length * 4 };
}

/**
 * 按字段顺序打包 uniform 结构体。字段均为 4 字节标量，末尾自动补齐到 16 字节。
 * 字段顺序必须与 WGSL struct 声明一致。
 */
export class StructPacker
{
  private readonly raw = new ArrayBuffer( 128 );
  private readonly view = new DataView( this.raw );
  private pos = 0;

  u32 ( value: number ): this
  {
    this.view.setUint32( this.pos, value, true );
    this.pos += 4;
    return this;
  }

  f32 ( value: number ): this
  {
    this.view.setFloat32( this.pos, value, true );
    this.pos += 4;
    return this;
  }

  bytes (): Uint8Array
  {
    return new Uint8Array( this.raw, 0, Math.max( 16, Math.ceil( this.pos / 16 ) * 16 ) );
  }
}