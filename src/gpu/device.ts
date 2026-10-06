/**
 * WebGPU 设备初始化与 limits 协商。
 *
 * 背景：实测本机 maxBufferSize = 2 GiB（Chrome 硬上限，requiredLimits 只能降到适配器上限，
 * 不能超过）。这里把适配器支持的核心 limits 全部协商过来，避免使用默认的保守值。
 */

/** 需要协商的核心 limit 名称（均为 Chrome 113+ 稳定存在的核心 limit，避免请求未知键导致 requestDevice 抛错）。 */
const NEGOTIATED_LIMITS = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxUniformBufferBindingSize',
  'maxStorageBuffersPerShaderStage',
  'maxStorageTexturesPerShaderStage',
  'maxUniformBuffersPerShaderStage',
  'maxBindGroups',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension',
  'minStorageBufferOffsetAlignment',
  'minUniformBufferOffsetAlignment',
] as const;

export interface GpuContext
{
  adapter: GPUAdapter;
  device: GPUDevice;
  /** 适配器支持的上限 */
  adapterLimits: GPUSupportedLimits;
  /** 实际协商拿到的上限 */
  limits: GPUSupportedLimits;
  features: ReadonlySet<string>;
  adapterInfo: Partial<GPUAdapterInfo>;
  hasF16: boolean;
}

export async function initGpu (): Promise<GpuContext>
{
  if ( !( 'gpu' in navigator ) )
  {
    throw new Error( '当前浏览器不支持 WebGPU（navigator.gpu 不存在）。请使用 Chrome / Edge 113+。' );
  }

  const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
  if ( !adapter )
  {
    throw new Error( '未能获取 GPUAdapter（可能被驱动或浏览器策略禁用）。' );
  }

  const adapterInfo: Partial<GPUAdapterInfo> =
    ( adapter as unknown as { info?: GPUAdapterInfo } ).info ?? {};

  // 用适配器上限构造 requiredLimits
  const requiredLimits: Record<string, number> = {};
  const raw = adapter.limits as unknown as Record<string, number>;
  for ( const name of NEGOTIATED_LIMITS )
  {
    const value = raw[ name ];
    if ( typeof value === 'number' ) requiredLimits[ name ] = value;
  }

  let device: GPUDevice;
  const requiredFeatures = [ ...adapter.features ] as GPUFeatureName[];
  try
  {
    // requiredFeatures 只请求适配器实际支持的 feature；这样 shader-f16 等扩展才会在 device 上真正启用。
    device = await adapter.requestDevice( { requiredLimits, requiredFeatures } );
  } catch ( err )
  {
    console.warn( 'limits/features 协商失败，回退到默认 device：', err );
    device = await adapter.requestDevice();
  }

  device.lost.then( ( info ) =>
  {
    console.error( `WebGPU device lost: ${ info.reason } - ${ info.message }` );
  } );

  // 兜底：把未被 error scope 捕获的校验错误打印出来，便于定位 WGSL / 绑定问题。
  device.addEventListener( 'uncapturederror', ( event ) =>
  {
    const err = ( event as GPUUncapturedErrorEvent ).error;
    console.error( '[WebGPU] uncaptured error:', err.message );
  } );

  return {
    adapter,
    device,
    adapterLimits: adapter.limits,
    limits: device.limits,
    features: device.features,
    adapterInfo,
    hasF16: device.features.has( 'shader-f16' ),
  };
}