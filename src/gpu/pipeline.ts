/** WGSL 着色器模块与 compute pipeline 的创建工具。 */

export function createShaderModule(device: GPUDevice, code: string, label?: string): GPUShaderModule {
  return device.createShaderModule({ label, code });
}

/**
 * 用 `layout: 'auto'` 创建 compute pipeline。
 * 后续算子都会用同一入口名 `main`，便于统一调用。
 */
export function createComputePipeline(
  device: GPUDevice,
  code: string,
  entryPoint = 'main',
  label?: string,
): GPUComputePipeline {
  return device.createComputePipeline({
    label,
    layout: 'auto',
    compute: {
      module: createShaderModule(device, code, label),
      entryPoint,
    },
  });
}

/** 单次 dispatch（一个 bind group，最多 3 维 workgroup）。 */
export function dispatch(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroups: number | readonly number[],
): void {
  const wg = typeof workgroups === 'number' ? [workgroups] : workgroups;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wg[0], wg[1] ?? 1, wg[2] ?? 1);
  pass.end();
  device.queue.submit([encoder.finish()]);
}