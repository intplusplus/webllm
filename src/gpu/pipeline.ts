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

/**
 * 批量提交：多个 compute pass 装进同一个 command encoder，最后只 submit 一次。
 *
 * 为什么需要：dispatch() 每次调用都独立创建 encoder + submit，实测每次 ~110µs CPU
 * 开销；解码一步约 360 次 dispatch ≈ 40ms 纯提交开销，与每步 942MiB 权重的
 * ~47ms 带宽下限相当，是端到端 tokens/s 的第二大瓶颈。
 *
 * 为什么需要显式 endPass()：同一 compute pass 内的 dispatch 之间没有隐式内存屏障，
 * 对同一 storage buffer 的任何读写冲突（RAW / WAR / WAW）行为未定义；
 * 只有 pass 边界才有隐式屏障。调用方必须在每个读写依赖处 endPass() 开新 pass。
 */
export class CommandBatch {
  private readonly encoder: GPUCommandEncoder;
  private pass: GPUComputePassEncoder | null = null;

  constructor(private readonly device: GPUDevice) {
    this.encoder = device.createCommandEncoder();
  }

  /** 往当前 pass 追加一次 dispatch；pass 尚未开启时会自动开启。 */
  dispatch(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: number | readonly number[],
  ): void {
    let pass = this.pass;
    if (!pass) {
      pass = this.encoder.beginComputePass();
      this.pass = pass;
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = typeof workgroups === 'number' ? [workgroups] : workgroups;
    pass.dispatchWorkgroups(wg[0], wg[1] ?? 1, wg[2] ?? 1);
  }

  /** 结束当前 pass（若有）。下一次 dispatch 会开启新 pass。 */
  endPass(): void {
    this.pass?.end();
    this.pass = null;
  }

  /** 结束剩余 pass 并把整个 encoder 提交到队列（仅一次 submit）。 */
  submit(): void {
    this.endPass();
    this.device.queue.submit([this.encoder.finish()]);
  }
}