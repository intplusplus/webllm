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
  private profile: BatchProfile | null = null;

  constructor(private readonly device: GPUDevice) {
    this.encoder = device.createCommandEncoder();
  }

  /** 开启剖析：之后每个 pass 都会写一对时间戳。必须在第一次 dispatch 前调用。 */
  enableProfile(profile: BatchProfile): void {
    if (this.pass) throw new Error('enableProfile 必须在首次 dispatch 之前调用');
    this.profile = profile;
  }

  /** 往当前 pass 追加一次 dispatch；pass 尚未开启时会自动开启。 */
  dispatch(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: number | readonly number[],
  ): void {
    let pass = this.pass;
    if (!pass) {
      let ts: GPUComputePassTimestampWrites | undefined;
      if (this.profile) {
        const p = this.profile;
        if (p.passCount >= p.maxPasses) throw new Error(`剖析 pass 数超出上限 ${ p.maxPasses }`);
        ts = {
          querySet: p.querySet,
          beginningOfPassWriteIndex: p.passCount * 2,
          endOfPassWriteIndex: p.passCount * 2 + 1,
        };
        p.passCount++;
      }
      pass = this.encoder.beginComputePass(ts ? { timestampWrites: ts } : {});
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
    // resolve + 回拷必须在 encoder.finish() 之前编码；它们排在队列上
    // kernel 之后，readBatchProfile 的 mapAsync 会等它们完成
    if (this.profile) {
      const p = this.profile;
      this.encoder.resolveQuerySet(p.querySet, 0, p.passCount * 2, p.resolveBuffer, 0);
      this.encoder.copyBufferToBuffer(p.resolveBuffer, 0, p.readBuffer, 0, p.passCount * 2 * 8);
    }
    const commandBuffer = this.encoder.finish();
    this.device.queue.submit([commandBuffer]);
  }
}

/** 一次剖析会话的资源与状态。 */
export interface BatchProfile {
  querySet: GPUQuerySet;
  resolveBuffer: GPUBuffer;
  /** 时间戳单位为纳秒（Dawn 的 timestampPeriod 恒为 1）。 */
  readBuffer: GPUBuffer;
  /** 每个 pass 占 2 个 query（begin/end）。 */
  maxPasses: number;
  passCount: number;
}

/** 创建剖析资源：querySet + resolve/read 缓冲。用完必须 destroy()。 */
export function createBatchProfile(device: GPUDevice, maxPasses: number): BatchProfile {
  const querySet = device.createQuerySet({ type: 'timestamp', count: maxPasses * 2 });
  const size = maxPasses * 2 * 8;
  const resolveBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  return { querySet, resolveBuffer, readBuffer, maxPasses, passCount: 0 };
}

/** 提交剖析批后调用：等待并读回每 pass 的耗时（纳秒，长度 = passCount）。 */
export async function readBatchProfile(
  device: GPUDevice,
  profile: BatchProfile,
): Promise<BigInt64Array> {
  await device.queue.onSubmittedWorkDone();
  await profile.readBuffer.mapAsync(GPUMapMode.READ);
  const raw = new BigInt64Array(profile.readBuffer.getMappedRange().slice(0));
  profile.readBuffer.unmap();
  const out = new BigInt64Array(profile.passCount);
  for (let i = 0; i < profile.passCount; i++) {
    // begin/end 差值；异常负值按 0 处理
    const delta = raw[i * 2 + 1] - raw[i * 2];
    out[i] = delta > 0n ? delta : 0n;
  }
  return out;
}