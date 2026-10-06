import { alignTo, type Tensor } from './buffer';

/**
 * 激活张量分配器。
 *
 * 为什么不是「单块大 buffer + offset」：WebGPU 禁止在同一次 compute pass 中把同一个 GPUBuffer
 * 同时绑定为只读 storage 与可写 storage。Dawn 的 usage 校验是 buffer 粒度的，不区分 offset 区间，
 * 因此「所有激活共享一块 buffer」的方案必然触发
 *   "[Buffer] usage (Storage(read-write)|Storage(read-only)) includes writable usage and another usage
 *    in the same synchronization scope"
 * 并导致整个 command encoder 校验失败（dispatch 全部不执行）。
 *
 * 这里改为「每个张量独立 GPUBuffer」，并按字节尺寸做对象池复用：reset() 归还、后续前向复用同尺寸
 * buffer，避免每步重复创建。单个 kernel 最多绑定 4 个 storage buffer，远低于
 * maxStorageBuffersPerShaderStage（默认 8），因此不需要用一个 buffer 装下所有张量。
 */
export class Arena {
  private readonly device: GPUDevice;
  private readonly pool = new Map<number, GPUBuffer[]>();
  private active: { size: number; buffer: GPUBuffer }[] = [];
  private used = 0;
  /** 容量提示，仅用于展示；实际按需分配，不再强制上限。 */
  readonly capacity: number;

  constructor(device: GPUDevice, byteLength = 0) {
    this.device = device;
    this.capacity = Math.max(alignTo(byteLength, 256), 256);
  }

  get usedBytes(): number {
    return this.used;
  }

  /** 归还本次前向分配的所有张量，供下一轮复用。 */
  reset(): void {
    for (const { size, buffer } of this.active) {
      const list = this.pool.get(size);
      if (list) list.push(buffer);
      else this.pool.set(size, [buffer]);
    }
    this.active = [];
    this.used = 0;
  }

  /** 分配一个 fp32 张量，返回其独立 buffer 切片（offset 恒为 0）。 */
  allocFloat32(length: number, shape: number[] = [length]): Tensor {
    const size = Math.max(alignTo(length * 4), 4);
    let buffer = this.pool.get(size)?.pop();
    if (!buffer) {
      buffer = this.device.createBuffer({
        label: 'activation',
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }
    this.active.push({ size, buffer });
    this.used += size;
    return { buffer, offset: 0, length, shape };
  }
}