import type { GpuContext } from './gpu/device';
import { allTests, type TestResult } from './tests';

export type { TestResult };

/**
 * 依次运行全部自检。每个测试用 validation error scope 包裹，
 * 这样即使 WGSL 编译 / 绑定校验失败也能拿到明确错误信息。
 */
export async function runSelfTest(
  gpu: GpuContext,
  onResult: (result: TestResult) => void,
): Promise<void> {
  for (const test of allTests) {
    gpu.device.pushErrorScope('validation');
    let pass = false;
    let detail = '';
    try {
      detail = await test.run(gpu);
      pass = true;
    } catch (err) {
      detail = (err as Error).message;
    }
    const scopeError = await gpu.device.popErrorScope();
    if (scopeError) {
      pass = false;
      detail = `[validation] ${scopeError.message}${detail ? ` | ${detail}` : ''}`;
    }
    onResult({ name: test.name, pass, detail });
  }
}