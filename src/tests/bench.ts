import type { GpuContext } from '../gpu/device';
import { initWeights } from '../model/init';
import { TinyGpt } from '../model/tiny-gpt';

/** M3 性能报告：fp32 / f16 GEMM 两条推理路径的前向 tokens/s 与 steps/s。 */
export async function testPerformanceBench(gpu: GpuContext): Promise<string> {
  const config = { vocabSize: 65, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 64, bias: true };
  const weights = initWeights(config, 4242);
  const B = 4;
  const T = 32;
  const M = B * T;
  const iterations = 50;

  const tokens = new Uint32Array(M);
  for (let i = 0; i < M; i++) tokens[i] = i % config.vocabSize;

  const bench = async (useF16: boolean): Promise<{ steps: number; tokens: number }> => {
    const model = new TinyGpt(gpu, config, weights, B, useF16);
    for (let i = 0; i < 5; i++) model.forward(tokens, B, T);

    const started = performance.now();
    for (let i = 0; i < iterations; i++) model.forward(tokens, B, T);
    await gpu.device.queue.onSubmittedWorkDone();
    const elapsed = Math.max(1e-6, (performance.now() - started) / 1000);

    return { steps: iterations / elapsed, tokens: (iterations * M) / elapsed };
  };

  const fp32 = await bench(false);
  const parts = [
    `tile=64x64x32 wg=16x16 fused=add_layernorm B=${B} T=${T} iters=${iterations}`,
    `fp32 ${fp32.tokens.toFixed(0)} tokens/s，${fp32.steps.toFixed(1)} steps/s`,
  ];

  if (gpu.hasF16) {
    const f16 = await bench(true);
    const speedup = f16.tokens / fp32.tokens;
    parts.push(
      `f16 ${f16.tokens.toFixed(0)} tokens/s，${f16.steps.toFixed(1)} steps/s，speedup ${speedup.toFixed(2)}x`,
    );
  }

  return parts.join('；');
}
