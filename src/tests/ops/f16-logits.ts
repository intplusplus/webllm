import type { GpuContext } from '../../gpu/device';
import { checkTolerance, makeRng } from '../../reference/cpu-ref';
import { DEFAULT_CONFIG } from '../../model/config';
import { initWeights } from '../../model/init';
import { TinyGpt } from '../../model/tiny-gpt';

/**
 * M3：同一份权重下，f16 GEMM 推理 logits 与 fp32 推理 logits 回归。
 * 这不是为了完全一致，而是确认 f16 输入路径没有结构性错位（布局/bias/scale）。
 */
export async function testGptForwardF16(gpu: GpuContext): Promise<string> {
  if (!gpu.features.has('shader-f16')) {
    return 'shader-f16 不可用，跳过 f16 logits 对拍';
  }

  const config = DEFAULT_CONFIG;
  const weights = initWeights(config, 20241005);
  const B = 2;
  const T = 8;
  const M = B * T;
  const rng = makeRng(999);
  const tokens = new Uint32Array(M);
  for (let i = 0; i < M; i++) tokens[i] = Math.floor(rng() * config.vocabSize);

  const fp32 = new TinyGpt(gpu, config, weights, 4, false);
  const f16 = new TinyGpt(gpu, config, weights, 4, true);

  const a = await fp32.readTensor(fp32.forward(tokens, B, T));
  const b = await f16.readTensor(f16.forward(tokens, B, T));

  const r = checkTolerance(b, a, 2e-3, 2e-2);
  if (!r.ok) {
    throw new Error(
      `f16 logits 回归失败：maxAbs=${r.maxAbs.toExponential(3)} maxRel=${r.maxRel.toExponential(3)}`,
    );
  }
  return `B=${B} T=${T} vocab=${config.vocabSize}，maxAbs=${r.maxAbs.toExponential(2)} maxRel=${r.maxRel.toExponential(2)}`;
}
