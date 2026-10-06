import type { GpuContext } from '../gpu/device';
import type { GPTConfig } from '../model/config';
import type { GPTWeights } from '../model/init';
import { TinyGpt } from '../model/tiny-gpt';
import { sampleToken, type SamplerOptions } from './sampler';

export interface GenerateOptions extends SamplerOptions {}

export interface GenerateResult {
  /** prompt + 生成 token 的完整 id 序列 */
  ids: Uint32Array;
  promptLength: number;
  generated: number;
  tokensPerSecond: number;
}

/**
 * 无 KV cache 的自回归生成：每步用当前上下文跑一次完整前向，取最后一行 logits 采样。
 * 超过 blockSize 时滑动窗口取最后 blockSize 个 token。
 */
export async function generateTokens(
  gpu: GpuContext,
  config: GPTConfig,
  weights: GPTWeights,
  prompt: Uint32Array,
  maxNewTokens: number,
  options: GenerateOptions = {},
): Promise<GenerateResult> {
  if (prompt.length === 0) throw new Error('generateTokens: prompt 不能为空');
  if (prompt.length > config.blockSize) {
    throw new Error(`generateTokens: prompt 长度 ${prompt.length} 超过 blockSize=${config.blockSize}`);
  }
  if (maxNewTokens <= 0) throw new Error('generateTokens: maxNewTokens 必须为正');

  const model = new TinyGpt(gpu, config, weights, 1);
  const context: number[] = Array.from(prompt);
  const started = performance.now();

  for (let i = 0; i < maxNewTokens; i++) {
    const from = Math.max(0, context.length - config.blockSize);
    const window = Uint32Array.from(context.slice(from));
    const logits = model.forward(window, 1, window.length);
    const logitsHost = await model.readTensor(logits);
    const next = sampleToken(
      logitsHost,
      (window.length - 1) * config.vocabSize,
      config.vocabSize,
      options,
    );
    context.push(next);
  }

  const elapsed = Math.max(1e-6, (performance.now() - started) / 1000);
  return {
    ids: Uint32Array.from(context),
    promptLength: prompt.length,
    generated: maxNewTokens,
    tokensPerSecond: maxNewTokens / elapsed,
  };
}
