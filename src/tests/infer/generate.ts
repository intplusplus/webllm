import type { GpuContext } from '../../gpu/device';
import { makeRng } from '../../reference/cpu-ref';
import { initWeights } from '../../model/init';
import { generateTokens } from '../../infer/generate';

/** 生成器冒烟测试：验证自回归循环、logits 读取、sampler 和输出 id 合法性。 */
export async function testGenerate(gpu: GpuContext): Promise<string> {
  const config = { vocabSize: 13, blockSize: 16, nLayer: 1, nHead: 2, nEmbd: 16, bias: true };
  const weights = initWeights(config, 1234);
  const prompt = Uint32Array.from([1, 2, 3]);
  const maxNew = 12;

  const result = await generateTokens(gpu, config, weights, prompt, maxNew, {
    temperature: 0.8,
    topK: 5,
    topP: 0.9,
    rng: makeRng(20251005),
  });

  if (result.ids.length !== prompt.length + maxNew) {
    throw new Error(`生成数量错误：${result.ids.length} != ${prompt.length + maxNew}`);
  }
  if (result.generated !== maxNew || result.promptLength !== prompt.length) {
    throw new Error('生成元数据错误');
  }
  for (const id of result.ids) {
    if (id >= config.vocabSize) throw new Error(`生成非法 token id：${id}`);
  }

  return `prompt=3 new=${maxNew} temp=0.8 topK=5 topP=0.9，${result.tokensPerSecond.toFixed(1)} tokens/s，ids=${Array.from(result.ids).join(',')}`;
}
