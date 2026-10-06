/** 采样器：temperature / top-k / top-p。纯 CPU 侧作用在 logits 上。 */

export interface SamplerOptions {
  /** 温度；<=0 表示贪心解码。 */
  temperature?: number;
  /** 仅保留概率最高的 k 个 token；<=0 表示不限制。 */
  topK?: number;
  /** 核采样累计概率；>=1 表示不限制。 */
  topP?: number;
  /** 可复现随机源；默认 Math.random。 */
  rng?: () => number;
}

/**
 * 从 logits[offset, offset+vocabSize) 中采样一个 token id。
 * top-k 与 top-p 均为「先按概率降序，再截断」的标准做法。
 */
export function sampleToken(
  logits: Float32Array,
  offset: number,
  vocabSize: number,
  options: SamplerOptions = {},
): number {
  if (vocabSize <= 0) throw new Error('sampleToken: vocabSize 必须为正');
  if (offset < 0 || offset + vocabSize > logits.length) {
    throw new Error(`sampleToken: logits 越界 offset=${offset} vocab=${vocabSize}`);
  }

  const temperature = options.temperature ?? 1;
  if (!(temperature > 0)) {
    let best = offset;
    for (let i = 1; i < vocabSize; i++) {
      if (logits[offset + i] > logits[best]) best = offset + i;
    }
    return best - offset;
  }

  const scaled = new Float32Array(vocabSize);
  let max = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    scaled[i] = logits[offset + i] / temperature;
    if (scaled[i] > max) max = scaled[i];
  }
  const weights = new Float32Array(vocabSize);
  let sum = 0;
  for (let i = 0; i < vocabSize; i++) {
    weights[i] = Math.exp(scaled[i] - max);
    sum += weights[i];
  }

  const order: number[] = [];
  for (let i = 0; i < vocabSize; i++) order.push(i);
  order.sort((a, b) => weights[b] - weights[a]);

  const topK = options.topK ?? 0;
  const limit = topK > 0 ? Math.min(vocabSize, Math.max(1, Math.floor(topK))) : vocabSize;
  const kept = order.slice(0, limit);

  const topP = options.topP ?? 1;
  const finalOrder = topP >= 1 ? kept : (() => {
    let cum = 0;
    for (let i = 0; i < kept.length; i++) {
      cum += weights[kept[i]] / sum;
      if (cum >= topP) return kept.slice(0, i + 1);
    }
    return kept;
  })();

  let keptSum = 0;
  for (const idx of finalOrder) keptSum += weights[idx];
  const r = (options.rng ?? Math.random)() * keptSum;

  let acc = 0;
  for (const idx of finalOrder) {
    acc += weights[idx];
    if (acc >= r) return idx;
  }
  return finalOrder[finalOrder.length - 1];
}
