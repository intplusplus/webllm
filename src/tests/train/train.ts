import type { GpuContext } from '../../gpu/device';
import { initWeights } from '../../model/init';
import { Trainer } from '../../train/trainer';

/**
 * 训练闭环验证：固定小语料 + AdamW，loss 应显著下降并趋近 0（过拟合）。
 *
 * 语料：next-token = (cur+1) mod vocab，位置/批次固定，模型只需记住这张表。
 */
export async function testTraining(gpu: GpuContext): Promise<string> {
  const config = { vocabSize: 12, blockSize: 8, nLayer: 2, nHead: 2, nEmbd: 16, bias: true };
  const weights = initWeights(config, 4242);
  const trainer = new Trainer(gpu, config, weights, 4);

  const B = 4;
  const T = 8;
  const M = B * T;
  const V = config.vocabSize;
  const tokens = new Uint32Array(M);
  const targets = new Uint32Array(M);
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const v = (b * 3 + t) % V;
      tokens[b * T + t] = v;
      targets[b * T + t] = (v + 1) % V;
    }
  }

  const opts = { lr: 0.02, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
  const steps = 300;
  let first = 0;
  let last = 0;
  const curve: { step: number; loss: number }[] = [];

  for (let s = 0; s < steps; s++) {
    trainer.forward(tokens, B, T);
    const l = await trainer.loss(targets);
    if (s === 0) first = l;
    last = l;
    if (s === 0 || (s + 1) % 50 === 0) curve.push({ step: s + 1, loss: l });
    trainer.backward(targets);
    trainer.step(opts);
  }

  // 最后一步参数更新后的 loss（上面 loop 里的 last 是最后一次更新前的 loss）。
  trainer.forward(tokens, B, T);
  last = await trainer.loss(targets);

  if (!(last < 0.1)) {
    const seen = curve.map((p) => `${p.step}:${p.loss.toExponential(2)}`).join(' -> ');
    throw new Error(`过拟合失败：${seen}，最终 loss ${last.toFixed(6)}（未降到 0.1 以下）`);
  }
  const seen = curve.map((p) => `${p.step}:${p.loss.toExponential(2)}`).join(' -> ');
  return `steps=${steps} lr=${opts.lr}，loss ${first.toFixed(4)} → ${last.toFixed(6)}；曲线 ${seen} -> final:${last.toExponential(2)}`;
}