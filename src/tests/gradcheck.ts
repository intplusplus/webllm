import type { GpuContext } from '../gpu/device';
import { makeRng } from '../reference/cpu-ref';
import { initWeights } from '../model/init';
import { Trainer } from '../train/trainer';

/**
 * 梯度检验：中心差分数值梯度 vs 反向传播解析梯度。
 *
 * 小配置 + 固定数据，覆盖：共享权重的 wte（lm_head + embedding 两条路径）、
 * 各 Linear 的 w/b、LayerNorm 的 w/b、以及 bias 的 sum_rows 归约。
 */
export async function testGradientCheck(gpu: GpuContext): Promise<string> {
  const config = { vocabSize: 11, blockSize: 6, nLayer: 2, nHead: 2, nEmbd: 8, bias: true };
  const weights = initWeights(config, 20241005);
  const trainer = new Trainer(gpu, config, weights, 4);

  const B = 2;
  const T = 4;
  const M = B * T;
  const rng = makeRng(777);
  const tokens = new Uint32Array(M);
  const targets = new Uint32Array(M);
  for (let i = 0; i < M; i++) {
    tokens[i] = Math.floor(rng() * config.vocabSize);
    targets[i] = Math.floor(rng() * config.vocabSize);
  }

  // 解析梯度（一次前向 + 一次反向）
  trainer.forward(tokens, B, T);
  const loss0 = await trainer.loss(targets);
  trainer.backward(targets);

  const names = [
    'wte',
    'wpe',
    'L0.ln1W',
    'L0.ln1B',
    'L0.wq.w',
    'L0.mlpProj.w',
    'L1.fc.w',
    'L1.ln2B',
    'lmHead.b',
  ];
  const eps = 1e-3;
  const relTol = 2e-2;
  const absTol = 1e-3;

  const failures: string[] = [];
  let worst = 0;
  let checks = 0;

  for (const name of names) {
    const param = await trainer.readParam(name);
    const grad = await trainer.readGrad(name);
    const n = param.length;
    const idxs = n >= 3 ? [0, Math.floor(n / 2), n - 1] : [0];

    for (const idx of idxs) {
      const orig = param[idx];

      param[idx] = orig + eps;
      trainer.writeParam(name, param);
      trainer.forward(tokens, B, T);
      const lp = await trainer.loss(targets);

      param[idx] = orig - eps;
      trainer.writeParam(name, param);
      trainer.forward(tokens, B, T);
      const lm = await trainer.loss(targets);

      param[idx] = orig;
      trainer.writeParam(name, param);

      const num = (lp - lm) / (2 * eps);
      const ana = grad[idx];
      const denom = Math.max(Math.abs(num) + Math.abs(ana), 1e-8);
      const rel = Math.abs(num - ana) / denom;
      const abs = Math.abs(num - ana);
      if (rel > worst) worst = rel;
      checks++;
      if (rel > relTol && abs > absTol) {
        failures.push(`${name}[${idx}] num=${num.toExponential(3)} ana=${ana.toExponential(3)} rel=${rel.toExponential(2)}`);
      }
    }
  }

  if (failures.length > 0) {
    throw new Error(`梯度不符（${failures.length}/${checks}）：${failures.join(' | ')}`);
  }
  return `loss=${loss0.toFixed(4)}，${names.length} 个参数 × ${checks / names.length} 点位，最差相对误差=${worst.toExponential(2)}`;
}
