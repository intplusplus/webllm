import type { GpuContext } from '../../gpu/device';
import { makeRng } from '../../reference/cpu-ref';
import { initWeights } from '../../model/init';
import { Trainer } from '../../train/trainer';
import { loadTinyShakespeare } from '../../train/data';
import { generateTokens } from '../../infer/generate';

interface Metric {
  step: number;
  train: number;
  val: number;
  ppl: number;
  grad: number;
  lr: number;
  tps: number;
  arena: number;
}


/**
 * 真实数据预训练监测：Tiny Shakespeare 字符级语言模型。
 *
 * 覆盖：真实语料拉取、train/val 切分、随机批次、warmup+cosine LR、
 * train EMA / val / PPL / 全局梯度 L2 / 吞吐 / arena 占用监测。
 */
export async function testPretrain(gpu: GpuContext): Promise<string> {
  const started = performance.now();
  const corpus = await loadTinyShakespeare();
  const downloadMs = performance.now() - started;

  const split = Math.floor(corpus.ids.length * 0.9);
  const trainData = corpus.ids.subarray(0, split);
  const valData = corpus.ids.subarray(split);
  if (trainData.length < 1024 || valData.length < 1024) {
    throw new Error(`数据切分过短：train=${trainData.length} val=${valData.length}`);
  }

  const B = 8;
  const T = 64;
  const M = B * T;
  const steps = 1200;
  const evalEvery = 100;
  const baseLr = 3e-3;
  const warmupSteps = 20;
  const valBatches = 4;

  const config = {
    vocabSize: corpus.vocab.length,
    blockSize: T,
    nLayer: 2,
    nHead: 4,
    nEmbd: 64,
    bias: true,
  };
  const trainer = new Trainer(gpu, config, initWeights(config, 4242), B);

  const trainRng = makeRng(20241005);
  const valRng = makeRng(1234);

  const makeBatch = (source: Uint32Array, rand: () => number) => {
    const tokens = new Uint32Array(M);
    const targets = new Uint32Array(M);
    const maxStart = source.length - T - 1;
    if (maxStart <= 0) throw new Error(`数据切分不足以采样：${source.length}`);
    for (let b = 0; b < B; b++) {
      const start = Math.floor(rand() * (maxStart + 1));
      for (let t = 0; t < T; t++) {
        tokens[b * T + t] = source[start + t];
        targets[b * T + t] = source[start + t + 1];
      }
    }
    return { tokens, targets };
  };

  const valSet = Array.from({ length: valBatches }, () => makeBatch(valData, valRng));

  const evalVal = async (): Promise<number> => {
    let sum = 0;
    for (const batch of valSet) {
      trainer.forward(batch.tokens, B, T);
      sum += await trainer.loss(batch.targets);
    }
    return sum / valBatches;
  };

  const lrAt = (step: number): number => {
    if (step < warmupSteps) return (baseLr * (step + 1)) / warmupSteps;
    const t = (step - warmupSteps) / Math.max(1, steps - warmupSteps);
    const cosine = 0.5 * (1 + Math.cos(Math.PI * t));
    return baseLr * (0.05 + 0.95 * cosine);
  };

  const curve: Metric[] = [];
  const start = performance.now();
  let tokens = 0;
  let ema = 0;

  const baseBatch = makeBatch(trainData, trainRng);
  trainer.forward(baseBatch.tokens, B, T);
  const train0 = await trainer.loss(baseBatch.targets);
  ema = train0;
  const val0 = await evalVal();
  curve.push({
    step: 0,
    train: train0,
    val: val0,
    ppl: Math.exp(val0),
    grad: 0,
    lr: 0,
    tps: 0,
    arena: trainer.arenaUsedBytes,
  });

  for (let s = 0; s < steps; s++) {
    const batch = makeBatch(trainData, trainRng);
    trainer.forward(batch.tokens, B, T);
    const l = await trainer.loss(batch.targets);
    ema = ema === 0 ? l : 0.9 * ema + 0.1 * l;

    const lr = lrAt(s);
    const at = (s + 1) % evalEvery === 0 || s === steps - 1;

    trainer.backward(batch.targets);

    if (!at) {
      trainer.step({ lr, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 });
      tokens += M;
      continue;
    }

    const grad = await trainer.gradNorm();
    trainer.step({ lr, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 });
    tokens += M;

    const val = await evalVal();
    const elapsed = Math.max(1, performance.now() - start);
    curve.push({
      step: s + 1,
      train: ema,
      val,
      ppl: Math.exp(val),
      grad,
      lr,
      tps: tokens / (elapsed / 1000),
      arena: trainer.arenaUsedBytes,
    });
  }

  const first = curve[0];
  const last = curve[curve.length - 1];
  const elapsed = Math.max(1, performance.now() - start);
  const avgTps = tokens / (elapsed / 1000);

  if (!(last.val < first.val * 0.98) || !(last.train < first.train * 0.95)) {
    const seen = curve.map((p) => `${p.step}:${p.train.toFixed(3)}/${p.val.toFixed(3)}`).join(' | ');
    throw new Error(
      `预训练未达预期：train ${first.train.toFixed(4)}→${last.train.toFixed(4)}，val ${first.val.toFixed(4)}→${last.val.toFixed(4)}；曲线 ${seen}`,
    );
  }

  const curveText = curve
    .map(
      (p) =>
        `${p.step}:t=${p.train.toFixed(3)}/v=${p.val.toFixed(3)}/ppl=${p.ppl.toFixed(1)}/g=${p.grad.toFixed(2)}/lr=${p.lr.toExponential(1)}/${p.tps.toFixed(0)}tps/arena=${(p.arena / 1024 / 1024).toFixed(1)}MiB`,
    )
    .join(' | ');

  // --- 训练后采样：导回权重，进入自回归生成，验证「训练 → 采样」闭环 ---
  const vocabIndex = new Map(corpus.vocab.map((ch, i) => [ch, i]));
  const promptText = 'ROMEO:';
  const promptIds = Uint32Array.from(Array.from(promptText, (ch) => vocabIndex.get(ch) ?? 0));
  const trainedWeights = await trainer.exportWeights();
  const generated = await generateTokens(gpu, config, trainedWeights, promptIds, 120, {
    temperature: 0.7,
    topK: 16,
    topP: 0.9,
    rng: makeRng(20251005),
  });
  const sampleText = Array.from(generated.ids.slice(promptIds.length), (id) => corpus.vocab[id] ?? '?')
    .join('')
    .replace(/[\r\n]+/g, ' / ');

  return [
    `真实数据预训练：Tiny Shakespeare chars=${corpus.ids.length} vocab=${corpus.vocab.length} train/val=${trainData.length}/${valData.length}，下载 ${(downloadMs / 1000).toFixed(2)}s`,
    `模型：params=${trainer.paramCount()} vocab=${config.vocabSize} L=${config.nLayer} H=${config.nHead} C=${config.nEmbd} B=${B} T=${T} steps=${steps}`,
    `训练：lr=${baseLr.toExponential(1)} warmup=${warmupSteps} cosine，tokens=${tokens} avg=${avgTps.toFixed(0)} tokens/s，train ${first.train.toFixed(4)}→${last.train.toFixed(4)}，val ${first.val.toFixed(4)}→${last.val.toFixed(4)}，PPL ${first.ppl.toFixed(1)}→${last.ppl.toFixed(1)}`,
    `曲线：${curveText}`,
    `采样：prompt="${promptText}" new=${generated.generated} temp=0.7 topK=16 topP=0.9 ${generated.tokensPerSecond.toFixed(1)} tokens/s，样本：${sampleText}`,
  ].join('；');
}
