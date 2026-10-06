import type { GpuContext } from '../gpu/device';
import { readBatchProfile } from '../gpu/pipeline';
import { loadQwenWeights } from '../weights/qwen-loader';
import { QwenGpt } from '../model/qwen';
import { QWEN25_05B } from '../model/qwen-config';
import { BpeTokenizer } from '../tokenizer/bpe';
import { buildQwenChatText } from '../infer/qwen-generate';

/**
 * M5 诊断：解码一步的逐 pass 耗时剖析（timestamp-query）。
 *
 * 背景：解码 10.2 t/s ≈ 92ms/步，而每步 942MiB 权重在 20.2 GB/s 实测带宽下的
 * 下限约 47ms —— 剩下的 ~45ms 去向不明。本测试给 decodeStep 的每个 compute pass
 * 打一对时间戳，把 wall time 精确分解为：
 *   - GEMV 权重流式（q/k/v/o、gate/up、down、lm_head）—— 对照带宽下限
 *   - norm / rope / kvstore / attn / add / silu 等非权重 kernel
 *   - pass 总和与 wall time 的差值 = GPU 空隙（CPU 提交 + 屏障 + 调度）
 *
 * pass 布局是确定性的：embed + 24 层 × 12 pass + final norm + lm_head = 291。
 */
export async function testDecodeProfile(gpu: GpuContext): Promise<string> {
  if (!gpu.features.has('timestamp-query')) return 'timestamp-query 不可用，跳过解码剖析';
  const config = { ...QWEN25_05B, blockSize: 128 };
  const tokenizer = await BpeTokenizer.load('/models/qwen2.5-0.5b-int4/tokenizer.json');

  const model = new QwenGpt(gpu, config, 1);
  await loadQwenWeights(gpu, model, {});

  const promptIds = tokenizer.encode(buildQwenChatText('Hello'));
  model.resetCache();
  await model.prefill(Uint32Array.from(promptIds));

  // 预热 3 步（驱动编译 pipeline / 缓存池稳定），并记录未剖析时的步进耗时
  let token = 11;
  const warmMs: number[] = [];
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    const lg = model.decodeStep(token++);
    await model.readTensorSlice(lg, 0, 1);
    warmMs.push(performance.now() - t0);
  }

  // --- 剖析一步 ---
  const profile = model.profileNextForward(512);
  const t0 = performance.now();
  const logits = model.decodeStep(token++);
  await model.readTensorSlice(logits, 0, 1);
  const wallMs = performance.now() - t0;
  const passNs = await readBatchProfile(gpu.device, profile);
  profile.querySet.destroy();
  profile.resolveBuffer.destroy();
  profile.readBuffer.destroy();

  const L = config.nLayer;
  const ns2ms = (ns: bigint) => Number(ns) / 1e6;
  let gpuSumMs = 0;
  for (const ns of passNs) gpuSumMs += ns2ms(ns);

  // pass 索引 → 类别（布局见 runForward：embed + 24×12 + finalNorm + lmHead = 291）
  const cls = (idx: number): string => {
    if (idx === 0) return 'embed';
    if (idx === 1 + L * 12) return 'norm(final)';
    if (idx === 2 + L * 12) return 'lm_head';
    const inLayer = (idx - 1) % 12;
    const layer = Math.floor((idx - 1) / 12);
    if (layer < 0 || layer >= L) return `?${idx}`;
    return [
      'norm1', 'gemv q/k/v', 'rope', 'kvstore', 'attn', 'gemv o',
      'add', 'norm2', 'gemv gate/up', 'silu', 'gemv down', 'add2',
    ][inLayer];
  };

  const byClass = new Map<string, { total: number; count: number; max: number }>();
  passNs.forEach((ns, idx) => {
    const name = cls(idx);
    const e = byClass.get(name) ?? { total: 0, count: 0, max: 0 };
    e.total += ns2ms(ns);
    e.count++;
    e.max = Math.max(e.max, ns2ms(ns));
    byClass.set(name, e);
  });

  // GEMV 权重流式类（可对照 942MiB / 20.2 GB/s ≈ 47ms 的带宽下限）
  const gemvClasses = ['gemv q/k/v', 'gemv o', 'gemv gate/up', 'gemv down', 'lm_head'];
  let gemvMs = 0;
  for (const name of gemvClasses) gemvMs += byClass.get(name)?.total ?? 0;

  const lines = [...byClass.entries()]
    .sort((a, b) => b[1].total - a[1].total)
    .map(([name, e]) => `${name}: 总 ${e.total.toFixed(2)}ms × ${e.count}（最大单 pass ${e.max.toFixed(2)}ms）`);
  const gapMs = wallMs - gpuSumMs;
  // 942MiB ≈ 987.8MB；GEMV 合计时间下的等效带宽，对照单 kernel bench 的 20.2 GB/s 峰值
  const weightMB = 942 * 1.048576;
  const gemvBW = weightMB / (gemvMs / 1000) / 1000;
  const floorMs = (weightMB / 20.2);

  return [
    `wall=${wallMs.toFixed(1)}ms/步（预热 ${warmMs.map((m) => m.toFixed(0)).join('/')}ms），pass=${passNs.length}，GPU 累计 ${gpuSumMs.toFixed(1)}ms（${((gpuSumMs / wallMs) * 100).toFixed(0)}%），空隙 ${gapMs.toFixed(1)}ms（${((gapMs / wallMs) * 100).toFixed(0)}%）`,
    `GEMV 权重流式合计 ${gemvMs.toFixed(1)}ms —— 等效带宽 ${gemvBW.toFixed(1)} GB/s（峰值实测 20.2，带宽下限 ${floorMs.toFixed(0)}ms）`,
    ...lines,
  ].join('；');
}
