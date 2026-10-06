import type { GpuContext } from '../../gpu/device';
import { f16BitsToF32 } from '../../weights/safetensors';
import { loadQwenWeights, QWEN_LOCAL_DIR } from '../../weights/qwen-loader';
import { QwenGpt } from '../../model/qwen';
import { QWEN25_05B } from '../../model/qwen-config';
import { BpeTokenizer } from '../../tokenizer/bpe';
import {
    buildQwenChatText,
    generateQwen,
    QWEN_ENDOFTEXT,
    QWEN_IM_END,
} from '../../infer/qwen-generate';

/** f32 → 最近偶数 f16 → f32，与 buffer.ts 的 writeF16 舍入方式一致。 */
function roundF16(x: number): number {
  const fn = (Math as unknown as { f16round?: (v: number) => number }).f16round;
  if (!fn) throw new Error('本环境缺少 Math.f16round，无法校验 f16 上传');
  return fn(x);
}

/** 把 f16 buffer 的原始字节解码成 f32。 */
function decodeF16(buf: ArrayBuffer): Float32Array {
  const view = new DataView(buf);
  const out = new Float32Array(buf.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = f16BitsToF32(view.getUint16(i * 2, true));
  return out;
}

/**
 * 校验一个投影：GPU 上的 f16 权重必须等于「CPU 反量化值按 f16 舍入」。
 * 这直接验证 int4 反量化 + 布局 + f16 上传三步都正确；布局错会带来 O(1) 级偏差。
 */
async function checkWeight(
  model: QwenGpt,
  name: string,
  cpu: Float32Array,
  label: string,
): Promise<{ line: string; ok: boolean }> {
  const n = Math.min(cpu.length, 4096);
  const gpu = decodeF16(await model.readWeightF16Bytes(name, 0, n));
  let maxAbs = 0;
  let maxRel = 0;
  let exact = 0;
  for (let i = 0; i < n; i++) {
    const want = roundF16(cpu[i]);
    const diff = Math.abs(gpu[i] - want);
    if (diff === 0) exact++;
    if (diff > maxAbs) maxAbs = diff;
    const rel = diff / Math.max(1e-30, Math.abs(want));
    if (rel > maxRel) maxRel = rel;
  }
  const ok = maxAbs < 1e-6;
  return {
    line: `${label}[0..${n}) exact=${exact}/${n} maxAbs=${maxAbs.toExponential(2)} maxRel=${maxRel.toExponential(2)}${ok ? '' : ' FAIL'}`,
    ok,
  };
}

/** 判断生成的文本是否「格式正常」：无替换字符、字母占比合理。 */
function looksLikeText(text: string): boolean {
  if (text.length === 0) return false;
  if (text.includes('\uFFFD')) return false;
  let good = 0;
  for (const ch of text) {
    if (/[\p{L}\p{N}\s.,!?'"():;\-]/u.test(ch)) good++;
  }
  return good / text.length > 0.9;
}

/**
 * M4-4：加载真实 Qwen2.5-0.5B-Instruct-GPTQ-Int4 权重并做推理。
 *
 * 覆盖三件事：
 *   1. 加载器正确性：GPU 上的 f16 权重 == CPU 反量化值按 f16 舍入（逐元素，误差必须为 0）
 *   2. 真实权重下的贪心生成，输出可读文本
 *   3. 报告权重显存占用与生成吞吐
 */
export async function testQwenInfer(gpu: GpuContext): Promise<string> {
  const config = { ...QWEN25_05B, blockSize: 128 };
  const tokenizer = await BpeTokenizer.load(`${QWEN_LOCAL_DIR}tokenizer.json`);

  const model = new QwenGpt(gpu, config, 1);
  const loadStarted = performance.now();
  // 保留第 0 层的 fp32 权重与 embed 的前 64 行，用于校验加载正确性
  const loaded = await loadQwenWeights(gpu, model, {
    keepLayer: 0,
    keepEmbedRows: { start: 0, rows: 64 },
  });
  const loadMs = performance.now() - loadStarted;
  if (!loaded.layer) throw new Error('未保留第 0 层权重，无法校验');

  // --- 1) 加载器校验：量化投影 + f16 bias + embed 行 ---
  const checks = await Promise.all([
    checkWeight(model, 'L0.gateW', loaded.layer.gateW, 'gateW'),
    checkWeight(model, 'L0.upW', loaded.layer.upW, 'upW'),
    checkWeight(model, 'L0.downW', loaded.layer.downW, 'downW'),
    checkWeight(model, 'L0.qW', loaded.layer.qW, 'qW'),
    checkWeight(model, 'L0.kW', loaded.layer.kW, 'kW'),
    checkWeight(model, 'L0.oW', loaded.layer.oW, 'oW'),
    checkWeight(model, 'embed', loaded.embedRows!, 'embed(H)'),
  ]);
  const failed = checks.filter((c) => !c.ok);
  if (failed.length > 0) {
    throw new Error(`权重加载校验失败：${failed.map((f) => f.line).join(' | ')}`);
  }

  // --- 2) 真实权重贪心生成（KV cache 增量解码） ---
  const question = 'What is the capital of France? Answer with only the city name.';
  const promptIds = tokenizer.encode(buildQwenChatText(question));
  const gen = await generateQwen(model, config, promptIds, 32, {
    temperature: 0,
    stopTokens: [QWEN_IM_END, QWEN_ENDOFTEXT],
  });
  const answer = tokenizer.decode(gen.ids.slice(gen.promptLength));
  if (!looksLikeText(answer)) {
    throw new Error(`生成的文本格式异常：${JSON.stringify(answer)}`);
  }

  // 对照：无 cache 时每步要对整段上下文做一次全量前向。用一次带同步回读的前向计时作为参照。
  const ctxTokens = Uint32Array.from(gen.ids);
  const uncachedStart = performance.now();
  const uncachedLogits = model.forward(ctxTokens, 1, ctxTokens.length);
  await model.readTensorSlice(uncachedLogits, (ctxTokens.length - 1) * config.vocabSize, 1);
  const uncachedMs = performance.now() - uncachedStart;
  model.resetCache();

  const weightMiB = loaded.gpuBytes / 1024 / 1024;
  const fileMiB = loaded.fileBytes / 1024 / 1024;
  const cacheKiB = model.cacheBytes / 1024;
  return [
    `加载：${fileMiB.toFixed(0)}MiB safetensors → ${loaded.tensorCount} 个 tensor → GPU f16 ${weightMiB.toFixed(0)}MiB，耗时 ${(loadMs / 1000).toFixed(1)}s`,
    `加载校验（GPU f16 vs CPU 反量化后 f16 舍入）：${checks.map((c) => c.line).join(' | ')}`,
    `问答：Q="${question}" A=${JSON.stringify(answer)}（${gen.generated} tokens）`,
    `KV cache：prefill ${gen.prefillMs.toFixed(0)}ms（${promptIds.length} tokens），解码 ${gen.tokensPerSecond.toFixed(2)} tokens/s；无 cache 时每步全量前向需 ${uncachedMs.toFixed(0)}ms，cache 占用 ${cacheKiB.toFixed(0)}KiB`,
  ].join('；');
}
