import type { GpuContext } from '../gpu/device';
import { checkTolerance, makeRng } from '../reference/cpu-ref';
import { qwenForwardRef } from '../reference/qwen-ref';
import type { QwenConfig } from '../model/qwen-config';
import { QwenGpt, type QwenLayerWeights, type QwenWeights } from '../model/qwen';

/** 与真实 Qwen2.5 同构但极小：保留 GQA(4→2)、奇数 intermediate、24 万级词表外的长 dim。 */
const TEST_CONFIG: QwenConfig = {
  vocabSize: 97,
  blockSize: 16,
  nLayer: 2,
  nHead: 4,
  nKvHead: 2,
  nEmbd: 32,
  intermediate: 53,
  rmsEps: 1e-6,
  ropeTheta: 1000000,
};

/** f32 → 最近偶数 f16 → f32，保证 GPU 的 f16 权重存储无损。 */
function roundF16(x: number): number {
  const fn = (Math as unknown as { f16round?: (v: number) => number }).f16round;
  if (!fn) throw new Error('本环境缺少 Math.f16round，无法做 f16 精确对拍');
  return fn(x);
}

/** 生成按 f16 舍入过的随机矩阵（[-scale, scale)）。 */
function randMatrix(length: number, rng: () => number, scale: number): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = roundF16((rng() * 2 - 1) * scale);
  return out;
}

/** 生成一组小随机 Qwen 权重。 */
function randomQwenWeights(config: QwenConfig, seed: number): QwenWeights {
  const rng = makeRng(seed);
  const C = config.nEmbd;
  const KV = (config.nEmbd / config.nHead) * config.nKvHead;
  const I = config.intermediate;
  const s = 0.12;
  const norm = (n: number) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = roundF16(1 + (rng() - 0.5) * 0.2);
    return out;
  };

  const layers: QwenLayerWeights[] = [];
  for (let i = 0; i < config.nLayer; i++) {
    layers.push({
      inputNorm: norm(C),
      qW: randMatrix(C * C, rng, s),
      qB: randMatrix(C, rng, s),
      kW: randMatrix(KV * C, rng, s),
      kB: randMatrix(KV, rng, s),
      vW: randMatrix(KV * C, rng, s),
      vB: randMatrix(KV, rng, s),
      oW: randMatrix(C * C, rng, s),
      postNorm: norm(C),
      gateW: randMatrix(I * C, rng, s),
      upW: randMatrix(I * C, rng, s),
      downW: randMatrix(C * I, rng, s),
    });
  }

  return {
    embed: randMatrix(config.vocabSize * C, rng, s),
    finalNorm: norm(C),
    layers,
  };
}

/**
 * M4-3：Qwen2 架构端到端对拍。
 *
 * 小随机权重（按 f16 舍入，使 GPU 的 f16 权重存储无损）+ 独立 CPU 参考实现，
 * 逐层 hidden 与最终 logits 全部对拍。GQA、half-split RoPE、SwiGLU、RMSNorm、
 * tied embedding 任一接错都会让这里出现 O(1) 级偏差。
 */
export async function testQwenArch(gpu: GpuContext): Promise<string> {
  const config = TEST_CONFIG;
  const B = 2;
  const T = 6;
  const M = B * T;
  const rng = makeRng(0xbeef);
  const tokens = new Uint32Array(M);
  for (let i = 0; i < M; i++) tokens[i] = Math.floor(rng() * config.vocabSize);

  const weights = randomQwenWeights(config, 20241006);
  const model = new QwenGpt(gpu, config, B);
  model.uploadWeights(weights);

  const expected = qwenForwardRef(weights, config, { tokens, B, T });
  model.forward(tokens, B, T);

  // 1) 逐层 hidden 对拍，定位偏差来自哪一层
  const lines: string[] = [];
  const C = config.nEmbd;
  let allOk = true;
  for (let i = 0; i < config.nLayer; i++) {
    const got = await model.readTensor(model.layerOutputs[i]);
    const r = checkTolerance(got, expected.layerOut[i], 2e-3, 2e-2);
    if (!r.ok) allOk = false;
    lines.push(`L${i} maxAbs=${r.maxAbs.toExponential(2)} maxRel=${r.maxRel.toExponential(2)}${r.ok ? '' : ' FAIL'}`);
  }

  // 2) 最终 logits 对拍
  const gotLogits = await model.readTensor(model.forward(tokens, B, T));
  const rLogits = checkTolerance(gotLogits, expected.logits, 2e-3, 2e-2);
  if (!rLogits.ok) allOk = false;
  lines.push(`logits maxAbs=${rLogits.maxAbs.toExponential(2)} maxRel=${rLogits.maxRel.toExponential(2)}${rLogits.ok ? '' : ' FAIL'}`);

  // 3) 每行 argmax 必须完全一致（预测语义层面的强校验）
  const V = config.vocabSize;
  const argmax = (arr: Float32Array, row: number) => {
    let best = 0;
    for (let v = 1; v < V; v++) if (arr[row * V + v] > arr[row * V + best]) best = v;
    return best;
  };
  const mismatched: number[] = [];
  for (let m = 0; m < M; m++) {
    if (argmax(gotLogits, m) !== argmax(expected.logits, m)) mismatched.push(m);
  }
  if (mismatched.length > 0) allOk = false;

  if (!allOk) {
    throw new Error(`Qwen2 架构对拍失败：${lines.join(' | ')}；argmax 不一致行=[${mismatched}]`);
  }

  return `config vocab=${config.vocabSize} L=${config.nLayer} nHead=${config.nHead} nKvHead=${config.nKvHead} C=${C} inter=${config.intermediate} ropeTheta=1e6 B=${B} T=${T}；${lines.join(' | ')}；argmax ${M}/${M} 行一致；权重显存 ${(model.weightBytes / 1024).toFixed(0)} KiB`;
}

/**
 * M5：KV cache 增量解码的正确性。
 *
 * 同一组权重下，把「prefill 整段 prompt + 逐 token decode」的结果与
 * 「无 cache 的完整前向」逐元素对拍。两条路径必须一致，否则 cache 的写入位置、
 * RoPE 的绝对位置或注意力的可见范围有误。
 */
export async function testQwenKvCache(gpu: GpuContext): Promise<string> {
  const config = TEST_CONFIG;
  const T = 6;
  const V = config.vocabSize;
  const rng = makeRng(0xca11);
  const tokens = new Uint32Array(T);
  for (let i = 0; i < T; i++) tokens[i] = Math.floor(rng() * V);

  const weights = randomQwenWeights(config, 20241006);
  const model = new QwenGpt(gpu, config, 1);
  model.uploadWeights(weights);

  // 基线 1：整段 prompt 的无 cache 前向
  const base = await model.readTensor(model.forward(tokens, 1, T));

  // cache 路径：prefill 整段
  model.resetCache();
  const pre = await model.readTensor(model.prefill(tokens));
  const r1 = checkTolerance(pre, base, 1e-4, 1e-3);
  if (model.cacheLength !== T) throw new Error(`prefill 后 cacheLength=${model.cacheLength}，期望 ${T}`);

  // 基线 2：prompt + 1 个新 token 的无 cache 前向，取最后一行的 logits
  const nextTok = Math.floor(rng() * V);
  const extended = new Uint32Array(T + 1);
  extended.set(tokens);
  extended[T] = nextTok;
  const baseFull = await model.readTensor(model.forward(extended, 1, T + 1));
  const baseLast = baseFull.slice(T * V, (T + 1) * V);

  // cache 路径：decode 一步
  //
  // 注意容差比 prefill 松：解码走 GEMV kernel（激活保持 f32），而 M>1 的通用 GEMM 会把
  // 激活舍入成 f16 再参与乘法。两条路径的权重完全相同，差异只来自这一处 f16 激活舍入，
  // 量级约 1e-3。真正接错（位置/可见范围/写入偏移）会产生 O(1) 级偏差，仍会被拦住。
  const dec = await model.readTensor(model.decodeStep(nextTok));
  const r2 = checkTolerance(dec, baseLast, 2e-3, 2e-2);

  // 语义层面的强校验：两条路径的 argmax 必须一致
  const argmaxOf = (arr: Float32Array) => {
    let best = 0;
    for (let v = 1; v < V; v++) if (arr[v] > arr[best]) best = v;
    return best;
  };
  const argmaxSame = argmaxOf(dec) === argmaxOf(baseLast);

  // 超出上下文必须报错，且 resetCache 后可重新预填充
  let overflowRejected = false;
  try {
    const huge = new Uint32Array(config.blockSize + 1);
    model.prefill(huge);
  } catch {
    overflowRejected = true;
  }
  model.resetCache();
  if (model.cacheLength > 0) throw new Error(`resetCache 未清空：cacheLength=${model.cacheLength}`);

  const ok = r1.ok && r2.ok && overflowRejected && argmaxSame;
  if (!ok) {
    throw new Error(
      `KV cache 对拍失败：prefill ${r1.ok ? 'OK' : `maxAbs=${r1.maxAbs.toExponential(2)} maxRel=${r1.maxRel.toExponential(2)}`} | ` +
        `decode ${r2.ok ? 'OK' : `maxAbs=${r2.maxAbs.toExponential(2)} maxRel=${r2.maxRel.toExponential(2)}`} | ` +
        `argmax 一致=${argmaxSame} | 超长拒绝=${overflowRejected}`,
    );
  }

  return [
    `config L=${config.nLayer} C=${config.nEmbd} nHead=${config.nHead} nKvHead=${config.nKvHead} T=${T}`,
    `prefill 整段 vs 无 cache 全量前向：maxAbs=${r1.maxAbs.toExponential(2)} maxRel=${r1.maxRel.toExponential(2)}`,
    `decode 1 步 vs 无 cache 的 T+1 全量前向：maxAbs=${r2.maxAbs.toExponential(2)} maxRel=${r2.maxRel.toExponential(2)}，argmax 一致=${argmaxSame}`,
    `cache ${model.cacheBytes / 1024} KiB（${config.blockSize} 长度上限）；超长上下文已拒绝`,
  ].join('；');
}
