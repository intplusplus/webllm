/**
 * CPU 参考实现（对拍基线）。
 *
 * 所有 GPU 算子都必须与这里的纯 JS 实现逐一对拍，容差 rel < 1e-5（fp32）。
 */

/** 逐元素加法参考实现。 */
export function vecAddRef(a: Float32Array, b: Float32Array): Float32Array {
  if (a.length !== b.length) throw new Error('vecAddRef: 长度不一致');
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

export interface ToleranceResult {
  ok: boolean;
  maxAbs: number;
  maxRel: number;
}

/**
 * 混合容差判定：逐元素要求 |got - ref| <= atol + rtol * |ref|。
 *
 * 单用相对误差会在输出接近 0 处产生假阳性（例如 gelu 负值尾部，
 * tanh 饱和导致 1+tanh 灾难性抵消），因此必须叠加绝对容差。
 */
export function checkTolerance(
  got: Float32Array,
  ref: Float32Array,
  atol: number,
  rtol: number,
): ToleranceResult {
  if (got.length !== ref.length) {
    throw new Error(`checkTolerance: 长度不一致 got=${got.length} ref=${ref.length}`);
  }
  let ok = true;
  let maxAbs = 0;
  let maxRel = 0;
  for (let i = 0; i < ref.length; i++) {
    const diff = Math.abs(got[i] - ref[i]);
    if (diff > atol + rtol * Math.abs(ref[i])) ok = false;
    if (diff > maxAbs) maxAbs = diff;
    const rel = diff / Math.max(1e-30, Math.abs(ref[i]));
    if (rel > maxRel) maxRel = rel;
  }
  return { ok, maxAbs, maxRel };
}

/** 生成 [-1, 1) 的随机 fp32 数组（可复现：传入固定 seed 的 rng）。 */
export function randomF32(length: number, rng: () => number = Math.random): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = rng() * 2 - 1;
  return out;
}

/** 可复现的 xorshift32 随机数生成器。 */
export function makeRng(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}