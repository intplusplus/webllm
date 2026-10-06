import { makeRng } from '../reference/cpu-ref';
import type { GPTConfig } from './config';

export interface LinearWeights {
  /** [out, in] */
  w: Float32Array;
  b: Float32Array | null;
}

export interface LayerWeights {
  ln1W: Float32Array;
  ln1B: Float32Array;
  ln2W: Float32Array;
  ln2B: Float32Array;
  wq: LinearWeights;
  wk: LinearWeights;
  wv: LinearWeights;
  attnProj: LinearWeights;
  fc: LinearWeights;
  mlpProj: LinearWeights;
}

export interface GPTWeights {
  /** [vocabSize, nEmbd] */
  wte: Float32Array;
  /** [blockSize, nEmbd] */
  wpe: Float32Array;
  layers: LayerWeights[];
  lnFW: Float32Array;
  lnFB: Float32Array;
  /** 与 wte 共享（tie_word_embeddings） */
  lmHead: LinearWeights;
}

/** Box-Muller 正态分布。 */
function makeNormal(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    const angle = 2 * Math.PI * v;
    spare = r * Math.sin(angle);
    return r * Math.cos(angle);
  };
}

/** 用固定 seed 初始化一份可复现的权重（std = 0.02，与 GPT-2 一致）。 */
export function initWeights(config: GPTConfig, seed: number): GPTWeights {
  const norm = makeNormal(makeRng(seed));
  const fill = (n: number): Float32Array => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = norm() * 0.02;
    return out;
  };
  const C = config.nEmbd;
  const linear = (out: number, inp: number): LinearWeights => ({
    w: fill(out * inp),
    b: config.bias ? fill(out) : null,
  });

  const wte = fill(config.vocabSize * C);
  const layers: LayerWeights[] = [];
  for (let i = 0; i < config.nLayer; i++) {
    layers.push({
      ln1W: fill(C),
      ln1B: fill(C),
      ln2W: fill(C),
      ln2B: fill(C),
      wq: linear(C, C),
      wk: linear(C, C),
      wv: linear(C, C),
      attnProj: linear(C, C),
      fc: linear(4 * C, C),
      mlpProj: linear(C, 4 * C),
    });
  }

  return {
    wte,
    wpe: fill(config.blockSize * C),
    layers,
    lnFW: fill(C),
    lnFB: fill(C),
    // tie_word_embeddings：lm_head 直接复用 wte
    lmHead: { w: wte, b: config.bias ? fill(config.vocabSize) : null },
  };
}