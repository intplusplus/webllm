import type { GPTConfig } from '../model/config';
import { headDim } from '../model/config';
import type { GPTWeights } from '../model/init';
import {
  attentionRef,
  embeddingRef,
  geluRef,
  gemmNTRef,
  layernormRef,
  ropeRef,
} from './ops';

export interface ForwardInput {
  /** 长度 B*T，行主序 [B, T] */
  tokens: Uint32Array;
  B: number;
  T: number;
}

export interface ForwardOutput {
  /** [B*T, vocabSize] */
  logits: Float32Array;
  M: number;
}

/**
 * 完整 GPT 前向的 CPU 参考实现（nanoGPT / GPT-2 结构）。
 * 与 GPU 实现走完全不同的代码路径，用于 logits 对拍。
 */
export function gptForwardRef(
  weights: GPTWeights,
  config: GPTConfig,
  input: ForwardInput,
  ropeBase = 10000,
  eps = 1e-5,
): ForwardOutput {
  const { B, T } = input;
  const M = B * T;
  const C = config.nEmbd;
  const D = headDim(config);
  const V = config.vocabSize;

  const tokEmb = embeddingRef(weights.wte, C, input.tokens);
  const posTokens = new Uint32Array(M);
  for (let i = 0; i < M; i++) posTokens[i] = i % T;
  const posEmb = embeddingRef(weights.wpe, C, posTokens);

  let x = new Float32Array(M * C);
  for (let i = 0; i < M * C; i++) x[i] = tokEmb[i] + posEmb[i];

  for (const L of weights.layers) {
    // --- attention ---
    const h = layernormRef(x, M, C, L.ln1W, L.ln1B, eps, config.bias);
    const q = gemmNTRef(h, L.wq.w, M, C, C, L.wq.b ?? undefined);
    const k = gemmNTRef(h, L.wk.w, M, C, C, L.wk.b ?? undefined);
    const v = gemmNTRef(h, L.wv.w, M, C, C, L.wv.b ?? undefined);
    const qr = ropeRef(q, M, config.nHead, D, T, ropeBase);
    const kr = ropeRef(k, M, config.nHead, D, T, ropeBase);
    const att = attentionRef(qr, kr, v, B, T, config.nHead, D, true);
    const proj = gemmNTRef(att, L.attnProj.w, M, C, C, L.attnProj.b ?? undefined);
    for (let i = 0; i < M * C; i++) x[i] += proj[i];

    // --- mlp ---
    const h2 = layernormRef(x, M, C, L.ln2W, L.ln2B, eps, config.bias);
    const fc = gemmNTRef(h2, L.fc.w, M, 4 * C, C, L.fc.b ?? undefined);
    const act = geluRef(fc);
    const mp = gemmNTRef(act, L.mlpProj.w, M, C, 4 * C, L.mlpProj.b ?? undefined);
    for (let i = 0; i < M * C; i++) x[i] += mp[i];
  }

  const lnf = layernormRef(x, M, C, weights.lnFW, weights.lnFB, eps, config.bias);
  const logits = gemmNTRef(lnf, weights.lmHead.w, M, V, C, weights.lmHead.b ?? undefined);
  return { logits, M };
}