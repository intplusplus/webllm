/**
 * Qwen2 前向的独立 CPU 参考实现。
 *
 * 刻意不复用 GPU 侧任何算子代码、也不复用 gpt-ref：参考实现的价值在于「另一条独立路径」。
 * 全部按 HF Qwen2 的定义直写，用于与 GPU 前向逐元素对拍。
 */
import type { QwenConfig } from '../model/qwen-config';
import { qwenHeadDim, qwenKvDim } from '../model/qwen-config';
import type { QwenWeights } from '../model/qwen';

/** y = x / sqrt(mean(x^2) + eps) * w，x 为 [rows, D]。 */
export function rmsNormRef(
    x: Float32Array,
    w: Float32Array,
    rows: number,
    D: number,
    eps: number,
): Float32Array {
    const out = new Float32Array(rows * D);
    for (let r = 0; r < rows; r++) {
        const base = r * D;
        let sum = 0;
        for (let d = 0; d < D; d++) sum += x[base + d] * x[base + d];
        const rstd = 1 / Math.sqrt(sum / D + eps);
        for (let d = 0; d < D; d++) out[base + d] = x[base + d] * rstd * w[d];
    }
    return out;
}

/**
 * Qwen2 的 half-split RoPE（对应 HF rotate_half）。
 * x 为 [rows, nHead, headDim]，pos = row % T + posOffset（posOffset 用于 KV cache 增量解码）。
 */
export function ropeHalfRef(
    x: Float32Array,
    rows: number,
    nHead: number,
    D: number,
    T: number,
    base: number,
    posOffset = 0,
): Float32Array {
    const half = D / 2;
    const out = new Float32Array(x.length);
    for (let r = 0; r < rows; r++) {
        const pos = (r % T) + posOffset;
        for (let h = 0; h < nHead; h++) {
            const o = (r * nHead + h) * D;
            for (let i = 0; i < half; i++) {
                const exponent = (2 * i) / D;
                const angle = pos * Math.pow(base, -exponent);
                const c = Math.cos(angle);
                const s = Math.sin(angle);
                const x0 = x[o + i];
                const x1 = x[o + i + half];
                out[o + i] = x0 * c - x1 * s;
                out[o + i + half] = x0 * s + x1 * c;
            }
        }
    }
    return out;
}

/** 因果 GQA 注意力。q [B*T, nHead, D]，k/v [B*T, nKvHead, D]。 */
export function attentionGqaRef(
    q: Float32Array,
    k: Float32Array,
    v: Float32Array,
    B: number,
    T: number,
    nHead: number,
    nKvHead: number,
    D: number,
): Float32Array {
    const scale = 1 / Math.sqrt(D);
    const group = nHead / nKvHead;
    const out = new Float32Array(B * T * nHead * D);
    for (let b = 0; b < B; b++) {
        for (let h = 0; h < nHead; h++) {
            const kvh = Math.floor(h / group);
            for (let t = 0; t < T; t++) {
                const qBase = ((b * T + t) * nHead + h) * D;
                // scores
                const scores = new Float32Array(t + 1);
                let maxScore = -Infinity;
                for (let s = 0; s <= t; s++) {
                    const kBase = ((b * T + s) * nKvHead + kvh) * D;
                    let dot = 0;
                    for (let d = 0; d < D; d++) dot += q[qBase + d] * k[kBase + d];
                    dot *= scale;
                    scores[s] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                let sum = 0;
                for (let s = 0; s <= t; s++) {
                    const e = Math.exp(scores[s] - maxScore);
                    scores[s] = e;
                    sum += e;
                }
                for (let s = 0; s <= t; s++) scores[s] /= sum;
                for (let d = 0; d < D; d++) {
                    let acc = 0;
                    for (let s = 0; s <= t; s++) {
                        const vBase = ((b * T + s) * nKvHead + kvh) * D;
                        acc += scores[s] * v[vBase + d];
                    }
                    out[qBase + d] = acc;
                }
            }
        }
    }
    return out;
}

/** SwiGLU：out = silu(gate) * up。 */
export function siluMulRef(gate: Float32Array, up: Float32Array): Float32Array {
    const out = new Float32Array(gate.length);
    for (let i = 0; i < gate.length; i++) {
        const g = gate[i];
        out[i] = (g / (1 + Math.exp(-g))) * up[i];
    }
    return out;
}

/** C[M,N] = A[M,K] @ W[N,K]^T + bias[N]（bias 可为 null，表示不加）。 */
export function gemmNTRef(
    A: Float32Array,
    W: Float32Array,
    M: number,
    N: number,
    K: number,
    bias: Float32Array | null = null,
): Float32Array {
    const C = new Float32Array(M * N);
    for (let m = 0; m < M; m++) {
        for (let n = 0; n < N; n++) {
            let acc = 0;
            const aBase = m * K;
            const wBase = n * K;
            for (let k = 0; k < K; k++) acc += A[aBase + k] * W[wBase + k];
            C[m * N + n] = bias ? acc + bias[n] : acc;
        }
    }
    return C;
}

export interface QwenForwardRefResult
{
    logits: Float32Array;
    /** 每个 block 之后的 hidden（可选，便于定位偏差来自哪一层）。 */
    layerOut: Float32Array[];
}

/**
 * 完整 Qwen2 前向参考：embed → 24×(RMSNorm+[GQA Attn] / RMSNorm+[SwiGLU MLP]) → RMSNorm → tied lm_head。
 */
export function qwenForwardRef(
    weights: QwenWeights,
    config: QwenConfig,
    input: { tokens: Uint32Array; B: number; T: number },
): QwenForwardRefResult {
    const { tokens, B, T } = input;
    const C = config.nEmbd;
    const H = config.nHead;
    const KV = qwenKvDim(config);
    const D = qwenHeadDim(config);
    const I = config.intermediate;
    const V = config.vocabSize;
    const M = B * T;

    // 词嵌入查表
    let x = new Float32Array(M * C);
    for (let m = 0; m < M; m++) {
        const tok = tokens[m];
        const src = tok * C;
        for (let d = 0; d < C; d++) x[m * C + d] = weights.embed[src + d];
    }

    const layerOut: Float32Array[] = [];

    for (let i = 0; i < config.nLayer; i++) {
        const L = weights.layers[i];

        // --- attention 子层 ---
        const n1 = rmsNormRef(x, L.inputNorm, M, C, config.rmsEps);
        const q = gemmNTRef(n1, L.qW, M, C, C, L.qB);
        const k = gemmNTRef(n1, L.kW, M, KV, C, L.kB);
        const v = gemmNTRef(n1, L.vW, M, KV, C, L.vB);
        const qr = ropeHalfRef(q, M, H, D, T, config.ropeTheta);
        const kr = ropeHalfRef(k, M, config.nKvHead, D, T, config.ropeTheta);
        const att = attentionGqaRef(qr, kr, v, B, T, H, config.nKvHead, D);
        const proj = gemmNTRef(att, L.oW, M, C, C);
        const xa = new Float32Array(M * C);
        for (let j = 0; j < M * C; j++) xa[j] = x[j] + proj[j];

        // --- MLP 子层（SwiGLU） ---
        const n2 = rmsNormRef(xa, L.postNorm, M, C, config.rmsEps);
        const gate = gemmNTRef(n2, L.gateW, M, I, C);
        const up = gemmNTRef(n2, L.upW, M, I, C);
        const act = siluMulRef(gate, up);
        const d = gemmNTRef(act, L.downW, M, C, I);
        x = new Float32Array(M * C);
        for (let j = 0; j < M * C; j++) x[j] = xa[j] + d[j];

        layerOut.push(x);
    }

    const xn = rmsNormRef(x, weights.finalNorm, M, C, config.rmsEps);
    const logits = gemmNTRef(xn, weights.embed, M, V, C);
    return { logits, layerOut };
}
