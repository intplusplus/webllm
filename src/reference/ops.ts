/**
 * 前向算子的 CPU 参考实现。所有 GPU kernel 都必须与此对拍。
 */

/** C[M,N] = A[M,K] @ B[N,K]^T + bias[N]（B 为 [out, in] 布局）。 */
export function gemmNTRef(
  A: Float32Array,
  B: Float32Array,
  M: number,
  N: number,
  K: number,
  bias?: Float32Array,
): Float32Array {
  const C = new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let acc = 0;
      for (let k = 0; k < K; k++) acc += A[m * K + k] * B[n * K + k];
      C[m * N + n] = acc + (bias ? bias[n] : 0);
    }
  }
  return C;
}

/** 词嵌入查表：W[token, :]，W 形状 [vocab, D]。 */
export function embeddingRef(W: Float32Array, D: number, tokens: Uint32Array): Float32Array {
  const T = tokens.length;
  const out = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const tok = tokens[t];
    for (let d = 0; d < D; d++) out[t * D + d] = W[tok * D + d];
  }
  return out;
}

/** RMSNorm：y = x / sqrt(mean(x^2) + eps) * w。 */
export function rmsnormRef(
  x: Float32Array,
  rows: number,
  D: number,
  w: Float32Array,
  eps: number,
): Float32Array {
  const y = new Float32Array(rows * D);
  for (let r = 0; r < rows; r++) {
    const base = r * D;
    let sum = 0;
    for (let d = 0; d < D; d++) {
      const v = x[base + d];
      sum += v * v;
    }
    const rstd = 1 / Math.sqrt(sum / D + eps);
    for (let d = 0; d < D; d++) y[base + d] = x[base + d] * rstd * w[d];
  }
  return y;
}

/** LayerNorm：y = (x - mean) / sqrt(var + eps) * w + b。 */
export function layernormRef(
  x: Float32Array,
  rows: number,
  D: number,
  w: Float32Array,
  b: Float32Array,
  eps: number,
  hasBias: boolean,
): Float32Array {
  const y = new Float32Array(rows * D);
  for (let r = 0; r < rows; r++) {
    const base = r * D;
    let sum = 0;
    let sumsq = 0;
    for (let d = 0; d < D; d++) {
      const v = x[base + d];
      sum += v;
      sumsq += v * v;
    }
    const mean = sum / D;
    const variance = Math.max(sumsq / D - mean * mean, 0);
    const rstd = 1 / Math.sqrt(variance + eps);
    for (let d = 0; d < D; d++) {
      let v = (x[base + d] - mean) * rstd * w[d];
      if (hasBias) v += b[d];
      y[base + d] = v;
    }
  }
  return y;
}

/** GELU（tanh 近似）。 */
export function geluRef(x: Float32Array): Float32Array {
  const y = new Float32Array(x.length);
  const C0 = Math.sqrt(2 / Math.PI);
  const C1 = 0.044715;
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    y[i] = 0.5 * v * (1 + Math.tanh(C0 * (v + C1 * v * v * v)));
  }
  return y;
}

/** RoPE（GPT-NeoX 相邻成对）。x 形状 [rows, H, D]，pos = row % T。 */
export function ropeRef(
  x: Float32Array,
  rows: number,
  H: number,
  D: number,
  T: number,
  base: number,
): Float32Array {
  const out = new Float32Array(x.length);
  const half = D / 2;
  for (let row = 0; row < rows; row++) {
    const pos = row % T;
    for (let h = 0; h < H; h++) {
      const baseIdx = (row * H + h) * D;
      for (let p = 0; p < half; p++) {
        const d0 = 2 * p;
        const d1 = d0 + 1;
        const theta = Math.pow(base, -d0 / D);
        const angle = pos * theta;
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        const x0 = x[baseIdx + d0];
        const x1 = x[baseIdx + d1];
        out[baseIdx + d0] = x0 * c - x1 * s;
        out[baseIdx + d1] = x0 * s + x1 * c;
      }
    }
  }
  return out;
}

/** 行内 softmax，x 形状 [rows, D]。 */
export function softmaxRef(x: Float32Array, rows: number, D: number): Float32Array {
  const y = new Float32Array(rows * D);
  for (let r = 0; r < rows; r++) {
    const base = r * D;
    let m = -Infinity;
    for (let d = 0; d < D; d++) m = Math.max(m, x[base + d]);
    let sum = 0;
    for (let d = 0; d < D; d++) {
      const e = Math.exp(x[base + d] - m);
      y[base + d] = e;
      sum += e;
    }
    for (let d = 0; d < D; d++) y[base + d] /= sum;
  }
  return y;
}

/** C[M,N] = A[M,K] @ B[K,N] + bias[N]（B 为 [in, out] 布局，即 nn.Linear 转置后的形状）。 */
export function gemmNNRef(
  A: Float32Array,
  B: Float32Array,
  M: number,
  N: number,
  K: number,
  bias?: Float32Array,
): Float32Array {
  const C = new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let acc = 0;
      for (let k = 0; k < K; k++) acc += A[m * K + k] * B[k * N + n];
      C[m * N + n] = acc + (bias ? bias[n] : 0);
    }
  }
  return C;
}

/** C[M,N] = A[K,M]^T @ B[K,N] + bias[N]（反向的权重梯度 dW = dY^T @ X）。 */
export function gemmTNRef(
  A: Float32Array,
  B: Float32Array,
  M: number,
  N: number,
  K: number,
  bias?: Float32Array,
): Float32Array {
  const C = new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let acc = 0;
      for (let k = 0; k < K; k++) acc += A[k * M + m] * B[k * N + n];
      C[m * N + n] = acc + (bias ? bias[n] : 0);
    }
  }
  return C;
}

/** GELU（tanh 近似）反向：dx = dy * gelu'(x)。 */
export function geluBwdRef(x: Float32Array, dy: Float32Array): Float32Array {
  const C0 = Math.sqrt(2 / Math.PI);
  const C1 = 0.044715;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    const u = C0 * (v + C1 * v * v * v);
    const t = Math.tanh(u);
    const dudx = C0 * (1 + 3 * C1 * v * v);
    const g = 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * dudx;
    out[i] = dy[i] * g;
  }
  return out;
}

/** RoPE 反向（逆旋转）。d 形状 [rows, H, D]，pos = row % T。 */
export function ropeBwdRef(
  d: Float32Array,
  rows: number,
  H: number,
  D: number,
  T: number,
  base: number,
): Float32Array {
  const out = new Float32Array(d.length);
  const half = D / 2;
  for (let row = 0; row < rows; row++) {
    const pos = row % T;
    for (let h = 0; h < H; h++) {
      const baseIdx = (row * H + h) * D;
      for (let p = 0; p < half; p++) {
        const d0 = 2 * p;
        const d1 = d0 + 1;
        const theta = Math.pow(base, -d0 / D);
        const angle = pos * theta;
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        const g0 = d[baseIdx + d0];
        const g1 = d[baseIdx + d1];
        out[baseIdx + d0] = g0 * c + g1 * s;
        out[baseIdx + d1] = -g0 * s + g1 * c;
      }
    }
  }
  return out;
}

/** 因果自注意力（前向）。q/k/v 形状 [B*T, H, D]。 */
export function attentionRef(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  B: number,
  T: number,
  H: number,
  D: number,
  causal = true,
): Float32Array {
  const out = new Float32Array(B * T * H * D);
  const scale = 1 / Math.sqrt(D);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      for (let t = 0; t < T; t++) {
        const qBase = ((b * T + t) * H + h) * D;
        const sMax = causal ? t : T - 1;
        const scores = new Float32Array(sMax + 1);
        let m = -Infinity;
        for (let s = 0; s <= sMax; s++) {
          const kBase = ((b * T + s) * H + h) * D;
          let dot = 0;
          for (let d = 0; d < D; d++) dot += q[qBase + d] * k[kBase + d];
          scores[s] = dot * scale;
          if (scores[s] > m) m = scores[s];
        }
        let sum = 0;
        for (let s = 0; s <= sMax; s++) {
          scores[s] = Math.exp(scores[s] - m);
          sum += scores[s];
        }
        for (let d = 0; d < D; d++) {
          let acc = 0;
          for (let s = 0; s <= sMax; s++) {
            const vBase = ((b * T + s) * H + h) * D;
            acc += (scores[s] / sum) * v[vBase + d];
          }
          out[qBase + d] = acc;
        }
      }
    }
  }
  return out;
}

/** 因果自注意力反向。返回 dq/dk/dv，形状同 q。 */
export function attentionBwdRef(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  dout: Float32Array,
  B: number,
  T: number,
  H: number,
  D: number,
): { dq: Float32Array; dk: Float32Array; dv: Float32Array } {
  const dq = new Float32Array(B * T * H * D);
  const dk = new Float32Array(B * T * H * D);
  const dv = new Float32Array(B * T * H * D);
  const scale = 1 / Math.sqrt(D);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      for (let t = 0; t < T; t++) {
        const qBase = ((b * T + t) * H + h) * D;
        const scores = new Float32Array(t + 1);
        const p = new Float32Array(t + 1);
        const dp = new Float32Array(t + 1);
        let m = -Infinity;
        for (let s = 0; s <= t; s++) {
          const kBase = ((b * T + s) * H + h) * D;
          let dot = 0;
          for (let d = 0; d < D; d++) dot += q[qBase + d] * k[kBase + d];
          scores[s] = dot * scale;
          if (scores[s] > m) m = scores[s];
        }
        let sum = 0;
        for (let s = 0; s <= t; s++) {
          p[s] = Math.exp(scores[s] - m);
          sum += p[s];
        }
        for (let s = 0; s <= t; s++) {
          const vBase = ((b * T + s) * H + h) * D;
          let acc = 0;
          for (let d = 0; d < D; d++) acc += dout[qBase + d] * v[vBase + d];
          dp[s] = acc;
        }
        let sumPdp = 0;
        for (let s = 0; s <= t; s++) sumPdp += (p[s] / sum) * dp[s];
        for (let s = 0; s <= t; s++) {
          const ds = (p[s] / sum) * (dp[s] - sumPdp);
          const kBase = ((b * T + s) * H + h) * D;
          for (let d = 0; d < D; d++) {
            dq[qBase + d] += ds * k[kBase + d] * scale;
            dk[kBase + d] += ds * q[qBase + d] * scale;
            dv[kBase + d] += (p[s] / sum) * dout[qBase + d];
          }
        }
      }
    }
  }
  return { dq, dk, dv };
}

/** LayerNorm 反向。返回 dx、dw、db。 */
export function layernormBwdRef(
  x: Float32Array,
  rows: number,
  D: number,
  w: Float32Array,
  dy: Float32Array,
  eps: number,
): { dx: Float32Array; dw: Float32Array; db: Float32Array } {
  const dx = new Float32Array(rows * D);
  const dw = new Float32Array(D);
  const db = new Float32Array(D);
  for (let r = 0; r < rows; r++) {
    const base = r * D;
    let sum = 0;
    let sumsq = 0;
    for (let d = 0; d < D; d++) {
      const v = x[base + d];
      sum += v;
      sumsq += v * v;
    }
    const mean = sum / D;
    const variance = Math.max(sumsq / D - mean * mean, 0);
    const rstd = 1 / Math.sqrt(variance + eps);
    let mg = 0;
    let mgx = 0;
    for (let d = 0; d < D; d++) {
      const xhat = (x[base + d] - mean) * rstd;
      const g = dy[base + d] * w[d];
      mg += g;
      mgx += g * xhat;
    }
    mg /= D;
    mgx /= D;
    for (let d = 0; d < D; d++) {
      const xhat = (x[base + d] - mean) * rstd;
      dx[base + d] = rstd * (dy[base + d] * w[d] - mg - xhat * mgx);
      dw[d] += dy[base + d] * xhat;
      db[d] += dy[base + d];
    }
  }
  return { dx, dw, db };
}

/** 对行求和：out[col] = sum_row x[row, col]。 */
export function sumRowsRef(x: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(cols);
  for (let c = 0; c < cols; c++) {
    let s = 0;
    for (let r = 0; r < rows; r++) s += x[r * cols + c];
    out[c] = s;
  }
  return out;
}

/** softmax + 交叉熵反向：dlogits = (softmax(logits) - onehot) / M。 */
export function ceSoftmaxBwdRef(
  logits: Float32Array,
  M: number,
  V: number,
  targets: Uint32Array,
): Float32Array {
  const out = new Float32Array(M * V);
  for (let r = 0; r < M; r++) {
    const base = r * V;
    let m = -Infinity;
    for (let v = 0; v < V; v++) m = Math.max(m, logits[base + v]);
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(logits[base + v] - m);
    for (let v = 0; v < V; v++) {
      const p = Math.exp(logits[base + v] - m) / sum;
      out[base + v] = (p - (v === targets[r] ? 1 : 0)) / M;
    }
  }
  return out;
}

/** 词嵌入反向（scatter-add）：dW[v,d] = Σ_{row: tokens[row]==v} dx[row,d]。 */
export function embeddingBwdRef(
  tokens: Uint32Array,
  dx: Float32Array,
  vocab: number,
  D: number,
): Float32Array {
  const dW = new Float32Array(vocab * D);
  for (let r = 0; r < tokens.length; r++) {
    const v = tokens[r];
    for (let d = 0; d < D; d++) dW[v * D + d] += dx[r * D + d];
  }
  return dW;
}

/** AdamW 单步（就地更新 param/m/v）。 */
export function adamwRef(
  param: Float32Array,
  grad: Float32Array,
  m: Float32Array,
  v: Float32Array,
  opts: { lr: number; b1: number; b2: number; eps: number; wd: number; t: number },
): void {
  const { lr, b1, b2, eps, wd, t } = opts;
  const b1t = Math.pow(b1, t);
  const b2t = Math.pow(b2, t);
  for (let i = 0; i < param.length; i++) {
    const g = grad[i];
    const mi = b1 * m[i] + (1 - b1) * g;
    const vi = b2 * v[i] + (1 - b2) * g * g;
    m[i] = mi;
    v[i] = vi;
    const mhat = mi / (1 - b1t);
    const vhat = vi / (1 - b2t);
    param[i] = param[i] - lr * (mhat / (Math.sqrt(vhat) + eps) + wd * param[i]);
  }
}