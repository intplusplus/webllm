// Qwen2 风格 RoPE（half-split，对应 HF 的 rotate_half）。
//
// 与 GPT-NeoX 相邻成对约定不同：HF Qwen2 把 head_dim 均分成前后两半，旋转的是
// (i, i + half) 这一对。因为直接加载 HF 原始权重（未做 permute），必须用本约定，
// 不能复用 rope.wgsl（那是相邻成对）。
//
//   out[i]      = x[i] * cos(angle_i) - x[i+half] * sin(angle_i)
//   out[i+half] = x[i+half] * cos(angle_i) + x[i] * sin(angle_i)
//   angle_i = pos * base^(-2i/head_dim),  i ∈ [0, half)
//
// x 形状 [rows, n_head, head_dim]，rows = B*T，pos = row % T + posOffset。
// posOffset 用于 KV cache 的增量解码：单步只算 1 行，位置是它在整段上下文里的绝对下标。

struct Dims {
  rows: u32,
  n_head: u32,
  head_dim: u32,
  T: u32,
  base: f32,
  posOffset: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = dims.head_dim / 2u;
  let total = dims.rows * dims.n_head * half;
  let i = gid.x;
  if (i >= total) {
    return;
  }

  let pair = i % half;
  let h = (i / half) % dims.n_head;
  let row = i / (half * dims.n_head);
  let pos = row % dims.T + dims.posOffset;

  let baseIdx = (row * dims.n_head + h) * dims.head_dim;
  let d0 = pair;
  let d1 = pair + half;

  // 频率下标 pair 对应 2*pair/head_dim 的指数
  let exponent = f32(pair * 2u) / f32(dims.head_dim);
  let theta = pow(dims.base, -exponent);
  let angle = f32(pos) * theta;
  let c = cos(angle);
  let s = sin(angle);

  let x0 = x[baseIdx + d0];
  let x1 = x[baseIdx + d1];
  out[baseIdx + d0] = x0 * c - x1 * s;
  out[baseIdx + d1] = x0 * s + x1 * c;
}
