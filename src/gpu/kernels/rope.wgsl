// RoPE（GPT-NeoX 风格，相邻成对）。x 形状 [rows, n_head, head_dim]，rows = B*T。
// out[2i]   = x[2i] * cos - x[2i+1] * sin
// out[2i+1] = x[2i] * sin + x[2i+1] * cos
// angle = pos * base^(-2i/head_dim)，pos = row % T
struct Dims {
  rows: u32,
  n_head: u32,
  head_dim: u32,
  T: u32,
  base: f32,
  _p0: f32,
  _p1: f32,
  _p2: f32,
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
  let pos = row % dims.T;

  let baseIdx = (row * dims.n_head + h) * dims.head_dim;
  let d0 = pair * 2u;
  let d1 = d0 + 1u;

  let exponent = f32(d0) / f32(dims.head_dim);
  let theta = pow(dims.base, -exponent);
  let angle = f32(pos) * theta;
  let c = cos(angle);
  let s = sin(angle);

  let x0 = x[baseIdx + d0];
  let x1 = x[baseIdx + d1];
  out[baseIdx + d0] = x0 * c - x1 * s;
  out[baseIdx + d1] = x0 * s + x1 * c;
}