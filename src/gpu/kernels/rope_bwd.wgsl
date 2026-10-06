// RoPE 反向：旋转矩阵是正交的，逆变换即转置。
// 前向：out0 = x0*cos - x1*sin, out1 = x0*sin + x1*cos
// 反向：dx0 = d0*cos + d1*sin, dx1 = -d0*sin + d1*cos
// d 形状 [rows, n_head, head_dim]，pos = row % T。
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

@group(0) @binding(0) var<storage, read> d: array<f32>;
@group(0) @binding(1) var<storage, read_write> dx: array<f32>;
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

  let g0 = d[baseIdx + d0];
  let g1 = d[baseIdx + d1];
  dx[baseIdx + d0] = g0 * c + g1 * s;
  dx[baseIdx + d1] = -g0 * s + g1 * c;
}