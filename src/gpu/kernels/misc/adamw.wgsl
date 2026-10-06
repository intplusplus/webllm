// AdamW 逐元素参数更新（解耦权重衰减）。
// m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g^2
// p -= lr * ( (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps) + wd*p )
struct Dims {
  n: u32,
  _a: u32,
  _b: u32,
  _c: u32,
  lr: f32,
  b1: f32,
  b2: f32,
  eps: f32,
  wd: f32,
  b1t: f32,
  b2t: f32,
  _d: f32,
};

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) { return; }
  let g = grad[i];
  let mi = dims.b1 * m[i] + (1.0 - dims.b1) * g;
  let vi = dims.b2 * v[i] + (1.0 - dims.b2) * g * g;
  m[i] = mi;
  v[i] = vi;
  let mhat = mi / (1.0 - dims.b1t);
  let vhat = vi / (1.0 - dims.b2t);
  let p = param[i];
  param[i] = p - dims.lr * (mhat / (sqrt(vhat) + dims.eps) + dims.wd * p);
}