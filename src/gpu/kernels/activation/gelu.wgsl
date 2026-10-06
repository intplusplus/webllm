// GELU（tanh 近似，与 nanoGPT 的 NewGELU 一致）。
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
struct Dims {
  n: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

const C0: f32 = 0.7978845608028654; // sqrt(2/pi)
const C1: f32 = 0.044715;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) {
    return;
  }
  let v = x[i];
  let inner = C0 * (v + C1 * v * v * v);
  y[i] = 0.5 * v * (1.0 + tanh(inner));
}