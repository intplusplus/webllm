// GELU（tanh 近似）反向：dx = dy * gelu'(x)。
// gelu(x) = 0.5*x*(1+tanh(u)), u = c0*(x + c1*x^3), c0=sqrt(2/pi), c1=0.044715
// gelu'(x) = 0.5*(1+tanh(u)) + 0.5*x*(1-tanh(u)^2)*c0*(1+3*c1*x^2)

struct Dims { n: u32, _a: u32, _b: u32, _c: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const C0: f32 = 0.7978845608028654; // sqrt(2/pi)
const C1: f32 = 0.044715;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) {
    return;
  }
  let v = x[i];
  let u = C0 * (v + C1 * v * v * v);
  let t = tanh(u);
  let dudx = C0 * (1.0 + 3.0 * C1 * v * v);
  let g = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * dudx;
  dx[i] = dy[i] * g;
}