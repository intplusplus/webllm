// LayerNorm 反向 —— 第 2 步：输入梯度。
// dx = rstd * (dy*w - mg - xhat*mgx)
struct Dims { rows: u32, D: u32, _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.rows * dims.D) { return; }
  let row = i / dims.D;
  let d = i % dims.D;
  let mean = stats[row * 4u + 0u];
  let rstd = stats[row * 4u + 1u];
  let mg = stats[row * 4u + 2u];
  let mgx = stats[row * 4u + 3u];
  let xhat = (x[i] - mean) * rstd;
  dx[i] = rstd * (dy[i] * w[d] - mg - xhat * mgx);
}