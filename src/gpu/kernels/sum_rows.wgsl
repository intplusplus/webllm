// 对行求和：out[col] = sum_row x[row, col]。用于 Linear 的 bias 梯度。
// 一个 workgroup 负责一列，线程在行维度上归约（避免 fp32 原子加）。
struct Dims { rows: u32, cols: u32, _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let col = wid.x;
  if (col >= dims.cols) { return; }
  let tid = lid.x;

  var s = 0.0;
  var r = tid;
  loop {
    if (r >= dims.rows) { break; }
    s = s + x[r * dims.cols + col];
    r = r + 256u;
  }
  red[tid] = s;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (tid == 0u) { out[col] = red[0]; }
}