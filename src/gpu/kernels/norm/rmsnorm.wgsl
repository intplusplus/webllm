// RMSNorm：y = x / sqrt(mean(x^2) + eps) * weight。每个 workgroup 处理一行。
struct Dims {
  rows: u32,
  D: u32,
  eps: f32,
  _p0: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= dims.rows) {
    return;
  }
  let tid = lid.x;
  let base = row * dims.D;

  var sum = 0.0;
  var d = tid;
  loop {
    if (d >= dims.D) { break; }
    let v = x[base + d];
    sum = sum + v * v;
    d = d + 256u;
  }
  red[tid] = sum;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      red[tid] = red[tid] + red[tid + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  let rstd = inverseSqrt(red[0] / f32(dims.D) + dims.eps);
  workgroupBarrier();

  var d2 = tid;
  loop {
    if (d2 >= dims.D) { break; }
    y[base + d2] = x[base + d2] * rstd * weight[d2];
    d2 = d2 + 256u;
  }
}