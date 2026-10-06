// 行内 softmax：每个 workgroup 处理一行，长度 D。
struct Dims {
  rows: u32,
  D: u32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

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

  var localMax = -1e30;
  var d = tid;
  loop {
    if (d >= dims.D) { break; }
    localMax = max(localMax, x[base + d]);
    d = d + 256u;
  }
  red[tid] = localMax;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = max(red[tid], red[tid + stride]); }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let m = red[0];
  workgroupBarrier();

  var localSum = 0.0;
  var d2 = tid;
  loop {
    if (d2 >= dims.D) { break; }
    let e = exp(x[base + d2] - m);
    y[base + d2] = e;
    localSum = localSum + e;
    d2 = d2 + 256u;
  }
  red[tid] = localSum;
  workgroupBarrier();
  stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let sum = red[0];
  workgroupBarrier();

  var d3 = tid;
  loop {
    if (d3 >= dims.D) { break; }
    y[base + d3] = y[base + d3] / sum;
    d3 = d3 + 256u;
  }
}