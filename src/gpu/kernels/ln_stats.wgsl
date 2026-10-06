// LayerNorm 反向 —— 第 1 步：逐行统计量。
// 前向：y = (x-mean)/sqrt(var+eps)*w + b，记 xhat = (x-mean)*rstd。
// 反向需要每行：mean、rstd、mg = mean_d(dy*w)、mgx = mean_d((dy*w)*xhat)。
// 输出 stats[row*4 + {mean, rstd, mg, mgx}]。
struct Dims { rows: u32, D: u32, eps: f32, _p: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> stats: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

var<workgroup> red: array<f32, 256>;

fn reduceAdd(tid: u32) -> f32 {
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  return red[0];
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= dims.rows) { return; }
  let tid = lid.x;
  let D = dims.D;
  let base = row * D;

  var sx = 0.0;
  var sxx = 0.0;
  var d = tid;
  loop {
    if (d >= D) { break; }
    let v = x[base + d];
    sx = sx + v;
    sxx = sxx + v * v;
    d = d + 256u;
  }
  red[tid] = sx;
  workgroupBarrier();
  let mean = reduceAdd(tid) / f32(D);
  workgroupBarrier();
  red[tid] = sxx;
  workgroupBarrier();
  let meanSq = reduceAdd(tid) / f32(D);
  workgroupBarrier();
  let rstd = inverseSqrt(max(meanSq - mean * mean, 0.0) + dims.eps);

  var sg = 0.0;
  var sgx = 0.0;
  var d2 = tid;
  loop {
    if (d2 >= D) { break; }
    let xhat = (x[base + d2] - mean) * rstd;
    let g = dy[base + d2] * w[d2];
    sg = sg + g;
    sgx = sgx + g * xhat;
    d2 = d2 + 256u;
  }
  red[tid] = sg;
  workgroupBarrier();
  let mg = reduceAdd(tid) / f32(D);
  workgroupBarrier();
  red[tid] = sgx;
  workgroupBarrier();
  let mgx = reduceAdd(tid) / f32(D);
  workgroupBarrier();

  if (tid == 0u) {
    stats[row * 4u + 0u] = mean;
    stats[row * 4u + 1u] = rstd;
    stats[row * 4u + 2u] = mg;
    stats[row * 4u + 3u] = mgx;
  }
}