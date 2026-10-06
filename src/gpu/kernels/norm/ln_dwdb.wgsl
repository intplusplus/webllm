// LayerNorm 反向 —— 第 3 步：权重/偏置梯度（对行求和）。
// dw[d] = sum_m dy[m,d]*xhat[m,d], db[d] = sum_m dy[m,d]
// 一个 workgroup 负责一列 d，线程在行维度上归约（避免 fp32 原子加）。
struct Dims { rows: u32, D: u32, _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;
@group(0) @binding(4) var<storage, read_write> db: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

var<workgroup> red: array<f32, 64>;

fn reduceAdd(tid: u32) -> f32 {
  var stride = 32u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  return red[0];
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let d = wid.x;
  if (d >= dims.D) { return; }
  let tid = lid.x;

  var sdw = 0.0;
  var sdb = 0.0;
  var m = tid;
  loop {
    if (m >= dims.rows) { break; }
    let base = m * dims.D + d;
    let mean = stats[m * 4u + 0u];
    let rstd = stats[m * 4u + 1u];
    let xhat = (x[base] - mean) * rstd;
    sdw = sdw + dy[base] * xhat;
    sdb = sdb + dy[base];
    m = m + 64u;
  }

  red[tid] = sdw;
  workgroupBarrier();
  let vw = reduceAdd(tid);
  workgroupBarrier();
  red[tid] = sdb;
  workgroupBarrier();
  let vb = reduceAdd(tid);
  workgroupBarrier();

  if (tid == 0u) {
    dw[d] = vw;
    db[d] = vb;
  }
}