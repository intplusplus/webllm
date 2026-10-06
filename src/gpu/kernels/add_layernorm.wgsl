// 融合「残差相加 + LayerNorm」：y = LN(a + b)，同时输出需要的残差 r = a + b。
// 与「vec_add 写出 r → layernorm 读回 r」相比，少一次全量 f32 residual 的写+读。
// fp32 累加/输出，block 内 workgroup 并行归约，不使用 subgroup。
struct Dims {
  rows: u32,
  D: u32,
  eps: f32,
  hasBias: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read_write> r: array<f32>;
@group(0) @binding(6) var<uniform> dims: Dims;

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
  if (row >= dims.rows) {
    return;
  }
  let tid = lid.x;
  let base = row * dims.D;

  // 1) sum(a + b), sum((a + b)^2)
  var sum = 0.0;
  var sumsq = 0.0;
  var d = tid;
  loop {
    if (d >= dims.D) { break; }
    let v = a[base + d] + b[base + d];
    sum = sum + v;
    sumsq = sumsq + v * v;
    d = d + 256u;
  }

  red[tid] = sum;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let mean = red[0] / f32(dims.D);
  workgroupBarrier();

  red[tid] = sumsq;
  workgroupBarrier();
  stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let meanSq = red[0] / f32(dims.D);
  let rstd = inverseSqrt(max(meanSq - mean * mean, 0.0) + dims.eps);
  workgroupBarrier();

  // 2) 同时输出 residual 与 LayerNorm 结果
  var d2 = tid;
  loop {
    if (d2 >= dims.D) { break; }
    let v = a[base + d2] + b[base + d2];
    r[base + d2] = v;

    var yv = (v - mean) * rstd * weight[d2];
    if (dims.hasBias != 0u) {
      yv = yv + bias[d2];
    }
    y[base + d2] = yv;
    d2 = d2 + 256u;
  }
}
