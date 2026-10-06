// 交叉熵（softmax + NLL）反向：dlogits[m,v] = (softmax(logits[m])[v] - [v==target[m]]) / M。
// 一个 workgroup 负责一行。
struct Dims { M: u32, V: u32, _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> dlogits: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= dims.M) { return; }
  let tid = lid.x;
  let V = dims.V;
  let base = row * V;

  // max
  var mx = -1e30;
  var v = tid;
  loop {
    if (v >= V) { break; }
    mx = max(mx, logits[base + v]);
    v = v + 256u;
  }
  red[tid] = mx;
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

  // sum exp
  var s = 0.0;
  var v2 = tid;
  loop {
    if (v2 >= V) { break; }
    s = s + exp(logits[base + v2] - m);
    v2 = v2 + 256u;
  }
  red[tid] = s;
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

  let tgt = targets[row];
  let invM = 1.0 / f32(dims.M);
  var v3 = tid;
  loop {
    if (v3 >= V) { break; }
    let p = exp(logits[base + v3] - m) / sum;
    let one = select(0.0, 1.0, v3 == tgt);
    dlogits[base + v3] = (p - one) * invM;
    v3 = v3 + 256u;
  }
}