// 交叉熵（softmax + NLL）反向，支持逐行权重：
//   dlogits[m,v] = w[m] * (softmax(logits[m])[v] - [v==target[m]])
//
// w[m] 的三种用法（由调用方归一后传入）：
//   - 预训练：w ≡ 1/M —— 等价于平均 CE（旧行为）
//   - SFT：prompt/masked 位置 w=0（loss 只算 completion），有效行 w=1/N
//   - DPO：chosen 行 w=+β(1-σ(z))/N、rejected 行 w=-β(1-σ(z))/N ——
//     把标量 loss 对 token-logprob 之和的梯度折进 CE 反向
// 一个 workgroup 负责一行。
struct Dims { M: u32, V: u32, _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> dlogits: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

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
  let w = weights[row];

  if (w == 0.0) {
    // masked 行：零梯度，直接清零返回（softmax 也无需计算）
    var vz = tid;
    loop {
      if (vz >= V) { break; }
      dlogits[base + vz] = 0.0;
      vz = vz + 256u;
    }
    return;
  }

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
  var v3 = tid;
  loop {
    if (v3 >= V) { break; }
    let p = exp(logits[base + v3] - m) / sum;
    let one = select(0.0, 1.0, v3 == tgt);
    dlogits[base + v3] = w * (p - one);
    v3 = v3 + 256u;
  }
}
