// 因果自注意力（前向，不含 KV cache）。
// q/k/v 形状均为 [B*T, H, D]。一个 workgroup 负责一个 (b, h, t) 组合：
//   scores[s] = q·k_s * scale   (s <= t)
//   softmax -> 对 v 加权求和
// 线程数 128；scores 存于 shared（上限 MAX_T）。

struct Dims {
  B: u32,
  T: u32,
  H: u32,
  D: u32,
  scale: f32,
  _p0: f32,
  _p1: f32,
  _p2: f32,
};

const MAX_T: u32 = 512u;
const WG: u32 = 128u;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

var<workgroup> scores: array<f32, 512>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let t = wid.x;
  let h = wid.y;
  let b = wid.z;
  let tid = lid.x;
  let T = dims.T;
  let D = dims.D;
  let H = dims.H;
  if (t >= T || h >= H || b >= dims.B) {
    return;
  }

  let qBase = ((b * T + t) * H + h) * D;

  // 1) 计算 scores[s]（s <= t）
  var s = tid;
  loop {
    if (s > t) { break; }
    let kBase = ((b * T + s) * H + h) * D;
    var dot = 0.0;
    for (var d = 0u; d < D; d = d + 1u) {
      dot = dot + q[qBase + d] * k[kBase + d];
    }
    scores[s] = dot * dims.scale;
    s = s + WG;
  }
  workgroupBarrier();

  // 2) max 归约
  var localMax = -1e30;
  var s2 = tid;
  loop {
    if (s2 > t) { break; }
    localMax = max(localMax, scores[s2]);
    s2 = s2 + WG;
  }
  red[tid] = localMax;
  workgroupBarrier();
  var stride = 64u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = max(red[tid], red[tid + stride]); }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let m = red[0];
  workgroupBarrier();

  // 3) exp + sum 归约
  var localSum = 0.0;
  var s3 = tid;
  loop {
    if (s3 > t) { break; }
    let e = exp(scores[s3] - m);
    scores[s3] = e;
    localSum = localSum + e;
    s3 = s3 + WG;
  }
  red[tid] = localSum;
  workgroupBarrier();
  stride = 64u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let sum = red[0];
  workgroupBarrier();

  // 4) 对 v 加权求和
  for (var d2 = tid; d2 < D; d2 = d2 + WG) {
    var acc = 0.0;
    var s4 = 0u;
    loop {
      if (s4 > t) { break; }
      let vBase = ((b * T + s4) * H + h) * D;
      acc = acc + (scores[s4] / sum) * v[vBase + d2];
      s4 = s4 + 1u;
    }
    out[qBase + d2] = acc;
  }
}