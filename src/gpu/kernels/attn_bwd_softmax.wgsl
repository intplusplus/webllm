// 注意力反向 —— 第 1 步：重算每个查询位置 t 的 softmax 概率 p[t,s] 与 dscores[t,s]。
//   scores[s] = (q_t·k_s)*scale, s<=t
//   p[s] = softmax(scores)
//   dp[s] = dout_t·v_s
//   sumPdp = Σ p[s]*dp[s]
//   dscores[s] = p[s]*(dp[s]-sumPdp)
// 输出 ps / ds，均为 [B,H,T,T] 布局：((b*H+h)*T + t)*T + s。
struct Dims {
  B: u32, T: u32, H: u32, D: u32,
  scale: f32,
  _p0: f32, _p1: f32, _p2: f32,
};

const MAX_T: u32 = 512u;
const WG: u32 = 128u;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> dout: array<f32>;
@group(0) @binding(4) var<storage, read_write> ps: array<f32>;
@group(0) @binding(5) var<storage, read_write> ds: array<f32>;
@group(0) @binding(6) var<uniform> dims: Dims;

var<workgroup> scores: array<f32, 512>;
var<workgroup> dp: array<f32, 512>;
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
  if (t >= T || h >= H || b >= dims.B) { return; }

  let qBase = ((b * T + t) * H + h) * D;

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

  var s4 = tid;
  loop {
    if (s4 > t) { break; }
    let vBase = ((b * T + s4) * H + h) * D;
    var acc = 0.0;
    for (var d = 0u; d < D; d = d + 1u) {
      acc = acc + dout[qBase + d] * v[vBase + d];
    }
    dp[s4] = acc;
    s4 = s4 + WG;
  }
  workgroupBarrier();

  var lpd = 0.0;
  var s5 = tid;
  loop {
    if (s5 > t) { break; }
    lpd = lpd + (scores[s5] / sum) * dp[s5];
    s5 = s5 + WG;
  }
  red[tid] = lpd;
  workgroupBarrier();
  stride = 64u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let sumPdp = red[0];
  workgroupBarrier();

  let rowBase = ((b * H + h) * T + t) * T;
  var s6 = tid;
  loop {
    if (s6 > t) { break; }
    let p = scores[s6] / sum;
    ps[rowBase + s6] = p;
    ds[rowBase + s6] = p * (dp[s6] - sumPdp);
    s6 = s6 + WG;
  }
}