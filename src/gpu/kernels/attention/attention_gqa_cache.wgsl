// 带 KV cache 的因果 GQA 注意力（增量解码）。
//
// q 只包含「本次新增的 M 行」（预填充时 M = prompt 长度，解码时 M = 1）；
// k/v 从 cache 中读取，cache 布局为扁平的 [maxT, nKvHead * D]。
// 第 m 行 query 的绝对位置是 pastLen + m，只可见 s ∈ [0, pastLen + m]。
//
// 与 attention_gqa.wgsl 的区别：那边 k/v 与 q 都是本次算出的 [B*T, nKvHead, D]，
// 这里 k/v 来自 cache 的长缓冲，且按「位置的模」不再区分 batch（B 恒为 1）。
// 一个 workgroup 负责一个 (m, h)；scores 存于 shared，上限 MAX_T。

struct Dims {
  M: u32,
  nHead: u32,
  nKvHead: u32,
  D: u32,
  scale: f32,
  pastLen: u32,
  _p0: u32,
  _p1: u32,
};

const MAX_T: u32 = 512u;
const WG: u32 = 128u;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kCache: array<f32>;
@group(0) @binding(2) var<storage, read> vCache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

var<workgroup> scores: array<f32, 512>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let m = wid.x;
  let h = wid.y;
  let tid = lid.x;
  let D = dims.D;
  if (m >= dims.M || h >= dims.nHead) {
    return;
  }

  let kvh = h / (dims.nHead / dims.nKvHead);
  let kvBase = kvh * D;
  let rowLen = dims.nKvHead * D;
  let visible = dims.pastLen + m; // 可见的最后一个 key 下标（含）
  if (visible >= MAX_T) {
    return;
  }
  let qBase = (m * dims.nHead + h) * D;

  // 1) scores[s] = q·k_s * scale
  var s = tid;
  loop {
    if (s > visible) { break; }
    let kBase = s * rowLen + kvBase;
    var dot = 0.0;
    for (var d = 0u; d < D; d = d + 1u) {
      dot = dot + q[qBase + d] * kCache[kBase + d];
    }
    scores[s] = dot * dims.scale;
    s = s + WG;
  }
  workgroupBarrier();

  // 2) max 归约
  var localMax = -1e30;
  var s2 = tid;
  loop {
    if (s2 > visible) { break; }
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
  let mx = red[0];
  workgroupBarrier();

  // 3) exp 与 sum 归约
  var localSum = 0.0;
  var s3 = tid;
  loop {
    if (s3 > visible) { break; }
    let e = exp(scores[s3] - mx);
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
      if (s4 > visible) { break; }
      let vBase = s4 * rowLen + kvBase;
      acc = acc + (scores[s4] / sum) * vCache[vBase + d2];
      s4 = s4 + 1u;
    }
    out[qBase + d2] = acc;
  }
}
