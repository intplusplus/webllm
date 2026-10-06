// 注意力反向 —— 第 2 步：查询梯度。
// dq[t,d] = scale * Σ_{s<=t} dscores[t,s] * k[s,d]
struct Dims {
  B: u32, T: u32, H: u32, D: u32,
  scale: f32,
  _p0: f32, _p1: f32, _p2: f32,
};

@group(0) @binding(0) var<storage, read> ds: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> dq: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let t = wid.x;
  let h = wid.y;
  let b = wid.z;
  let tid = lid.x;
  let D = dims.D;
  let T = dims.T;
  if (t >= T || h >= dims.H || b >= dims.B) { return; }

  let rowBase = ((b * dims.H + h) * T + t) * T;
  let qBase = ((b * T + t) * dims.H + h) * D;

  var d = tid;
  loop {
    if (d >= D) { break; }
    var acc = 0.0;
    for (var s = 0u; s <= t; s = s + 1u) {
      acc = acc + ds[rowBase + s] * k[((b * T + s) * dims.H + h) * D + d];
    }
    dq[qBase + d] = acc * dims.scale;
    d = d + 128u;
  }
}