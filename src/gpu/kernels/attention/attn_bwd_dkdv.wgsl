// 注意力反向 —— 第 3 步：键/值梯度。
// dk[s,d] = scale * Σ_{t>=s} dscores[t,s] * q[t,d]
// dv[s,d] =        Σ_{t>=s} p[t,s]       * dout[t,d]
struct Dims {
  B: u32, T: u32, H: u32, D: u32,
  scale: f32,
  _p0: f32, _p1: f32, _p2: f32,
};

@group(0) @binding(0) var<storage, read> ds: array<f32>;
@group(0) @binding(1) var<storage, read> ps: array<f32>;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> dout: array<f32>;
@group(0) @binding(4) var<storage, read_write> dk: array<f32>;
@group(0) @binding(5) var<storage, read_write> dv: array<f32>;
@group(0) @binding(6) var<uniform> dims: Dims;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let s = wid.x;
  let h = wid.y;
  let b = wid.z;
  let tid = lid.x;
  let D = dims.D;
  let T = dims.T;
  if (s >= T || h >= dims.H || b >= dims.B) { return; }

  let kBase = ((b * T + s) * dims.H + h) * D;

  var d = tid;
  loop {
    if (d >= D) { break; }
    var adk = 0.0;
    var adv = 0.0;
    for (var t = s; t < T; t = t + 1u) {
      let idx = ((b * dims.H + h) * T + t) * T + s;
      let rowBase = ((b * T + t) * dims.H + h) * D + d;
      adk = adk + ds[idx] * q[rowBase];
      adv = adv + ps[idx] * dout[rowBase];
    }
    dk[kBase + d] = adk * dims.scale;
    dv[kBase + d] = adv;
    d = d + 128u;
  }
}