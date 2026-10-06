// 词嵌入查表：out[t, :] = W[tokens[t], :]，W 形状 [vocab, D]。
struct Dims {
  T: u32,
  D: u32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = dims.T * dims.D;
  if (i >= total) {
    return;
  }
  let t = i / dims.D;
  let d = i % dims.D;
  let tok = tokens[t];
  out[i] = W[tok * dims.D + d];
}