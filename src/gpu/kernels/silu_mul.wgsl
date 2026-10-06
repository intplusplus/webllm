// SwiGLU 融合激活：out = silu(gate) * up。
//
// Qwen2 的 MLP 为 down(silu(gate(x)) * up(x))。把 SiLU 与逐元素相乘融合成一个
// kernel，省掉一次全局读写与 dispatch。
// silu(v) = v / (1 + exp(-v))，与 hidden_act="silu" 一致。

struct Dims {
  n: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) {
    return;
  }
  let g = gate[i];
  out[i] = (g / (1.0 + exp(-g))) * up[i];
}
