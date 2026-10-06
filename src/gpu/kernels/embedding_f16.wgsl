// f16 词嵌入查表：out[t, :] = W[tokens[t], :]，W 形状 [vocab, D] 且为 f16。
//
// Qwen2.5-0.5B 权重原生 fp16，embed_tokens 有 151936×896 = 1.36e8 个元素；
// 以 f16 常驻只需 272MB（f32 要 544MB），输出仍为 f32 供后续算子使用。
// 需要 shader-f16 扩展。

enable f16;

struct Dims {
  rows: u32,
  D: u32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read> W: array<f16>;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.rows * dims.D) {
    return;
  }
  let t = i / dims.D;
  let d = i % dims.D;
  out[i] = f32(W[tokens[t] * dims.D + d]);
}
