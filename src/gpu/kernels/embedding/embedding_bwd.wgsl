// 词嵌入反向（scatter-add）：dW[v,d] += sum_{row: tokens[row]==v} dX[row,d]。
// 需要「累加」到已有 dW（wte 与 lm_head 共享，梯度会从两条路径汇入）。
// 一个 workgroup 负责一个词表条目 v，线程在 d 维度上并行、在行维度上串行求和（无 fp32 原子加）。
struct Dims { rows: u32, D: u32, vocab: u32, _a: u32 };

@group(0) @binding(0) var<storage, read> tokens: array<u32>;
@group(0) @binding(1) var<storage, read> dx: array<f32>;
@group(0) @binding(2) var<storage, read_write> dW: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let v = wid.x;
  if (v >= dims.vocab) { return; }
  let D = dims.D;
  var d = lid.x;
  loop {
    if (d >= D) { break; }
    var acc = 0.0;
    var r = 0u;
    loop {
      if (r >= dims.rows) { break; }
      if (tokens[r] == v) {
        acc = acc + dx[r * D + d];
      }
      r = r + 1u;
    }
    dW[v * D + d] = dW[v * D + d] + acc;
    d = d + 64u;
  }
}