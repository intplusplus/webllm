// split-K GEMV 的归约端：C[n] = Σ_s partial[s*N + n] (+ bias[n])。
//
// 与 gemm_gemv_split_f16 共用同一份 Dims uniform 和 partial buffer（同一次提交内
// 按提交顺序执行，partial 在此之前已写完）。S ≤ 8，线性遍历即可；
// 归约的数据量只有 S*N*4 字节（最大 ~155KB），远小于投影本身的权重读取量（MB 级）。
// bias 在这里只加一次，split 阶段不碰 bias。

struct Dims {
  N: u32,
  K: u32,
  S: u32,
  hasBias: u32,
};

const WG: u32 = 256u;

@group(0) @binding(0) var<storage, read> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> C: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let n = wid.x * WG + lid.x;
  if (n >= dims.N) {
    return;
  }

  var v = 0.0;
  for (var s = 0u; s < dims.S; s = s + 1u) {
    v = v + partial[s * dims.N + n];
  }
  if (dims.hasBias != 0u) {
    v = v + bias[n];
  }
  C[n] = v;
}
