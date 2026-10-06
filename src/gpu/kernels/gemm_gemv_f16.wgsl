// 解码专用 GEMM（M=1）：C[1,N] = A[1,K] @ B[N,K]^T (+ bias[N])。
//
// 为什么需要单独一个 kernel：通用 gemm_nt 沿 M 方向按 BM=64 分块，解码时 M 恒为 1，
// 于是 98% 的算力与访存都被浪费在填充行上（M=1 与 M=43 耗时几乎相同）。
// 这里让每个线程只负责一个输出列：A 只有一行、被同一 workgroup 的所有线程广播读，
// 天然命中 L1；B 按行顺序流式读取，每个字节都被用到，DRAM 流量等于权重体积。
//
// B 以 array<u32> 绑定：一个 u32 = 2 个 f16，用 unpack2x16float 还原。这样每条加载
// 指令搬运 2 个权重，直接减半 load 指令数——实测该 kernel 是 load 指令受限而非带宽受限。
// 因此要求 K 为偶数（Qwen2.5 的所有投影 K ∈ {896, 4864} 均满足）。
//
// A 为 f32、输出 f32，与 gemm_nt_f16_from_f32 的约定一致。需要 shader-f16 扩展。

enable f16;

struct Dims {
  M: u32,
  N: u32,
  K: u32,
  hasBias: u32,
};

const WG: u32 = 256u;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let n = wid.x * WG + lid.x;
  if (n >= dims.N) {
    return;
  }

  let half = dims.K / 2u; // 每行的 u32 个数
  let base = n * half;

  // 4 路展开：每条 FMA 链独立，便于隐藏全局读取延迟
  var acc0 = 0.0;
  var acc1 = 0.0;
  var acc2 = 0.0;
  var acc3 = 0.0;
  let iMain = (half / 2u) * 2u;
  for (var i = 0u; i < iMain; i = i + 2u) {
    let p = unpack2x16float(B[base + i]);
    let q = unpack2x16float(B[base + i + 1u]);
    let k = 2u * i;
    acc0 = acc0 + A[k] * p.x;
    acc1 = acc1 + A[k + 1u] * p.y;
    acc2 = acc2 + A[k + 2u] * q.x;
    acc3 = acc3 + A[k + 3u] * q.y;
  }
  for (var i = iMain; i < half; i = i + 1u) {
    let p = unpack2x16float(B[base + i]);
    let k = 2u * i;
    acc0 = acc0 + A[k] * p.x;
    acc1 = acc1 + A[k + 1u] * p.y;
  }

  var v = (acc0 + acc1) + (acc2 + acc3);
  if (dims.hasBias != 0u) {
    v = v + bias[n];
  }
  C[n] = v;
}
