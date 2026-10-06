// 解码专用 split-K GEMV（M=1）—— 部分和阶段：C[1,N] = A[1,K] @ B[N,K]^T 拆成 S 段并行。
//
// 背景：gemm_gemv_f16 的并行度 = 输出列数 N。解码时真实投影的 workgroup 数：
//   k/v_proj N=128 → 1 个、q/o/down N=896 → 4 个、gate/up N=4864 → 19 个，
//   而核显有 8~11 个 CU——绝大多数计算单元在空转。实测解码 137ms/token，
//   远高于按 20.2 GB/s 实测带宽算出的 ~49ms 下限（每步流式读 ~990MB 权重）。
//
// 这里把 K 维切成 S 段并行：grid = [ceil(N/256), S]，第 wid.y 个 workgroup 只累加
// 自己那段 K 的部分和，写入 partial[wid.y*N + n]。随后由 gemm_gemv_split_reduce
// 把 S 份部分和归约成最终输出。
//
// 为什么归约必须独立成第二个 kernel：WGSL 没有 f32 原子加（atomic 仅整型），
// 也没有跨 workgroup 的同步原语，单 pass 无法合并不同 workgroup 的结果。
//
// 约束（CPU 侧 planGemvSplit 保证）：K % S == 0 且 (K/S) % 2 == 0——
// 后者保证每段的 K 起点落在 u32 边界上（一个 u32 = 2 个 f16）。

enable f16;

struct Dims {
  N: u32,
  K: u32,
  S: u32,
  hasBias: u32,
};

const WG: u32 = 256u;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> partial: array<f32>;
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

  let kLen = dims.K / dims.S; // 偶数，由 CPU 侧保证
  let kStart = wid.y * kLen;
  let half = kLen / 2u; // 本段的 u32 数
  let base = n * (dims.K / 2u) + kStart / 2u;

  // 4 路展开：每条 FMA 链独立，便于隐藏全局读取延迟（与 gemm_gemv_f16 相同）
  var acc0 = 0.0;
  var acc1 = 0.0;
  var acc2 = 0.0;
  var acc3 = 0.0;
  let iMain = (half / 2u) * 2u;
  for (var i = 0u; i < iMain; i = i + 2u) {
    let p = unpack2x16float(B[base + i]);
    let q = unpack2x16float(B[base + i + 1u]);
    let k = kStart + 2u * i;
    acc0 = acc0 + A[k] * p.x;
    acc1 = acc1 + A[k + 1u] * p.y;
    acc2 = acc2 + A[k + 2u] * q.x;
    acc3 = acc3 + A[k + 3u] * q.y;
  }
  for (var i = iMain; i < half; i = i + 1u) {
    let p = unpack2x16float(B[base + i]);
    let k = kStart + 2u * i;
    acc0 = acc0 + A[k] * p.x;
    acc1 = acc1 + A[k + 1u] * p.y;
  }

  partial[wid.y * dims.N + n] = (acc0 + acc1) + (acc2 + acc3);
}
