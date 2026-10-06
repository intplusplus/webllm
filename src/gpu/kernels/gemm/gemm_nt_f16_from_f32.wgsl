// C[M,N] = A[M,K] @ B[N,K]^T (+ bias[N])，A 仍为 f32、B 为 f16、输出 f32。
//
// 这是 M3 的融合版 f16 GEMM：避免「单独 cast A → f16 buffer → f16 GEMM」多一次
// 全局读写与 dispatch。A 在装载到 shared 时转 f16，寄存器累化仍为 f32。
// 需要 shader-f16 扩展。

enable f16;

struct Dims {
  M: u32,
  N: u32,
  K: u32,
  hasBias: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 32u;

var<workgroup> As: array<f16, 2048>; // BM * BK
var<workgroup> Bs: array<f16, 2048>; // BN * BK

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tx = lid.x;
  let ty = lid.y;
  let tid = ty * 16u + tx;
  let m0 = wid.x * BM;
  let n0 = wid.y * BN;
  let rm = ty * 4u;
  let rn = tx * 4u;

  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) {
    acc[i] = 0.0;
  }

  let kTiles = (dims.K + BK - 1u) / BK;
  for (var t = 0u; t < kTiles; t = t + 1u) {
    let k0 = t * BK;

    // A 分块 [BM, BK]：global f32 → shared f16
    for (var s = 0u; s < 8u; s = s + 1u) {
      let lin = s * 256u + tid;
      let r = lin / BK;
      let c = lin % BK;
      let gm = m0 + r;
      let gk = k0 + c;
      var v: f16 = f16(0.0);
      if (gm < dims.M && gk < dims.K) {
        v = f16(A[gm * dims.K + gk]);
      }
      As[lin] = v;
    }

    // B 分块 [BN, BK]：本来就是 f16
    for (var s = 0u; s < 8u; s = s + 1u) {
      let lin = s * 256u + tid;
      let r = lin / BK;
      let c = lin % BK;
      let gn = n0 + r;
      let gk = k0 + c;
      var v: f16 = f16(0.0);
      if (gn < dims.N && gk < dims.K) {
        v = B[gn * dims.K + gk];
      }
      Bs[lin] = v;
    }

    workgroupBarrier();

    for (var k = 0u; k < BK; k = k + 1u) {
      var ar: array<f16, 4>;
      var br: array<f16, 4>;
      for (var i = 0u; i < 4u; i = i + 1u) {
        ar[i] = As[(rm + i) * BK + k];
      }
      for (var j = 0u; j < 4u; j = j + 1u) {
        br[j] = Bs[(rn + j) * BK + k];
      }
      for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
          acc[i * 4u + j] = acc[i * 4u + j] + f32(ar[i]) * f32(br[j]);
        }
      }
    }

    workgroupBarrier();
  }

  for (var i = 0u; i < 4u; i = i + 1u) {
    for (var j = 0u; j < 4u; j = j + 1u) {
      let gm = m0 + rm + i;
      let gn = n0 + rn + j;
      if (gm < dims.M && gn < dims.N) {
        var v = acc[i * 4u + j];
        if (dims.hasBias != 0u) {
          v = v + bias[gn];
        }
        C[gm * dims.N + gn] = v;
      }
    }
  }
}
