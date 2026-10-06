// C[M,N] = A[M,K] @ B[K,N] (+ bias[N])
//
// 反向传播用：dX = dY @ W（W 为 nn.Linear 的 [out,in] 权重时，令 N=out、K=in）。
// 分块同 gemm_nt：BM=64, BN=64, BK=32；workgroup 256 线程（16x16），每线程 4x4 micro-tile。

struct Dims {
  M: u32,
  N: u32,
  K: u32,
  hasBias: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 32u;

var<workgroup> As: array<f32, 2048>; // BM * BK
var<workgroup> Bs: array<f32, 2048>; // BK * BN

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

    // A 分块 [BM, BK]
    for (var s = 0u; s < 8u; s = s + 1u) {
      let lin = s * 256u + tid;
      let r = lin / BK;
      let c = lin % BK;
      let gm = m0 + r;
      let gk = k0 + c;
      var v = 0.0;
      if (gm < dims.M && gk < dims.K) {
        v = A[gm * dims.K + gk];
      }
      As[lin] = v;
    }

    // B 分块 [BK, BN]
    for (var s = 0u; s < 8u; s = s + 1u) {
      let lin = s * 256u + tid;
      let r = lin / BN;
      let c = lin % BN;
      let gk = k0 + r;
      let gn = n0 + c;
      var v = 0.0;
      if (gk < dims.K && gn < dims.N) {
        v = B[gk * dims.N + gn];
      }
      Bs[lin] = v;
    }

    workgroupBarrier();

    for (var k = 0u; k < BK; k = k + 1u) {
      var ar: array<f32, 4>;
      var br: array<f32, 4>;
      for (var i = 0u; i < 4u; i = i + 1u) {
        ar[i] = As[(rm + i) * BK + k];
      }
      for (var j = 0u; j < 4u; j = j + 1u) {
        br[j] = Bs[k * BN + (rn + j)];
      }
      for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
          acc[i * 4u + j] = acc[i * 4u + j] + ar[i] * br[j];
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