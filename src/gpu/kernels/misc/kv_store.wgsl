// 把新算出的 k/v 追加写入 KV cache：dst[pos0 + row, :] = src[row, :]。
//
// cache 布局为扁平的 [maxT, rowLen]，rowLen = nKvHead * head_dim，与 k/v 投影输出一致。
// 用独立 kernel 而不是让 GEMM 直接写到 cache 的偏移处：GEMM 的行距固定为 N，
// 而 cache 若为了 256 字节绑定对齐而加行填充就会错位；这里保持两者都紧凑、无填充。

struct Dims {
  rows: u32,
  rowLen: u32,
  pos0: u32,
  _p0: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.rows * dims.rowLen) {
    return;
  }
  let row = i / dims.rowLen;
  let col = i % dims.rowLen;
  dst[(dims.pos0 + row) * dims.rowLen + col] = src[i];
}
