// M0 冒烟测试：逐元素加法。用于验证 device / buffer / pipeline / 回读链路是否打通。
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) {
    return;
  }
  out[i] = a[i] + b[i];
}