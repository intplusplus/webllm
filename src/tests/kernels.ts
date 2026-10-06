import type { GpuContext } from '../gpu/device';
import { createComputePipeline, dispatch } from '../gpu/pipeline';
import {
  createStorageBuffer,
  createUniform,
  readbackF32,
  StructPacker,
  writeF16,
  writeF32,
  writeU32,
} from '../gpu/buffer';
import { checkTolerance, makeRng, randomF32, vecAddRef } from '../reference/cpu-ref';
import { attentionGqaRef, gemmNTRef, ropeHalfRef, siluMulRef } from '../reference/qwen-ref';
import * as ref from '../reference/ops';

import vecAddWgsl from '../gpu/kernels/misc/vec_add.wgsl?raw';
import gemmNtWgsl from '../gpu/kernels/gemm/gemm_nt.wgsl?raw';
import embeddingWgsl from '../gpu/kernels/embedding/embedding.wgsl?raw';
import rmsnormWgsl from '../gpu/kernels/norm/rmsnorm.wgsl?raw';
import layernormWgsl from '../gpu/kernels/norm/layernorm.wgsl?raw';
import addLayernormWgsl from '../gpu/kernels/norm/add_layernorm.wgsl?raw';
import geluWgsl from '../gpu/kernels/activation/gelu.wgsl?raw';
import ropeWgsl from '../gpu/kernels/rope/rope.wgsl?raw';
import softmaxWgsl from '../gpu/kernels/attention/softmax.wgsl?raw';
import attentionWgsl from '../gpu/kernels/attention/attention.wgsl?raw';
import gemmNnWgsl from '../gpu/kernels/gemm/gemm_nn.wgsl?raw';
import gemmTnWgsl from '../gpu/kernels/gemm/gemm_tn.wgsl?raw';
import gemmNtF16Wgsl from '../gpu/kernels/gemm/gemm_nt_f16.wgsl?raw';
import geluBwdWgsl from '../gpu/kernels/activation/gelu_bwd.wgsl?raw';
import ropeBwdWgsl from '../gpu/kernels/rope/rope_bwd.wgsl?raw';
import lnStatsWgsl from '../gpu/kernels/norm/ln_stats.wgsl?raw';
import lnDxWgsl from '../gpu/kernels/norm/ln_dx.wgsl?raw';
import lnDwdbWgsl from '../gpu/kernels/norm/ln_dwdb.wgsl?raw';
import sumRowsWgsl from '../gpu/kernels/misc/sum_rows.wgsl?raw';
import ceSoftmaxBwdWgsl from '../gpu/kernels/misc/ce_softmax_bwd.wgsl?raw';
import embeddingBwdWgsl from '../gpu/kernels/embedding/embedding_bwd.wgsl?raw';
import adamwWgsl from '../gpu/kernels/misc/adamw.wgsl?raw';
import attnBwdSoftmaxWgsl from '../gpu/kernels/attention/attn_bwd_softmax.wgsl?raw';
import attnBwdDqWgsl from '../gpu/kernels/attention/attn_bwd_dq.wgsl?raw';
import attnBwdDkdvWgsl from '../gpu/kernels/attention/attn_bwd_dkdv.wgsl?raw';
import ropeHalfWgsl from '../gpu/kernels/rope/rope_half.wgsl?raw';
import attentionGqaWgsl from '../gpu/kernels/attention/attention_gqa.wgsl?raw';
import siluMulWgsl from '../gpu/kernels/activation/silu_mul.wgsl?raw';
import embeddingF16Wgsl from '../gpu/kernels/embedding/embedding_f16.wgsl?raw';
import gemmGemvF16Wgsl from '../gpu/kernels/gemm/gemm_gemv_f16.wgsl?raw';
import gemmGemvSplitF16Wgsl from '../gpu/kernels/gemm/gemm_gemv_split_f16.wgsl?raw';
import gemmGemvSplitReduceWgsl from '../gpu/kernels/gemm/gemm_gemv_split_reduce.wgsl?raw';
import { testDecodeProfile } from './perf/profile-decode';

export interface TestResult {
  name: string;
  pass: boolean;
  detail: string;
}

export interface SelfTest {
  name: string;
  run: (gpu: GpuContext) => Promise<string>;
}

function assertClose(
  got: Float32Array,
  expected: Float32Array,
  tol: number,
  what: string,
): string {
  const r = checkTolerance(got, expected, 1e-5, tol);
  if (!r.ok) {
    throw new Error(
      `${what}: 超出容差 maxAbs = ${r.maxAbs.toExponential(3)} (atol=1e-5)，maxRel = ${r.maxRel.toExponential(3)} (rtol=${tol.toExponential(0)})`,
    );
  }
  return `maxAbs = ${r.maxAbs.toExponential(2)}, maxRel = ${r.maxRel.toExponential(2)}`;
}

/** M0 冒烟：逐元素加法。 */
async function testVecAdd(gpu: GpuContext): Promise<string> {
  const n = 1 << 16;
  const rng = makeRng(0x12345678);
  const a = randomF32(n, rng);
  const b = randomF32(n, rng);
  const device = gpu.device;

  const bufA = createStorageBuffer(device, n * 4, 'a');
  const bufB = createStorageBuffer(device, n * 4, 'b');
  const bufOut = createStorageBuffer(device, n * 4, 'out');
  writeF32(device, bufA, 0, a);
  writeF32(device, bufB, 0, b);

  const pipeline = createComputePipeline(device, vecAddWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA } },
      { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufOut } },
    ],
  });
  dispatch(device, pipeline, bg, Math.ceil(n / 64));

  const got = await readbackF32(device, bufOut, 0, n);
  const detail = assertClose(got, vecAddRef(a, b), 1e-5, 'vec_add');
  [bufA, bufB, bufOut].forEach((x) => x.destroy());
  return `n = ${n}，${detail}`;
}

/** GEMM：C[M,N] = A[M,K] @ B[N,K]^T + bias，含非整除尺寸以验证边界保护。 */
async function testGemm(gpu: GpuContext): Promise<string> {
  const M = 37;
  const N = 53;
  const K = 41;
  const rng = makeRng(11);
  const A = randomF32(M * K, rng);
  const B = randomF32(N * K, rng);
  const bias = randomF32(N, rng);
  const device = gpu.device;

  const bufA = createStorageBuffer(device, M * K * 4, 'A');
  const bufB = createStorageBuffer(device, N * K * 4, 'B');
  const bufC = createStorageBuffer(device, M * N * 4, 'C');
  const bufBias = createStorageBuffer(device, N * 4, 'bias');
  writeF32(device, bufA, 0, A);
  writeF32(device, bufB, 0, B);
  writeF32(device, bufBias, 0, bias);

  const u = new StructPacker();
  u.u32(M);
  u.u32(N);
  u.u32(K);
  u.u32(1);
  const bufDims = createUniform(device, u.bytes(), 'dims');

  const pipeline = createComputePipeline(device, gemmNtWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA } },
      { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufC } },
      { binding: 3, resource: { buffer: bufBias } },
      { binding: 4, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, [Math.ceil(M / 64), Math.ceil(N / 64), 1]);

  const got = await readbackF32(device, bufC, 0, M * N);
  const detail = assertClose(got, ref.gemmNTRef(A, B, M, N, K, bias), 1e-4, 'gemm_nt');
  [bufA, bufB, bufC, bufBias, bufDims].forEach((x) => x.destroy());
  return `M=${M} N=${N} K=${K}，${detail}`;
}

/** 词嵌入查表。 */
async function testEmbedding(gpu: GpuContext): Promise<string> {
  const vocab = 100;
  const D = 32;
  const T = 17;
  const rng = makeRng(22);
  const W = randomF32(vocab * D, rng);
  const tokens = new Uint32Array(T);
  for (let i = 0; i < T; i++) tokens[i] = Math.floor(rng() * vocab);
  const device = gpu.device;

  const bufW = createStorageBuffer(device, vocab * D * 4, 'W');
  const bufTok = createStorageBuffer(device, T * 4, 'tokens');
  const bufOut = createStorageBuffer(device, T * D * 4, 'out');
  writeF32(device, bufW, 0, W);
  writeU32(device, bufTok, 0, tokens);

  const u = new StructPacker();
  u.u32(T);
  u.u32(D);
  u.u32(0);
  u.u32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, embeddingWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufW } },
      { binding: 1, resource: { buffer: bufTok } },
      { binding: 2, resource: { buffer: bufOut } },
      { binding: 3, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, Math.ceil((T * D) / 64));

  const got = await readbackF32(device, bufOut, 0, T * D);
  const detail = assertClose(got, ref.embeddingRef(W, D, tokens), 1e-5, 'embedding');
  [bufW, bufTok, bufOut, bufDims].forEach((x) => x.destroy());
  return `vocab=${vocab} D=${D} T=${T}，${detail}`;
}

/** RMSNorm。 */
async function testRmsnorm(gpu: GpuContext): Promise<string> {
  const rows = 5;
  const D = 64;
  const eps = 1e-5;
  const rng = makeRng(33);
  const x = randomF32(rows * D, rng);
  const w = randomF32(D, rng);
  const device = gpu.device;

  const bufX = createStorageBuffer(device, rows * D * 4);
  const bufW = createStorageBuffer(device, D * 4);
  const bufY = createStorageBuffer(device, rows * D * 4);
  writeF32(device, bufX, 0, x);
  writeF32(device, bufW, 0, w);

  const u = new StructPacker();
  u.u32(rows);
  u.u32(D);
  u.f32(eps);
  u.u32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, rmsnormWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufX } },
      { binding: 1, resource: { buffer: bufW } },
      { binding: 2, resource: { buffer: bufY } },
      { binding: 3, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, rows);

  const got = await readbackF32(device, bufY, 0, rows * D);
  const detail = assertClose(got, ref.rmsnormRef(x, rows, D, w, eps), 1e-4, 'rmsnorm');
  [bufX, bufW, bufY, bufDims].forEach((x2) => x2.destroy());
  return `rows=${rows} D=${D}，${detail}`;
}

/** LayerNorm（含 bias）。 */
async function testLayernorm(gpu: GpuContext): Promise<string> {
  const rows = 5;
  const D = 64;
  const eps = 1e-5;
  const rng = makeRng(44);
  const x = randomF32(rows * D, rng);
  const w = randomF32(D, rng);
  const b = randomF32(D, rng);
  const device = gpu.device;

  const bufX = createStorageBuffer(device, rows * D * 4);
  const bufW = createStorageBuffer(device, D * 4);
  const bufB = createStorageBuffer(device, D * 4);
  const bufY = createStorageBuffer(device, rows * D * 4);
  writeF32(device, bufX, 0, x);
  writeF32(device, bufW, 0, w);
  writeF32(device, bufB, 0, b);

  const u = new StructPacker();
  u.u32(rows);
  u.u32(D);
  u.f32(eps);
  u.u32(1);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, layernormWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufX } },
      { binding: 1, resource: { buffer: bufW } },
      { binding: 2, resource: { buffer: bufB } },
      { binding: 3, resource: { buffer: bufY } },
      { binding: 4, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, rows);

  const got = await readbackF32(device, bufY, 0, rows * D);
  const detail = assertClose(got, ref.layernormRef(x, rows, D, w, b, eps, true), 1e-4, 'layernorm');
  [bufX, bufW, bufB, bufY, bufDims].forEach((x2) => x2.destroy());
  return `rows=${rows} D=${D}，${detail}`;
}

/** GELU（tanh 近似）。 */
async function testGelu(gpu: GpuContext): Promise<string> {
  const n = 4096;
  const rng = makeRng(55);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = (rng() * 2 - 1) * 5;
  const device = gpu.device;

  const bufX = createStorageBuffer(device, n * 4);
  const bufY = createStorageBuffer(device, n * 4);
  writeF32(device, bufX, 0, x);

  const u = new StructPacker();
  u.u32(n);
  u.u32(0);
  u.u32(0);
  u.u32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, geluWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufX } },
      { binding: 1, resource: { buffer: bufY } },
      { binding: 2, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, Math.ceil(n / 64));

  const got = await readbackF32(device, bufY, 0, n);
  const detail = assertClose(got, ref.geluRef(x), 1e-4, 'gelu');
  [bufX, bufY, bufDims].forEach((x2) => x2.destroy());
  return `n = ${n}，${detail}`;
}

/** RoPE。 */
async function testRope(gpu: GpuContext): Promise<string> {
  const B = 2;
  const T = 4;
  const H = 2;
  const D = 8;
  const rows = B * T;
  const base = 10000;
  const rng = makeRng(66);
  const x = randomF32(rows * H * D, rng);
  const device = gpu.device;

  const bufX = createStorageBuffer(device, x.length * 4);
  const bufOut = createStorageBuffer(device, x.length * 4);
  writeF32(device, bufX, 0, x);

  const u = new StructPacker();
  u.u32(rows);
  u.u32(H);
  u.u32(D);
  u.u32(T);
  u.f32(base);
  u.f32(0);
  u.f32(0);
  u.f32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, ropeWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufX } },
      { binding: 1, resource: { buffer: bufOut } },
      { binding: 2, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, Math.ceil((rows * H * (D / 2)) / 64));

  const got = await readbackF32(device, bufOut, 0, x.length);
  const detail = assertClose(got, ref.ropeRef(x, rows, H, D, T, base), 1e-4, 'rope');
  [bufX, bufOut, bufDims].forEach((x2) => x2.destroy());
  return `B=${B} T=${T} H=${H} D=${D}，${detail}`;
}

/** 行内 softmax。 */
async function testSoftmax(gpu: GpuContext): Promise<string> {
  const rows = 3;
  const D = 50;
  const rng = makeRng(77);
  const x = randomF32(rows * D, rng);
  const device = gpu.device;

  const bufX = createStorageBuffer(device, rows * D * 4);
  const bufY = createStorageBuffer(device, rows * D * 4);
  writeF32(device, bufX, 0, x);

  const u = new StructPacker();
  u.u32(rows);
  u.u32(D);
  u.u32(0);
  u.u32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, softmaxWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufX } },
      { binding: 1, resource: { buffer: bufY } },
      { binding: 2, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, rows);

  const got = await readbackF32(device, bufY, 0, rows * D);
  const detail = assertClose(got, ref.softmaxRef(x, rows, D), 1e-4, 'softmax');
  [bufX, bufY, bufDims].forEach((x2) => x2.destroy());
  return `rows=${rows} D=${D}，${detail}`;
}

/** 因果自注意力（前向）。 */
async function testAttention(gpu: GpuContext): Promise<string> {
  const B = 1;
  const T = 6;
  const H = 2;
  const D = 8;
  const n = B * T * H * D;
  const rng = makeRng(88);
  const q = randomF32(n, rng);
  const k = randomF32(n, rng);
  const v = randomF32(n, rng);
  const device = gpu.device;

  const bufQ = createStorageBuffer(device, n * 4);
  const bufK = createStorageBuffer(device, n * 4);
  const bufV = createStorageBuffer(device, n * 4);
  const bufOut = createStorageBuffer(device, n * 4);
  writeF32(device, bufQ, 0, q);
  writeF32(device, bufK, 0, k);
  writeF32(device, bufV, 0, v);

  const u = new StructPacker();
  u.u32(B);
  u.u32(T);
  u.u32(H);
  u.u32(D);
  u.f32(1 / Math.sqrt(D));
  u.f32(0);
  u.f32(0);
  u.f32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, attentionWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufQ } },
      { binding: 1, resource: { buffer: bufK } },
      { binding: 2, resource: { buffer: bufV } },
      { binding: 3, resource: { buffer: bufOut } },
      { binding: 4, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, [T, H, B]);

  const got = await readbackF32(device, bufOut, 0, n);
  const detail = assertClose(got, ref.attentionRef(q, k, v, B, T, H, D), 1e-4, 'attention');
  [bufQ, bufK, bufV, bufOut, bufDims].forEach((x) => x.destroy());
  return `B=${B} T=${T} H=${H} D=${D}，${detail}`;
}

/** GEMM NN：C[M,N] = A[M,K] @ B[K,N]。 */
async function testGemmNn(gpu: GpuContext): Promise<string> {
  const M = 37;
  const N = 53;
  const K = 41;
  const rng = makeRng(111);
  const A = randomF32(M * K, rng);
  const B = randomF32(K * N, rng);
  const device = gpu.device;

  const bufA = createStorageBuffer(device, M * K * 4);
  const bufB = createStorageBuffer(device, K * N * 4);
  const bufC = createStorageBuffer(device, M * N * 4);
  const bufBias = createStorageBuffer(device, N * 4);
  writeF32(device, bufA, 0, A);
  writeF32(device, bufB, 0, B);

  const u = new StructPacker();
  u.u32(M);
  u.u32(N);
  u.u32(K);
  u.u32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, gemmNnWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA } },
      { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufC } },
      { binding: 3, resource: { buffer: bufBias } },
      { binding: 4, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, [Math.ceil(M / 64), Math.ceil(N / 64), 1]);

  const got = await readbackF32(device, bufC, 0, M * N);
  const detail = assertClose(got, ref.gemmNNRef(A, B, M, N, K), 1e-4, 'gemm_nn');
  [bufA, bufB, bufC, bufBias, bufDims].forEach((x) => x.destroy());
  return `M=${M} N=${N} K=${K}，${detail}`;
}

/** GEMM TN：C[M,N] = A[K,M]^T @ B[K,N]。 */
async function testGemmTn(gpu: GpuContext): Promise<string> {
  const M = 37;
  const N = 53;
  const K = 41;
  const rng = makeRng(222);
  const A = randomF32(K * M, rng);
  const B = randomF32(K * N, rng);
  const device = gpu.device;

  const bufA = createStorageBuffer(device, K * M * 4);
  const bufB = createStorageBuffer(device, K * N * 4);
  const bufC = createStorageBuffer(device, M * N * 4);
  const bufBias = createStorageBuffer(device, N * 4);
  writeF32(device, bufA, 0, A);
  writeF32(device, bufB, 0, B);

  const u = new StructPacker();
  u.u32(M);
  u.u32(N);
  u.u32(K);
  u.u32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, gemmTnWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA } },
      { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufC } },
      { binding: 3, resource: { buffer: bufBias } },
      { binding: 4, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, [Math.ceil(M / 64), Math.ceil(N / 64), 1]);

  const got = await readbackF32(device, bufC, 0, M * N);
  const detail = assertClose(got, ref.gemmTNRef(A, B, M, N, K), 1e-4, 'gemm_tn');
  [bufA, bufB, bufC, bufBias, bufDims].forEach((x) => x.destroy());
  return `M=${M} N=${N} K=${K}，${detail}`;
}

/** f32 → f16 舍入（优先浏览器原生 Math.f16round）。 */
function roundF16(x: number): number {
  const fn = (Math as unknown as { f16round?: (v: number) => number }).f16round;
  return fn ? fn(x) : x;
}

/** M3：f16 输入 GEMM。除正确性外，还报告 fp32 vs f16 GEMM 吞吐。 */
async function testGemmNtF16(gpu: GpuContext): Promise<string> {
  if (!gpu.features.has('shader-f16')) {
    return 'shader-f16 不可用，跳过 f16 GEMM 验证';
  }

  const M = 37;
  const N = 53;
  const K = 41;
  const rng = makeRng(333);
  const A = randomF32(M * K, rng);
  const B = randomF32(N * K, rng);
  const bias = randomF32(N, rng);
  const device = gpu.device;

  const bufA32 = createStorageBuffer(device, M * K * 4, 'A32');
  const bufB32 = createStorageBuffer(device, N * K * 4, 'B32');
  const bufA16 = createStorageBuffer(device, M * K * 2, 'A16');
  const bufB16 = createStorageBuffer(device, N * K * 2, 'B16');
  const bufC = createStorageBuffer(device, M * N * 4, 'C');
  const bufBias = createStorageBuffer(device, N * 4, 'bias');
  const bufDims = createUniform(device, new StructPacker()
    .u32(M)
    .u32(N)
    .u32(K)
    .u32(1)
    .bytes());

  writeF32(device, bufA32, 0, A);
  writeF32(device, bufB32, 0, B);
  writeF16(device, bufA16, 0, A);
  writeF16(device, bufB16, 0, B);
  writeF32(device, bufBias, 0, bias);

  const p32 = createComputePipeline(device, gemmNtWgsl, 'main', 'gemm_nt_f32');
  const p16 = createComputePipeline(device, gemmNtF16Wgsl, 'main', 'gemm_nt_f16');
  const bg16 = device.createBindGroup({
    layout: p16.getBindGroupLayout(0),
    entries: [bufA16, bufB16, bufC, bufBias, bufDims].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  });
  const workgroups: [number, number, number] = [Math.ceil(M / 64), Math.ceil(N / 64), 1];

  dispatch(device, p16, bg16, workgroups);
  const got = await readbackF32(device, bufC, 0, M * N);
  const aF16 = Float32Array.from(A, roundF16);
  const bF16 = Float32Array.from(B, roundF16);
  const detail = assertClose(got, ref.gemmNTRef(aF16, bF16, M, N, K, bias), 5e-3, 'gemm_nt_f16');

  // 小矩阵回读完成后，用更大形状做基准，避免结果被 dispatch 开销主导。
  const bM = 512;
  const bN = 512;
  const bK = 256;
  const iterations = 20;
  const workgroups32: [number, number, number] = [Math.ceil(bM / 64), Math.ceil(bN / 64), 1];

  const bA32 = createStorageBuffer(device, bM * bK * 4, 'benchA32');
  const bB32 = createStorageBuffer(device, bN * bK * 4, 'benchB32');
  const bA16 = createStorageBuffer(device, bM * bK * 2, 'benchA16');
  const bB16 = createStorageBuffer(device, bN * bK * 2, 'benchB16');
  const bC = createStorageBuffer(device, bM * bN * 4, 'benchC');
  const bDims = createUniform(device, new StructPacker().u32(bM).u32(bN).u32(bK).u32(0).bytes());

  const benchRng = makeRng(444);
  const bA = randomF32(bM * bK, benchRng);
  const bB = randomF32(bN * bK, benchRng);
  writeF32(device, bA32, 0, bA);
  writeF32(device, bB32, 0, bB);
  writeF16(device, bA16, 0, bA);
  writeF16(device, bB16, 0, bB);

  const bgBench32 = device.createBindGroup({
    layout: p32.getBindGroupLayout(0),
    entries: [bA32, bB32, bC, bufBias, bDims].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  });
  const bgBench16 = device.createBindGroup({
    layout: p16.getBindGroupLayout(0),
    entries: [bA16, bB16, bC, bufBias, bDims].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  });

  const flops = 2 * bM * bN * bK * iterations;
  const t32 = performance.now();
  for (let i = 0; i < iterations; i++) dispatch(device, p32, bgBench32, workgroups32);
  await device.queue.onSubmittedWorkDone();
  const dt32 = Math.max(1e-6, (performance.now() - t32) / 1000);

  const t16 = performance.now();
  for (let i = 0; i < iterations; i++) dispatch(device, p16, bgBench16, workgroups32);
  await device.queue.onSubmittedWorkDone();
  const dt16 = Math.max(1e-6, (performance.now() - t16) / 1000);

  const gflops32 = flops / dt32 / 1e9;
  const gflops16 = flops / dt16 / 1e9;
  const speedup = dt32 / dt16;

  [bufA32, bufB32, bufA16, bufB16, bufC, bufBias, bufDims].forEach((x) => x.destroy());
  [bA32, bB32, bA16, bB16, bC, bDims].forEach((x) => x.destroy());
  return `M=${M} N=${N} K=${K}，${detail}；bench M=${bM} N=${bN} K=${bK} iters=${iterations}：fp32 ${gflops32.toFixed(1)} GFLOP/s，f16 ${gflops16.toFixed(1)} GFLOP/s，speedup ${speedup.toFixed(2)}x`;
}

/** GELU 反向。 */
async function testGeluBwd(gpu: GpuContext): Promise<string> {
  const n = 4096;
  const rng = makeRng(133);
  const x = new Float32Array(n);
  const dy = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = (rng() * 2 - 1) * 5;
    dy[i] = rng() * 2 - 1;
  }
  const device = gpu.device;

  const bufX = createStorageBuffer(device, n * 4);
  const bufDy = createStorageBuffer(device, n * 4);
  const bufDx = createStorageBuffer(device, n * 4);
  writeF32(device, bufX, 0, x);
  writeF32(device, bufDy, 0, dy);

  const u = new StructPacker();
  u.u32(n);
  u.u32(0);
  u.u32(0);
  u.u32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, geluBwdWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufX } },
      { binding: 1, resource: { buffer: bufDy } },
      { binding: 2, resource: { buffer: bufDx } },
      { binding: 3, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, Math.ceil(n / 64));

  const got = await readbackF32(device, bufDx, 0, n);
  const detail = assertClose(got, ref.geluBwdRef(x, dy), 1e-4, 'gelu_bwd');
  [bufX, bufDy, bufDx, bufDims].forEach((x2) => x2.destroy());
  return `n = ${n}，${detail}`;
}

/** RoPE 反向。 */
async function testRopeBwd(gpu: GpuContext): Promise<string> {
  const B = 2;
  const T = 4;
  const H = 2;
  const D = 8;
  const rows = B * T;
  const base = 10000;
  const rng = makeRng(144);
  const d = randomF32(rows * H * D, rng);
  const device = gpu.device;

  const bufD = createStorageBuffer(device, d.length * 4);
  const bufOut = createStorageBuffer(device, d.length * 4);
  writeF32(device, bufD, 0, d);

  const u = new StructPacker();
  u.u32(rows);
  u.u32(H);
  u.u32(D);
  u.u32(T);
  u.f32(base);
  u.f32(0);
  u.f32(0);
  u.f32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, ropeBwdWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufD } },
      { binding: 1, resource: { buffer: bufOut } },
      { binding: 2, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, Math.ceil((rows * H * (D / 2)) / 64));

  const got = await readbackF32(device, bufOut, 0, d.length);
  const detail = assertClose(got, ref.ropeBwdRef(d, rows, H, D, T, base), 1e-4, 'rope_bwd');
  [bufD, bufOut, bufDims].forEach((x2) => x2.destroy());
  return `B=${B} T=${T} H=${H} D=${D}，${detail}`;
}

/** LayerNorm 反向（stats + dx + dw/db 三 kernel 串联）。 */
async function testLayernormBwd(gpu: GpuContext): Promise<string> {
  const rows = 5;
  const D = 64;
  const eps = 1e-5;
  const rng = makeRng(155);
  const x = randomF32(rows * D, rng);
  const dy = randomF32(rows * D, rng);
  const w = randomF32(D, rng);
  const device = gpu.device;
  const f = (n: number) => createStorageBuffer(device, n * 4);
  const [bufX, bufDy, bufW, bufStats, bufDx, bufDw, bufDb] = [
    f(rows * D), f(rows * D), f(D), f(rows * 4), f(rows * D), f(D), f(D),
  ];
  writeF32(device, bufX, 0, x);
  writeF32(device, bufDy, 0, dy);
  writeF32(device, bufW, 0, w);

  const u1 = new StructPacker();
  u1.u32(rows); u1.u32(D); u1.f32(eps); u1.u32(0);
  const d1 = createUniform(device, u1.bytes());
  const p1 = createComputePipeline(device, lnStatsWgsl);
  dispatch(device, p1, device.createBindGroup({
    layout: p1.getBindGroupLayout(0),
    entries: [bufX, bufDy, bufW, bufStats, d1].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  }), rows);

  const u2 = new StructPacker();
  u2.u32(rows); u2.u32(D); u2.u32(0); u2.u32(0);
  const d2 = createUniform(device, u2.bytes());
  const p2 = createComputePipeline(device, lnDxWgsl);
  dispatch(device, p2, device.createBindGroup({
    layout: p2.getBindGroupLayout(0),
    entries: [bufX, bufDy, bufW, bufStats, bufDx, d2].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  }), Math.ceil((rows * D) / 64));

  const p3 = createComputePipeline(device, lnDwdbWgsl);
  dispatch(device, p3, device.createBindGroup({
    layout: p3.getBindGroupLayout(0),
    entries: [bufX, bufDy, bufStats, bufDw, bufDb, d2].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  }), D);

  const r = ref.layernormBwdRef(x, rows, D, w, dy, eps);
  const gotDx = await readbackF32(device, bufDx, 0, rows * D);
  const gotDw = await readbackF32(device, bufDw, 0, D);
  const gotDb = await readbackF32(device, bufDb, 0, D);
  const detail =
    'dx ' + assertClose(gotDx, r.dx, 1e-4, 'ln dx') +
    '; dw ' + assertClose(gotDw, r.dw, 1e-4, 'ln dw') +
    '; db ' + assertClose(gotDb, r.db, 1e-4, 'ln db');
  [bufX, bufDy, bufW, bufStats, bufDx, bufDw, bufDb, d1, d2].forEach((b) => b.destroy());
  return `rows=${rows} D=${D}，${detail}`;
}

/** add_layernorm：一个 kernel 同时输出 residual 与 LayerNorm 结果。 */
async function testAddLayernorm(gpu: GpuContext): Promise<string> {
  const rows = 5;
  const D = 64;
  const eps = 1e-5;
  const rng = makeRng(266);
  const a = randomF32(rows * D, rng);
  const b = randomF32(rows * D, rng);
  const w = randomF32(D, rng);
  const bias = randomF32(D, rng);
  const device = gpu.device;

  const bufA = createStorageBuffer(device, rows * D * 4);
  const bufB = createStorageBuffer(device, rows * D * 4);
  const bufW = createStorageBuffer(device, D * 4);
  const bufBias = createStorageBuffer(device, D * 4);
  const bufY = createStorageBuffer(device, rows * D * 4);
  const bufR = createStorageBuffer(device, rows * D * 4);
  writeF32(device, bufA, 0, a);
  writeF32(device, bufB, 0, b);
  writeF32(device, bufW, 0, w);
  writeF32(device, bufBias, 0, bias);

  const u = new StructPacker();
  u.u32(rows);
  u.u32(D);
  u.f32(eps);
  u.u32(1);
  const d = createUniform(device, u.bytes());

  const p = createComputePipeline(device, addLayernormWgsl);
  dispatch(device, p, device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: [bufA, bufB, bufW, bufBias, bufY, bufR, d].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  }), rows);

  const residual = vecAddRef(a, b);
  const gotR = await readbackF32(device, bufR, 0, rows * D);
  const gotY = await readbackF32(device, bufY, 0, rows * D);
  const detail =
    'residual ' + assertClose(gotR, residual, 1e-4, 'add_ln residual') +
    '; y ' + assertClose(gotY, ref.layernormRef(residual, rows, D, w, bias, eps, true), 1e-4, 'add_ln y');
  [bufA, bufB, bufW, bufBias, bufY, bufR, d].forEach((b) => b.destroy());
  return `rows=${rows} D=${D}，${detail}`;
}

/** 对行求和（bias 梯度）。 */
async function testSumRows(gpu: GpuContext): Promise<string> {
  const rows = 7;
  const cols = 13;
  const rng = makeRng(166);
  const x = randomF32(rows * cols, rng);
  const device = gpu.device;
  const bufX = createStorageBuffer(device, rows * cols * 4);
  const bufOut = createStorageBuffer(device, cols * 4);
  writeF32(device, bufX, 0, x);
  const u = new StructPacker();
  u.u32(rows); u.u32(cols); u.u32(0); u.u32(0);
  const d = createUniform(device, u.bytes());
  const p = createComputePipeline(device, sumRowsWgsl);
  dispatch(device, p, device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: [bufX, bufOut, d].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  }), cols);
  const got = await readbackF32(device, bufOut, 0, cols);
  const detail = assertClose(got, ref.sumRowsRef(x, rows, cols), 1e-4, 'sum_rows');
  [bufX, bufOut, d].forEach((b) => b.destroy());
  return `rows=${rows} cols=${cols}，${detail}`;
}

/** softmax + 交叉熵反向（支持逐行权重：预训练 w≡1/M，SFT/DPO 用 0 与 ±β 系数）。 */
async function testCeSoftmaxBwd(gpu: GpuContext): Promise<string> {
  const M = 4;
  const V = 11;
  const rng = makeRng(177);
  const logits = randomF32(M * V, rng);
  const targets = new Uint32Array(M);
  for (let i = 0; i < M; i++) targets[i] = Math.floor(rng() * V);
  // 预训练语义：w ≡ 1/M（平均 CE）；另验证 w=0 的 masked 行输出全 0
  const weights = new Float32Array(M).fill(1 / M);
  weights[2] = 0.0;
  const device = gpu.device;
  const bufL = createStorageBuffer(device, M * V * 4);
  const bufT = createStorageBuffer(device, M * 4);
  const bufW = createStorageBuffer(device, M * 4);
  const bufOut = createStorageBuffer(device, M * V * 4);
  writeF32(device, bufL, 0, logits);
  writeU32(device, bufT, 0, targets);
  writeF32(device, bufW, 0, weights);
  const u = new StructPacker();
  u.u32(M); u.u32(V); u.u32(0); u.u32(0);
  const d = createUniform(device, u.bytes());
  const p = createComputePipeline(device, ceSoftmaxBwdWgsl);
  dispatch(device, p, device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: [bufL, bufT, bufW, bufOut, d].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  }), M);
  const got = await readbackF32(device, bufOut, 0, M * V);
  const want = ref.ceSoftmaxBwdRef(logits, M, V, targets);
  for (let v = 0; v < V; v++) want[2 * V + v] = 0.0; // masked 行
  const detail = assertClose(got, want, 1e-4, 'ce_bwd');
  [bufL, bufT, bufW, bufOut, d].forEach((b) => b.destroy());
  return `M=${M} V=${V}（含 masked 行），${detail}`;
}

/** 词嵌入反向（累加到已有 dW）。 */
async function testEmbeddingBwd(gpu: GpuContext): Promise<string> {
  const rows = 6;
  const D = 8;
  const vocab = 10;
  const rng = makeRng(188);
  const tokens = new Uint32Array(rows);
  for (let i = 0; i < rows; i++) tokens[i] = Math.floor(rng() * vocab);
  const dx = randomF32(rows * D, rng);
  const init = randomF32(vocab * D, rng);
  const device = gpu.device;
  const bufTok = createStorageBuffer(device, rows * 4);
  const bufDx = createStorageBuffer(device, rows * D * 4);
  const bufDW = createStorageBuffer(device, vocab * D * 4);
  writeU32(device, bufTok, 0, tokens);
  writeF32(device, bufDx, 0, dx);
  writeF32(device, bufDW, 0, init);
  const u = new StructPacker();
  u.u32(rows); u.u32(D); u.u32(vocab); u.u32(0);
  const d = createUniform(device, u.bytes());
  const p = createComputePipeline(device, embeddingBwdWgsl);
  dispatch(device, p, device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: [bufTok, bufDx, bufDW, d].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  }), vocab);
  const got = await readbackF32(device, bufDW, 0, vocab * D);
  const expected = ref.embeddingBwdRef(tokens, dx, vocab, D);
  for (let i = 0; i < expected.length; i++) expected[i] += init[i];
  const detail = assertClose(got, expected, 1e-4, 'embedding_bwd');
  [bufTok, bufDx, bufDW, d].forEach((b) => b.destroy());
  return `rows=${rows} D=${D} vocab=${vocab}，${detail}`;
}

/** AdamW 单步。 */
async function testAdamw(gpu: GpuContext): Promise<string> {
  const n = 8;
  const rng = makeRng(199);
  const param = randomF32(n, rng);
  const grad = randomF32(n, rng);
  const m0 = randomF32(n, rng);
  const v0 = randomF32(n, rng);
  const opts = { lr: 0.01, b1: 0.9, b2: 0.999, eps: 1e-8, wd: 0.01, t: 3 };
  const device = gpu.device;
  const bufP = createStorageBuffer(device, n * 4);
  const bufG = createStorageBuffer(device, n * 4);
  const bufM = createStorageBuffer(device, n * 4);
  const bufV = createStorageBuffer(device, n * 4);
  writeF32(device, bufP, 0, param);
  writeF32(device, bufG, 0, grad);
  writeF32(device, bufM, 0, m0);
  writeF32(device, bufV, 0, v0);
  const u = new StructPacker();
  u.u32(n); u.u32(0); u.u32(0); u.u32(0);
  u.f32(opts.lr); u.f32(opts.b1); u.f32(opts.b2); u.f32(opts.eps);
  u.f32(opts.wd); u.f32(Math.pow(opts.b1, opts.t)); u.f32(Math.pow(opts.b2, opts.t)); u.f32(0);
  const d = createUniform(device, u.bytes());
  const p = createComputePipeline(device, adamwWgsl);
  dispatch(device, p, device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: [bufP, bufG, bufM, bufV, d].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  }), Math.ceil(n / 64));
  const got = await readbackF32(device, bufP, 0, n);
  const ep = Float32Array.from(param);
  ref.adamwRef(ep, grad, Float32Array.from(m0), Float32Array.from(v0), opts);
  const detail = assertClose(got, ep, 1e-5, 'adamw');
  [bufP, bufG, bufM, bufV, d].forEach((b) => b.destroy());
  return `n=${n} t=${opts.t}，${detail}`;
}

/** 注意力反向（softmax + dq + dk/dv 三 kernel 串联）。 */
async function testAttentionBwd(gpu: GpuContext): Promise<string> {
  const B = 2;
  const T = 5;
  const H = 2;
  const D = 8;
  const n = B * T * H * D;
  const rng = makeRng(211);
  const q = randomF32(n, rng);
  const k = randomF32(n, rng);
  const v = randomF32(n, rng);
  const dout = randomF32(n, rng);
  const device = gpu.device;
  const f = (len: number) => createStorageBuffer(device, len * 4);
  const bufQ = f(n), bufK = f(n), bufV = f(n), bufDout = f(n);
  const bufPs = f(B * H * T * T), bufDs = f(B * H * T * T);
  const bufDq = f(n), bufDk = f(n), bufDv = f(n);
  writeF32(device, bufQ, 0, q);
  writeF32(device, bufK, 0, k);
  writeF32(device, bufV, 0, v);
  writeF32(device, bufDout, 0, dout);

  const u = new StructPacker();
  u.u32(B); u.u32(T); u.u32(H); u.u32(D);
  u.f32(1 / Math.sqrt(D)); u.f32(0); u.f32(0); u.f32(0);
  const dims = createUniform(device, u.bytes());

  const pS = createComputePipeline(device, attnBwdSoftmaxWgsl);
  dispatch(device, pS, device.createBindGroup({
    layout: pS.getBindGroupLayout(0),
    entries: [bufQ, bufK, bufV, bufDout, bufPs, bufDs, dims].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  }), [T, H, B]);

  const pQ = createComputePipeline(device, attnBwdDqWgsl);
  dispatch(device, pQ, device.createBindGroup({
    layout: pQ.getBindGroupLayout(0),
    entries: [bufDs, bufK, bufDq, dims].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  }), [T, H, B]);

  const pKV = createComputePipeline(device, attnBwdDkdvWgsl);
  dispatch(device, pKV, device.createBindGroup({
    layout: pKV.getBindGroupLayout(0),
    entries: [bufDs, bufPs, bufQ, bufDout, bufDk, bufDv, dims].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  }), [T, H, B]);

  const r = ref.attentionBwdRef(q, k, v, dout, B, T, H, D);
  const gotDq = await readbackF32(device, bufDq, 0, n);
  const gotDk = await readbackF32(device, bufDk, 0, n);
  const gotDv = await readbackF32(device, bufDv, 0, n);
  const detail =
    'dq ' + assertClose(gotDq, r.dq, 1e-4, 'attn dq') +
    '; dk ' + assertClose(gotDk, r.dk, 1e-4, 'attn dk') +
    '; dv ' + assertClose(gotDv, r.dv, 1e-4, 'attn dv');
  [bufQ, bufK, bufV, bufDout, bufPs, bufDs, bufDq, bufDk, bufDv].forEach((b) => b.destroy());
  dims.destroy();
  return `B=${B} T=${T} H=${H} D=${D}，${detail}`;
}

/** M4-3：Qwen2 风格 half-split RoPE（对应 HF 的 rotate_half），含 KV cache 的位置偏移。 */
async function testRopeHalf(gpu: GpuContext): Promise<string> {
  const B = 2;
  const T = 5;
  const H = 3;
  const D = 8;
  const rows = B * T;
  const base = 1e6;
  const rng = makeRng(0xa1);
  const x = randomF32(rows * H * D, rng);
  const device = gpu.device;

  const bufX = createStorageBuffer(device, x.length * 4);
  const bufOut = createStorageBuffer(device, x.length * 4);
  writeF32(device, bufX, 0, x);

  const pipeline = createComputePipeline(device, ropeHalfWgsl);
  const lines: string[] = [];

  // posOffset=0 走普通路径；posOffset=7 模拟增量解码（单行、绝对位置偏移）
  for (const posOffset of [0, 7]) {
    const u = new StructPacker();
    u.u32(rows);
    u.u32(H);
    u.u32(D);
    u.u32(T);
    u.f32(base);
    u.u32(posOffset);
    u.u32(0);
    u.u32(0);
    const bufDims = createUniform(device, u.bytes());

    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufX } },
        { binding: 1, resource: { buffer: bufOut } },
        { binding: 2, resource: { buffer: bufDims } },
      ],
    });
    dispatch(device, pipeline, bg, Math.ceil((rows * H * (D / 2)) / 64));

    const got = await readbackF32(device, bufOut, 0, x.length);
    const detail = assertClose(
      got,
      ropeHalfRef(x, rows, H, D, T, base, posOffset),
      1e-4,
      `rope_half(posOffset=${posOffset})`,
    );
    lines.push(`offset=${posOffset}: ${detail}`);
    bufDims.destroy();
  }

  [bufX, bufOut].forEach((b) => b.destroy());
  return `B=${B} T=${T} nHead=${H} D=${D} base=1e6（half-split）；${lines.join(' | ')}`;
}

/** M4-3：GQA 因果注意力（query 头数 > kv 头数）。 */
async function testAttentionGqa(gpu: GpuContext): Promise<string> {
  const B = 2;
  const T = 5;
  const H = 4;
  const KV = 2;
  const D = 8;
  const rows = B * T;
  const rng = makeRng(0xa2);
  const q = randomF32(rows * H * D, rng);
  const k = randomF32(rows * KV * D, rng);
  const v = randomF32(rows * KV * D, rng);
  const device = gpu.device;

  const bufQ = createStorageBuffer(device, q.length * 4);
  const bufK = createStorageBuffer(device, k.length * 4);
  const bufV = createStorageBuffer(device, v.length * 4);
  const bufOut = createStorageBuffer(device, q.length * 4);
  writeF32(device, bufQ, 0, q);
  writeF32(device, bufK, 0, k);
  writeF32(device, bufV, 0, v);

  const u = new StructPacker();
  u.u32(B);
  u.u32(T);
  u.u32(H);
  u.u32(KV);
  u.u32(D);
  u.f32(1 / Math.sqrt(D));
  u.f32(0);
  u.f32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, attentionGqaWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufQ } },
      { binding: 1, resource: { buffer: bufK } },
      { binding: 2, resource: { buffer: bufV } },
      { binding: 3, resource: { buffer: bufOut } },
      { binding: 4, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, [T, H, B]);

  const got = await readbackF32(device, bufOut, 0, q.length);
  const detail = assertClose(got, attentionGqaRef(q, k, v, B, T, H, KV, D), 1e-4, 'attention_gqa');
  [bufQ, bufK, bufV, bufOut, bufDims].forEach((b) => b.destroy());
  return `B=${B} T=${T} nHead=${H} nKvHead=${KV} D=${D}（1 个 kv head 服务 ${H / KV} 个 query head），${detail}`;
}

/** M4-3：SwiGLU 融合激活 out = silu(gate) * up。 */
async function testSiluMul(gpu: GpuContext): Promise<string> {
  const n = 4096;
  const rng = makeRng(0xa3);
  const gate = randomF32(n, rng);
  const up = randomF32(n, rng);
  const device = gpu.device;

  const bufGate = createStorageBuffer(device, n * 4);
  const bufUp = createStorageBuffer(device, n * 4);
  const bufOut = createStorageBuffer(device, n * 4);
  writeF32(device, bufGate, 0, gate);
  writeF32(device, bufUp, 0, up);

  const u = new StructPacker();
  u.u32(n);
  u.u32(0);
  u.u32(0);
  u.u32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, siluMulWgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufGate } },
      { binding: 1, resource: { buffer: bufUp } },
      { binding: 2, resource: { buffer: bufOut } },
      { binding: 3, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, Math.ceil(n / 64));

  const got = await readbackF32(device, bufOut, 0, n);
  const detail = assertClose(got, siluMulRef(gate, up), 1e-4, 'silu_mul');
  [bufGate, bufUp, bufOut, bufDims].forEach((b) => b.destroy());
  return `n = ${n}，${detail}`;
}

/** M4-3：f16 词嵌入查表（Qwen 词表 15 万行，f16 常驻省一半显存）。 */
async function testEmbeddingF16(gpu: GpuContext): Promise<string> {
  if (!gpu.features.has('shader-f16')) return 'shader-f16 不可用，跳过 f16 embedding 验证';
  const vocab = 100;
  const rows = 17;
  const D = 32;
  const rng = makeRng(0xa4);
  // 先按 f16 舍入，保证 GPU 侧 f16 存储无损、可与 CPU 精确对拍
  const table = Float32Array.from(randomF32(vocab * D, rng), roundF16);
  const tokens = new Uint32Array(rows);
  for (let i = 0; i < rows; i++) tokens[i] = Math.floor(rng() * vocab);
  const device = gpu.device;

  const bufW = createStorageBuffer(device, table.length * 2);
  const bufTok = createStorageBuffer(device, rows * 4);
  const bufOut = createStorageBuffer(device, rows * D * 4);
  writeF16(device, bufW, 0, table);
  writeU32(device, bufTok, 0, tokens);

  const u = new StructPacker();
  u.u32(rows);
  u.u32(D);
  u.u32(0);
  u.u32(0);
  const bufDims = createUniform(device, u.bytes());

  const pipeline = createComputePipeline(device, embeddingF16Wgsl);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufW } },
      { binding: 1, resource: { buffer: bufTok } },
      { binding: 2, resource: { buffer: bufOut } },
      { binding: 3, resource: { buffer: bufDims } },
    ],
  });
  dispatch(device, pipeline, bg, Math.ceil((rows * D) / 64));

  const want = new Float32Array(rows * D);
  for (let r = 0; r < rows; r++) {
    for (let d = 0; d < D; d++) want[r * D + d] = table[tokens[r] * D + d];
  }
  const got = await readbackF32(device, bufOut, 0, rows * D);
  const detail = assertClose(got, want, 1e-6, 'embedding_f16');
  [bufW, bufTok, bufOut, bufDims].forEach((b) => b.destroy());
  return `vocab=${vocab} rows=${rows} D=${D}，${detail}`;
}

/**
 * M5：解码专用 GEMV（M=1）+ 与通用 GEMM 的性能对比。
 *
 * 正确性用超出 4 倍展开尾部的 K（41）与常见形状（896）两类覆盖；
 * 性能对比取真实形状 gate_proj（N=4864, K=896），报告每步耗时与等效带宽。
 */
async function testGemmGemvF16(gpu: GpuContext): Promise<string> {
  if (!gpu.features.has('shader-f16')) return 'shader-f16 不可用，跳过 GEMV 验证';
  const device = gpu.device;
  const gemv = createComputePipeline(device, gemmGemvF16Wgsl);
  const tiled = createComputePipeline(device, gemmNtF16Wgsl);

  const runCase = async (
    pipeline: GPUComputePipeline,
    A: Float32Array,
    B: Float32Array,
    N: number,
    K: number,
    hasBias: boolean,
    grid: number | readonly number[],
    bias?: Float32Array,
  ): Promise<{ out: Float32Array; bufC: GPUBuffer; bg: GPUBindGroup }> => {
    const bufA = createStorageBuffer(device, A.length * 4);
    const bufB = createStorageBuffer(device, B.length * 2);
    const bufC = createStorageBuffer(device, N * 4);
    const bufBias = createStorageBuffer(device, Math.max(N, 1) * 4);
    writeF32(device, bufA, 0, A);
    writeF16(device, bufB, 0, B);
    if (bias) writeF32(device, bufBias, 0, bias);

    const u = new StructPacker();
    u.u32(1);
    u.u32(N);
    u.u32(K);
    u.u32(hasBias ? 1 : 0);
    const bufDims = createUniform(device, u.bytes());

    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufC } },
        { binding: 3, resource: { buffer: bufBias } },
        { binding: 4, resource: { buffer: bufDims } },
      ],
    });
    dispatch(device, pipeline, bg, grid);
    const out = await readbackF32(device, bufC, 0, N);
    return { out, bufC, bg };
  };

  // --- 1) 正确性：K=42（覆盖 u32 主循环之后的尾部）与 K=896（4 的倍数，与真实形状一致） ---
  const lines: string[] = [];
  for (const { N, K } of [{ N: 53, K: 42 }, { N: 64, K: 896 }]) {
    const rng = makeRng(0x9e00 + N);
    const A = randomF32(K, rng);
    const Bf = Float32Array.from(randomF32(N * K, rng), roundF16);
    const bias = randomF32(N, rng);
    const { out } = await runCase(gemv, A, Bf, N, K, true, Math.ceil(N / 256), bias);
    const detail = assertClose(out, gemmNTRef(A, Bf, 1, N, K, bias), 1e-4, `gemm_gemv(N=${N},K=${K})`);
    lines.push(`N=${N} K=${K}: ${detail}`);
  }

  // --- 2) 性能：真实形状 N=4864, K=896（gate_proj），M=1 ---
  const N = 4864;
  const K = 896;
  const rng = makeRng(0x9ef0);
  const A = randomF32(K, rng);
  const Bf = Float32Array.from(randomF32(N * K, rng), roundF16);

  const iters = 30;
  const flops = 2 * N * K;
  const bytes = N * K * 2 + K * 4 + N * 4;

  const bench = async (pipeline: GPUComputePipeline, grid: number | readonly number[], label: string) => {
    const { bufC, bg } = await runCase(pipeline, A, Bf, N, K, false, grid);
    dispatch(device, pipeline, bg, grid);
    await readbackF32(device, bufC, 0, 1);
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) dispatch(device, pipeline, bg, grid);
    await readbackF32(device, bufC, 0, 1);
    const t1 = performance.now();
    const ms = (t1 - t0) / iters;
    bufC.destroy();
    return `${label} ${ms.toFixed(2)}ms/步（${(flops / ms / 1e6).toFixed(1)} GFLOP/s，${(bytes / ms / 1e6).toFixed(1)} GB/s）`;
  };

  const gemvLine = await bench(gemv, Math.ceil(N / 256), 'GEMV');
  const tiledLine = await bench(tiled, [1, Math.ceil(N / 64), 1], '通用 GEMM(BM=64)');

  return [
    `正确性 M=1：${lines.join(' | ')}`,
    `性能 N=${N} K=${K} iters=${iters}：${gemvLine} | ${tiledLine}`,
  ].join('；');
}

/**
 * M5：split-K 解码 GEMV。
 *
 * gemm_gemv_f16 的并行度 = 输出列数 N：k/v_proj（N=128）只能排 1 个 workgroup、
 * q/o/down（N=896）只有 4 个，Vega 的 8~11 个 CU 大部分在空转。
 * split-K 把 K 维切成 S 段并行（grid=[ceil(N/256), S]），再用归约 kernel 合并，
 * 用 S 倍的 workgroup 换回带宽利用率。
 *
 * 正确性覆盖：K=42/S=3（非 2 幂段数 + u32 主循环后的尾部）、K=896/S=8（真实
 * k/v_proj 形状）、K=896/S=4（真实 q_proj 的另一档位），各带/不带 bias。
 * 性能取 q_proj 形状（N=896, K=896，原 GEMV 只有 4 个 workgroup），
 * 对比单列 GEMV 的每步耗时与等效带宽。
 */
async function testGemmGemvSplitF16(gpu: GpuContext): Promise<string> {
  if (!gpu.features.has('shader-f16')) return 'shader-f16 不可用，跳过 split-K GEMV 验证';
  const device = gpu.device;
  const splitP = createComputePipeline(device, gemmGemvSplitF16Wgsl);
  const reduceP = createComputePipeline(device, gemmGemvSplitReduceWgsl);
  const gemvP = createComputePipeline(device, gemmGemvF16Wgsl);

  const runSplit = async (
    A: Float32Array,
    B: Float32Array,
    N: number,
    K: number,
    S: number,
    hasBias: boolean,
    bias?: Float32Array,
  ): Promise<Float32Array> => {
    const wgN = Math.ceil(N / 256);
    const bufA = createStorageBuffer(device, A.length * 4);
    const bufB = createStorageBuffer(device, B.length * 2);
    const bufC = createStorageBuffer(device, N * 4);
    const bufPartial = createStorageBuffer(device, S * N * 4);
    const bufBias = createStorageBuffer(device, Math.max(N, 1) * 4);
    writeF32(device, bufA, 0, A);
    writeF16(device, bufB, 0, B);
    if (bias) writeF32(device, bufBias, 0, bias);

    const u = new StructPacker();
    u.u32(N);
    u.u32(K);
    u.u32(S);
    u.u32(hasBias ? 1 : 0);
    const bufDims = createUniform(device, u.bytes());

    const bgSplit = device.createBindGroup({
      layout: splitP.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufPartial } },
        { binding: 3, resource: { buffer: bufDims } },
      ],
    });
    const bgReduce = device.createBindGroup({
      layout: reduceP.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufPartial } },
        { binding: 1, resource: { buffer: bufC } },
        { binding: 2, resource: { buffer: bufBias } },
        { binding: 3, resource: { buffer: bufDims } },
      ],
    });

    dispatch(device, splitP, bgSplit, [wgN, S]);
    dispatch(device, reduceP, bgReduce, wgN);
    const out = await readbackF32(device, bufC, 0, N);
    [bufA, bufB, bufC, bufPartial, bufBias, bufDims].forEach((b) => b.destroy());
    return out;
  };

  // --- 1) 正确性：段数阶梯 + 尾部循环 + bias 有无 ---
  const lines: string[] = [];
  for (const { N, K, S } of [
    { N: 53, K: 42, S: 3 },
    { N: 128, K: 896, S: 8 },
    { N: 64, K: 896, S: 4 },
  ]) {
    const rng = makeRng(0x5e10 + N * 31 + S);
    const A = randomF32(K, rng);
    const Bf = Float32Array.from(randomF32(N * K, rng), roundF16);
    const bias = randomF32(N, rng);
    const withBias = assertClose(
      await runSplit(A, Bf, N, K, S, true, bias),
      gemmNTRef(A, Bf, 1, N, K, bias),
      1e-4,
      `split(N=${N},K=${K},S=${S},bias)`,
    );
    const noBias = assertClose(
      await runSplit(A, Bf, N, K, S, false),
      gemmNTRef(A, Bf, 1, N, K),
      1e-4,
      `split(N=${N},K=${K},S=${S})`,
    );
    lines.push(`N=${N} K=${K} S=${S}: ${withBias} | ${noBias}`);
  }

  // --- 2) 性能：q_proj 形状，split 前后对比 ---
  const N = 896;
  const K = 896;
  const S = 8;
  const rng = makeRng(0x5e11);
  const A = randomF32(K, rng);
  const Bf = Float32Array.from(randomF32(N * K, rng), roundF16);
  const wgN = Math.ceil(N / 256);
  const iters = 30;
  const bytes = N * K * 2 + K * 4 + N * 4; // split 额外的 partial 流量 ~57KB，占比 <4%，忽略

  const bufA = createStorageBuffer(device, A.length * 4);
  const bufB = createStorageBuffer(device, Bf.length * 2);
  const bufC1 = createStorageBuffer(device, N * 4);
  const bufC2 = createStorageBuffer(device, N * 4);
  const bufPartial = createStorageBuffer(device, S * N * 4);
  const bufBias = createStorageBuffer(device, N * 4);
  writeF32(device, bufA, 0, A);
  writeF16(device, bufB, 0, Bf);

  const u1 = new StructPacker();
  u1.u32(1);
  u1.u32(N);
  u1.u32(K);
  u1.u32(0);
  const bufDims1 = createUniform(device, u1.bytes());
  const u2 = new StructPacker();
  u2.u32(N);
  u2.u32(K);
  u2.u32(S);
  u2.u32(0);
  const bufDims2 = createUniform(device, u2.bytes());

  const bgGemv = device.createBindGroup({
    layout: gemvP.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA } },
      { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufC1 } },
      { binding: 3, resource: { buffer: bufBias } },
      { binding: 4, resource: { buffer: bufDims1 } },
    ],
  });
  const bgSplit = device.createBindGroup({
    layout: splitP.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA } },
      { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufPartial } },
      { binding: 3, resource: { buffer: bufDims2 } },
    ],
  });
  const bgReduce = device.createBindGroup({
    layout: reduceP.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufPartial } },
      { binding: 1, resource: { buffer: bufC2 } },
      { binding: 2, resource: { buffer: bufBias } },
      { binding: 3, resource: { buffer: bufDims2 } },
    ],
  });

  const timeIters = async (step: () => void): Promise<number> => {
    step();
    await readbackF32(device, bufC1, 0, 1); // 预热 + 同步
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) step();
    await readbackF32(device, bufC1, 0, 1);
    return (performance.now() - t0) / iters;
  };

  const plainMs = await timeIters(() => dispatch(device, gemvP, bgGemv, wgN));
  const splitMs = await timeIters(() => {
    dispatch(device, splitP, bgSplit, [wgN, S]);
    dispatch(device, reduceP, bgReduce, wgN);
  });

  [bufA, bufB, bufC1, bufC2, bufPartial, bufBias, bufDims1, bufDims2].forEach((b) => b.destroy());

  const bw = (ms: number) => `${(bytes / ms / 1e6).toFixed(1)} GB/s`;
  return [
    `正确性 M=1：${lines.join(' | ')}`,
    `性能 N=${N} K=${K} iters=${iters}：单列 GEMV ${plainMs.toFixed(2)}ms/步（${bw(plainMs)}） | split-K S=${S} ${splitMs.toFixed(2)}ms/步（${bw(splitMs)}）→ ${(plainMs / splitMs).toFixed(1)}x`,
  ].join('；');
}

export const kernelTests: SelfTest[] = [
  { name: 'vec_add（WGSL vs CPU）', run: testVecAdd },
  { name: 'gemm_nt（含 bias + 边界）', run: testGemm },
  { name: 'gemm_nn（反向 dInput）', run: testGemmNn },
  { name: 'gemm_tn（反向 dWeight）', run: testGemmTn },
  { name: 'gemm_nt_f16（f16 输入 + f32 累加 + 吞吐）', run: testGemmNtF16 },
  { name: 'embedding 查表', run: testEmbedding },
  { name: 'rmsnorm', run: testRmsnorm },
  { name: 'layernorm（含 bias）', run: testLayernorm },
  { name: 'gelu（tanh 近似）', run: testGelu },
  { name: 'gelu_bwd', run: testGeluBwd },
  { name: 'rope', run: testRope },
  { name: 'rope_bwd', run: testRopeBwd },
  { name: 'softmax', run: testSoftmax },
  { name: 'attention（因果）', run: testAttention },
  { name: 'layernorm_bwd（dx/dw/db）', run: testLayernormBwd },
  { name: 'add_layernorm（残差+归一化融合）', run: testAddLayernorm },
  { name: 'sum_rows（bias 梯度）', run: testSumRows },
  { name: 'ce_softmax_bwd', run: testCeSoftmaxBwd },
  { name: 'embedding_bwd（累加）', run: testEmbeddingBwd },
  { name: 'adamw 单步', run: testAdamw },
  { name: 'attention_bwd（dq/dk/dv）', run: testAttentionBwd },
  { name: 'M4-3 rope_half（Qwen half-split RoPE）', run: testRopeHalf },
  { name: 'M4-3 attention_gqa（分组查询注意力）', run: testAttentionGqa },
  { name: 'M4-3 silu_mul（SwiGLU 融合激活）', run: testSiluMul },
  { name: 'M4-3 embedding_f16（f16 词嵌入查表）', run: testEmbeddingF16 },
  { name: 'M5 gemm_gemv_f16（解码 M=1 专用 + 对比通用 GEMM）', run: testGemmGemvF16 },
  { name: 'M5 gemm_gemv_split_f16（split-K 解码 + 归约对拍）', run: testGemmGemvSplitF16 },
  { name: 'M5 解码瓶颈剖析（timestamp-query 逐 pass 耗时）', run: testDecodeProfile },
];