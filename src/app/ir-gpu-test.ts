/**
 * GPU 后端的浏览器验证入口（设计 06 §9 的 ENG-V7：多后端跑同一模型、数值对拍）。
 *
 * 为什么要一个专门的页面：WebGPU 只能在真实浏览器里跑，Node 里没有。
 * 于是这里把"跑一次对拍"暴露成 `window.__IR_GPU__.run()`，
 * 由 `scripts/verify-ir-gpu.mjs` 用 CDP 驱动 headless Chromium 调用它。
 *
 * 对比三方，缺一不可：
 *   ① GPU 上跑 IR（`runGpu` + `gpu-impls`）
 *   ② CPU 上跑**同一个 artifact**（`run` + `cpu-impls`）
 *   ③ 手写参考实现（`gptForwardRef`，与 IR 完全无关的一条路径）
 * ①↔② 是"同一份编译产物在两个后端一致"（ENG-V7）；①↔③ 是"IR 语义没走样"。
 */

import { initGpu } from '../gpu/device';
import { infer } from '../ir/infer';
import { plan } from '../ir/plan';
import { emit } from '../ir/emit';
import { run } from '../ir/exec';
import { builtinCpuImpls } from '../ir/cpu-impls';
import { buildGptIr, bindBatch } from '../ir/tinygpt-ir';
import { GpuRuntime, runGpu } from '../ir/gpu-exec';
import { builtinGpuImpls, builtinGpuKernels } from '../ir/gpu-impls';
import { DEFAULT_CONFIG } from '../model/config';
import { initWeights } from '../model/init';
import { gptForwardRef } from '../reference/gpt-ref';
import { checkTolerance } from '../reference/cpu-ref';

export interface GpuVerifyResult
{
  ok: boolean;
  error?: string;
  adapter?: { vendor: string; architecture: string; maxBufferSize: number; hasF16: boolean };
  B?: number; T?: number;
  nodeCount?: number;
  plannedPasses?: number;
  stats?: { dispatches: number; passBreaks: number; passthrough: string[] };
  gpuVsCpuIr?: { ok: boolean; maxAbs: number; maxRel: number; elements: number };
  gpuVsRef?: { ok: boolean; maxAbs: number; maxRel: number; elements: number };
  cpuIrVsRef?: { ok: boolean; maxAbs: number; maxRel: number; elements: number };
  sample?: number[];
}

function rng32 ( seed: number ): () => number
{
  let s = seed >>> 0 || 1;
  return () =>
  {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 13; s >>>= 0;
    return s / 0x100000000;
  };
}

export async function runGpuVerify ( opts: { B?: number; T?: number; seed?: number } = {} ): Promise<GpuVerifyResult>
{
  try
  {
    const cfg = DEFAULT_CONFIG;
    const B = opts.B ?? 2;
    const T = opts.T ?? 8;
    const seed = opts.seed ?? 1234;

    const gpu = await initGpu();
    const adapter = {
      vendor: gpu.adapterInfo?.vendor ?? '',
      architecture: gpu.adapterInfo?.architecture ?? '',
      maxBufferSize: gpu.limits.maxBufferSize,
      hasF16: gpu.hasF16,
    };

    // ---- 同一份 IR + 同一批权重 ----
    const weights = initWeights( cfg, seed );
    const gpt = buildGptIr( cfg, weights );
    const rnd = rng32( 7 );
    const tokens = new Uint32Array( B * T );
    for ( let i = 0; i < tokens.length; i++ ) tokens[i] = Math.floor( rnd() * cfg.vocabSize );
    const binding = bindBatch( gpt, tokens, B, T );

    // ---- 编译一次（infer → plan → emit），两个后端共用同一个 artifact ----
    const ir = infer( gpt.model, { symbols: { B, T } } );
    const execPlan = plan( gpt.model, ir, {
      backends: [ {
        backend: 'webgpu',
        maxBufferSize: gpu.limits.maxBufferSize,
        maxBindGroups: gpu.limits.maxBindGroups,
        maxWorkgroupsPerDimension: 65535,
        supportsF16: gpu.hasF16,
        supportsSubgroups: false,
      } ],
    } );
    const artifact = emit( gpt.model, ir, execPlan );

    // ---- ① CPU 上跑同一 artifact ----
    const cpuRes = run( gpt.model, ir, artifact, binding, builtinCpuImpls() );
    const cpuOut = cpuRes.rootOutput.data as Float32Array;

    // ---- ② GPU 上跑同一 artifact ----
    const rt = new GpuRuntime( gpu, builtinGpuKernels(), gpt.tensors, { B, T } );
    const gpuRes = await runGpu( gpt.model, ir, artifact, binding, rt, { impls: builtinGpuImpls() } );
    const gpuOut = gpuRes.rootOutput;
    rt.dispose();

    // ---- ③ 手写参考实现 ----
    const ref = gptForwardRef( weights, cfg, { tokens, B, T } );

    const gpuVsCpuIr = checkTolerance( gpuOut, cpuOut, 1e-4, 1e-4 );
    const gpuVsRef = checkTolerance( gpuOut, ref.logits, 1e-3, 1e-3 );
    const cpuIrVsRef = checkTolerance( cpuOut, ref.logits, 1e-4, 1e-4 );

    return {
      ok: gpuVsCpuIr.ok && gpuVsRef.ok && cpuIrVsRef.ok,
      adapter,
      B, T,
      nodeCount: Object.keys( gpt.model.nodes ).length,
      plannedPasses: artifact.passes.length,
      stats: {
        dispatches: gpuRes.stats.dispatches,
        passBreaks: gpuRes.stats.passBreaks,
        passthrough: gpuRes.stats.passthrough,
      },
      gpuVsCpuIr: { ok: gpuVsCpuIr.ok, maxAbs: gpuVsCpuIr.maxAbs, maxRel: gpuVsCpuIr.maxRel, elements: cpuOut.length },
      gpuVsRef: { ok: gpuVsRef.ok, maxAbs: gpuVsRef.maxAbs, maxRel: gpuVsRef.maxRel, elements: ref.logits.length },
      cpuIrVsRef: { ok: cpuIrVsRef.ok, maxAbs: cpuIrVsRef.maxAbs, maxRel: cpuIrVsRef.maxRel, elements: ref.logits.length },
      sample: [ ...gpuOut.slice( 0, 5 ) ],
    };
  }
  catch ( err )
  {
    return { ok: false, error: ( err as Error ).message + '\n' + ( ( err as Error ).stack ?? '' ) };
  }
}

( window as unknown as { __IR_GPU__: unknown } ).__IR_GPU__ = { run: runGpuVerify };
