/**
 * CPU 引擎（TinyMlpEngine）的耗时基准 —— 用「用户真实会点出的那组参数」实测。
 *
 * 为什么要它：真实浏览器测试一直跑在 GPU 引擎上（无头 Chromium 有 WebGPU），
 * 而「房主 + 手机」的房间会协商成 **CPU 引擎** —— 这条路径从未被测过耗时。
 * 用户反馈「点击训练会卡死」，第一嫌疑就是它：纯 JS 训练是同步的，
 * 一步算多久，主线程就卡多久。
 *
 * 跑法： scripts/bench-cpu.mjs
 */
import { buildStoi, encodeTo } from '../src/fed/corpus';
import { TinyMlpEngine } from '../src/fed/engine';
import type { MlpModelSpec } from '../src/fed/protocol';

export interface BenchResult
{
  params: number;
  ids: number;
  oneStepMs: number;
  roundMs: number;
  evalMs: number;
  perRoundBudgetMs: number;
}

/** preset → MLP 规格（与 protocol.ts 的 MODEL_PRESETS 保持一致，手工同步） */
const PRESETS: Record<string, { ctx: number; embDim: number; hidden: number }> = {
  small: { ctx: 8, embDim: 16, hidden: 64 },
  medium: { ctx: 16, embDim: 48, hidden: 192 },
  large: { ctx: 32, embDim: 96, hidden: 384 },
};

export async function runBench (
  text: string,
  preset: keyof typeof PRESETS,
  opts: { steps: number; batch: number },
): Promise<BenchResult>
{
  const dims = PRESETS[ preset ];
  const sample = text.slice( 0, Math.floor( text.length / 4 ) ); // 模拟 host 的 shard #0（1/4）
  const vocab = [ ...new Set( sample ) ].sort();
  const stoi = buildStoi( vocab );
  const ids = encodeTo( sample, stoi );

  const spec: MlpModelSpec = { engine: 'mlp', vocabSize: vocab.length, ...dims, seed: 20261006 };
  const eng = new TinyMlpEngine( spec );

  // 预热（JIT + 内存分配）
  await eng.trainBatch( ids, opts.batch, 0.02 );

  let t0 = performance.now();
  await eng.trainBatch( ids, opts.batch, 0.02 );
  const oneStepMs = performance.now() - t0;

  t0 = performance.now();
  for ( let i = 0; i < opts.steps; i++ ) await eng.trainBatch( ids, opts.batch, 0.02 );
  const roundMs = performance.now() - t0;

  t0 = performance.now();
  await eng.evalAt( ids, 1000, 256 );
  const evalMs = performance.now() - t0;

  return {
    params: eng.paramCount(),
    ids: ids.length,
    oneStepMs,
    roundMs,
    evalMs,
    perRoundBudgetMs: roundMs / opts.steps,
  };
}
