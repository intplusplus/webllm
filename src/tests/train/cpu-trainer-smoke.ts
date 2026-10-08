/**
 * CpuTrainer 无头冒烟：CPU 兜底训练台的三阶段管线（预训练 / SFT / DPO）
 * + checkpoint 往返 + 与手写参考实现的前向对拍。
 *
 * 立场：CpuTrainer 是「无 WebGPU 环境的训练台」，它的行为必须与 GPU Trainer
 * 在同一调用面上一致 —— 尤其是加权反向的 DPO 语义（±系数直接折进 CE 反向，
 * w 由调用方归一，不除 Σw）。这里用最接近 app.ts 真实用法的方式跑通全链路。
 */
import { CpuTrainer } from '../../train/cpu-trainer';
import { initWeights } from '../../model/init';
import { gptForwardRef } from '../../reference/gpt-ref';
import { checkTolerance } from '../../reference/cpu-ref';
import { makeSftSamples, SFT_VOCAB } from '../../train/sft-data';
import { buildBatch } from '../../train/sft';
import type { GPTConfig } from '../../model/config';

export interface SmokeResult { name: string; pass: boolean; detail: string; }

const UI_CONFIG: GPTConfig = {
  vocabSize: SFT_VOCAB.length,
  blockSize: 32,
  nLayer: 2,
  nHead: 2,
  nEmbd: 64,
  bias: true,
};

/** GPU Trainer.uploadWeights 的注册名单（bias=true 时）—— checkpoint 互通的契约面。 */
const GPU_PARAM_NAMES: string[] = (() =>
{
  const names: string[] = [];
  names.push( 'wte', 'wpe', 'lnFW', 'lnFB', 'lmHead.b' );
  for ( let i = 0; i < UI_CONFIG.nLayer; i++ )
  {
    const p = `L${ i }`;
    names.push( `${ p }.ln1W`, `${ p }.ln1B`, `${ p }.ln2W`, `${ p }.ln2B` );
    for ( const k of [ 'wq', 'wk', 'wv', 'attnProj', 'fc', 'mlpProj' ] )
      names.push( `${ p }.${ k }.w`, `${ p }.${ k }.b` );
  }
  return names;
})();

function mulberry32 ( seed: number ): () => number
{
  let a = seed >>> 0;
  return () =>
  {
    a |= 0; a = ( a + 0x6D2B79F5 ) | 0;
    let t = Math.imul( a ^ ( a >>> 15 ), 1 | a );
    t = ( t + Math.imul( t ^ ( t >>> 7 ), 61 | t ) ) ^ t;
    return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;
  };
}

export async function runCpuTrainerSmoke (): Promise<SmokeResult[]>
{
  const out: SmokeResult[] = [];
  const push = ( name: string, pass: boolean, detail: string ): void =>
  { out.push( { name, pass, detail } ); };

  const V = UI_CONFIG.vocabSize;
  const rnd = mulberry32( 2026 );

  // ------------------------------------------------------------ CT-1 前向对拍
  {
    const t = new CpuTrainer( UI_CONFIG, initWeights( UI_CONFIG, 4242 ) );
    const B = 2, T = 8;
    const tokens = new Uint32Array( B * T );
    for ( let i = 0; i < tokens.length; i++ ) tokens[ i ] = Math.floor( rnd() * V );
    t.forward( tokens, B, T );
    const got = await t.readLogits();
    const ref = gptForwardRef( initWeights( UI_CONFIG, 4242 ), UI_CONFIG, { tokens, B, T } );
    const cmp = checkTolerance( got, ref.logits, 1e-4, 1e-4 );
    push( 'CT-1 · CpuTrainer 前向 vs 手写参考实现（gptForwardRef）', cmp.ok,
      `maxAbs=${ cmp.maxAbs.toExponential( 2 ) } maxRel=${ cmp.maxRel.toExponential( 2 ) }（${ got.length } 元素）` );

    // CT-2：uniform loss = 平均 CE（手算对照）
    const targets = new Uint32Array( B * T );
    for ( let i = 0; i < targets.length; i++ ) targets[ i ] = Math.floor( rnd() * V );
    const loss = await t.loss( targets );
    const logits = got;
    let total = 0;
    for ( let r = 0; r < B * T; r++ )
    {
      const base = r * V;
      let mx = -Infinity;
      for ( let v = 0; v < V; v++ ) mx = Math.max( mx, logits[ base + v ] );
      let se = 0;
      for ( let v = 0; v < V; v++ ) se += Math.exp( logits[ base + v ] - mx );
      total += -( logits[ base + targets[ r ] ] - mx - Math.log( se ) );
    }
    const manual = total / ( B * T );
    const ok = Math.abs( loss - manual ) <= 1e-5 * Math.max( 1, Math.abs( manual ) );
    push( 'CT-2 · uniform loss = 平均 CE（手算对照）', ok,
      `IR=${ loss.toFixed( 6 ) } 手算=${ manual.toFixed( 6 ) }` );
  }

  // ------------------------------------------------------------ CT-3 SFT 加权 loss
  {
    const t = new CpuTrainer( UI_CONFIG, initWeights( UI_CONFIG, 777 ) );
    const samples = makeSftSamples( 20260101, 16 );
    const batch = buildBatch( [ 0, 1, 2, 3, 4, 5, 6, 7 ].map( ( i ) => samples[ i ] ), SFT_VOCAB.indexOf( ' ' ) );
    t.forward( batch.tokens, batch.B, batch.T );
    const lossW = await t.loss( batch.targets, batch.weights );
    // masked 行不贡献：把权重全清零后 loss 应报「权重和为 0」？不 —— loss() 手算路径
    // 与 GPU 一致（Σw·ce/Σw），全零由「Σw=0」兜底。这里只验证半掩码数值在合理范围。
    const logits = await t.readLogits();
    let wsum = 0, sumW = 0;
    for ( let r = 0; r < batch.B * batch.T; r++ )
    {
      const w = batch.weights[ r ];
      if ( w === 0 ) continue;
      const base = r * V;
      let mx = -Infinity;
      for ( let v = 0; v < V; v++ ) mx = Math.max( mx, logits[ base + v ] );
      let se = 0;
      for ( let v = 0; v < V; v++ ) se += Math.exp( logits[ base + v ] - mx );
      wsum += w * ( mx + Math.log( se ) - logits[ base + batch.targets[ r ] ] );
      sumW += w;
    }
    const manual = wsum / sumW;
    const ok = Math.abs( lossW - manual ) <= 1e-5 * Math.max( 1, Math.abs( manual ) );
    push( 'CT-3 · SFT 加权 loss（Σw·ce/Σw，只计 completion）', ok,
      `IR=${ lossW.toFixed( 6 ) } 手算=${ manual.toFixed( 6 ) } 有效行 ${ batch.weights.filter( ( w ) => w > 0 ).length }/${ batch.B * batch.T }` );
  }

  // ------------------------------------------------------------ CT-4/5 训练下降
  {
    const t = new CpuTrainer( UI_CONFIG, initWeights( UI_CONFIG, 4242 ) );
    // 预训练式：随机 token 序列，uniform loss 30 步
    const B = 4, T = 16;
    const tokens = new Uint32Array( B * T );
    const targets = new Uint32Array( B * T );
    for ( let i = 0; i < tokens.length; i++ )
    {
      tokens[ i ] = Math.floor( rnd() * V );
      targets[ i ] = Math.floor( rnd() * V );
    }
    const opts = { lr: 3e-3, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
    t.forward( tokens, B, T );
    const first = await t.loss( targets );
    let last = first;
    const t0 = performance.now();
    for ( let s = 0; s < 30; s++ )
    {
      t.forward( tokens, B, T );
      last = await t.loss( targets );
      t.backward( targets );
      t.step( opts );
    }
    const msPerStep = ( performance.now() - t0 ) / 30;
    const ok = last < first && Number.isFinite( last );
    push( 'CT-4 · 预训练式 30 步 loss 下降（无 NaN）', ok,
      `${ first.toFixed( 4 ) } → ${ last.toFixed( 4 ) }，${ msPerStep.toFixed( 1 ) } ms/步（B=${ B },T=${ T }）` );
  }
  {
    const t = new CpuTrainer( UI_CONFIG, initWeights( UI_CONFIG, 4242 ) );
    const samples = makeSftSamples( 20260101, 64 );
    const opts = { lr: 0.01, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
    const batchOf = ( s: number ) => buildBatch(
      [ 0, 1, 2, 3, 4, 5, 6, 7 ].map( ( i ) => samples[ ( s * 8 + i ) % samples.length ] ),
      SFT_VOCAB.indexOf( ' ' ),
    );
    let batch = batchOf( 0 );
    t.forward( batch.tokens, batch.B, batch.T );
    const first = await t.loss( batch.targets, batch.weights );
    let last = first;
    let nan = false;
    const t0 = performance.now();
    for ( let s = 0; s < 20; s++ )
    {
      batch = batchOf( s );
      t.forward( batch.tokens, batch.B, batch.T );
      last = await t.loss( batch.targets, batch.weights );
      if ( !Number.isFinite( last ) ) { nan = true; break; }
      t.backward( batch.targets, batch.weights );
      t.step( opts );
    }
    const msPerStep = ( performance.now() - t0 ) / 20;
    push( 'CT-5 · SFT 式 20 步加权 loss 下降（带掩码反向）',
      !nan && last < first,
      `${ first.toFixed( 4 ) } → ${ last.toFixed( 4 ) }，${ msPerStep.toFixed( 1 ) } ms/步（B=${ batch.B },T=${ batch.T }）` );
  }

  // ------------------------------------------------------------ CT-6 DPO ±系数
  {
    const t = new CpuTrainer( UI_CONFIG, initWeights( UI_CONFIG, 31337 ) );
    const samples = makeSftSamples( 31337, 16 );
    const seqs = [ samples[ 0 ], { prompt: samples[ 0 ].prompt, completion: samples[ 1 ].completion } ];
    const batch = buildBatch( seqs, SFT_VOCAB.indexOf( ' ' ) );
    const K = 1;
    const beta = 0.5;
    const oneMinusSigma = 0.7;
    const weights = new Float32Array( batch.weights );
    for ( let tt = 0; tt < batch.T; tt++ )
    {
      for ( const [ row, sign ] of [ [ 0, 1 ], [ 1, -1 ] ] as const )
      {
        const mask = batch.weights[ row * batch.T + tt ];
        weights[ row * batch.T + tt ] =
          mask === 0 ? 0 : ( sign * beta * oneMinusSigma ) / ( K * batch.completionLens[ row ] );
      }
    }
    const opts = { lr: 1e-3, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
    const before = ( await t.readParam( 'wte' ) ).slice();
    t.forward( batch.tokens, batch.B, batch.T );
    t.backward( batch.targets, weights );
    t.step( opts );
    const after = await t.readParam( 'wte' );
    let maxDelta = 0;
    let nan = false;
    for ( let i = 0; i < after.length; i++ )
    {
      if ( !Number.isFinite( after[ i ] ) ) { nan = true; break; }
      maxDelta = Math.max( maxDelta, Math.abs( after[ i ] - before[ i ] ) );
    }
    // Σw ≈ 0（± 系数对消）—— 旧语义（除 Σw）在这里会爆炸；新语义必须给出小而有限的更新
    const sumW = weights.reduce( ( a, b ) => a + b, 0 );
    push( 'CT-6 · DPO ±系数反向（Σw≈0 不爆炸、无 NaN、更新小而有限）',
      !nan && maxDelta > 0 && maxDelta < 1e-2,
      `Σw=${ sumW.toExponential( 2 ) }（≈0），maxΔwte=${ maxDelta.toExponential( 2 ) }` );
  }

  // ------------------------------------------------------------ CT-7 checkpoint 往返
  {
    const t = new CpuTrainer( UI_CONFIG, initWeights( UI_CONFIG, 4242 ) );
    const names = t.paramNames();
    const missing = GPU_PARAM_NAMES.filter( ( n ) => !names.includes( n ) );
    const extra = names.filter( ( n ) => !GPU_PARAM_NAMES.includes( n ) );
    const namesOk = missing.length === 0 && extra.length === 0;

    // 优化器状态往返
    const B = 2, T = 8;
    const tokens = new Uint32Array( B * T ).fill( 1 );
    const targets = new Uint32Array( B * T ).fill( 2 );
    t.forward( tokens, B, T );
    t.backward( targets );
    t.step( { lr: 1e-3, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 } );
    const probe = names[ 0 ];
    const st = await t.readOptimizerState( probe );
    const t2 = new CpuTrainer( UI_CONFIG, initWeights( UI_CONFIG, 4242 ) );
    t2.writeOptimizerState( probe, st.m, st.v );
    const st2 = await t2.readOptimizerState( probe );
    let stateOk = st.m.length === st2.m.length && st.v.length === st2.v.length;
    for ( let i = 0; i < st.m.length && stateOk; i++ )
      stateOk = st.m[ i ] === st2.m[ i ] && st.v[ i ] === st2.v[ i ];
    t.setStepCount( 123 );
    const stepOk = t.stepCount === 123;

    push( 'CT-7 · paramNames 与 GPU 名单一致 + 优化器状态往返 + stepCount',
      namesOk && stateOk && stepOk,
      `names 缺失=${ missing.length } 多余=${ extra.length }（共 ${ names.length }）；state 往返=${ stateOk }；stepCount=${ t.stepCount }` );
  }

  // ------------------------------------------------------------ CT-8 exportWeights → 新实例逐位一致
  {
    const t = new CpuTrainer( UI_CONFIG, initWeights( UI_CONFIG, 4242 ) );
    // 先训几步让权重偏离初始值（同时验证 writeParam 走 tie 不破坏别名）
    const B = 2, T = 8;
    const tokens = new Uint32Array( B * T ).fill( 3 );
    const targets = new Uint32Array( B * T ).fill( 4 );
    for ( let s = 0; s < 5; s++ )
    {
      t.forward( tokens, B, T );
      t.backward( targets );
      t.step( { lr: 1e-3, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 } );
    }
    const w = await t.exportWeights();
    const tieOk = w.lmHead.w === w.wte; // 同一引用（tie 语义保持）
    const t2 = new CpuTrainer( UI_CONFIG, w );
    t.forward( tokens, B, T );
    t2.forward( tokens, B, T );
    const a = await t.readLogits();
    const b = await t2.readLogits();
    let same = a.length === b.length;
    for ( let i = 0; i < a.length && same; i++ ) same = a[ i ] === b[ i ];
    push( 'CT-8 · exportWeights → 新 CpuTrainer logits 逐位一致（tie 引用保持）',
      tieOk && same, `lmHead.w===wte 引用相同=${ tieOk }；logits 逐位一致=${ same }（${ a.length } 元素）` );
  }

  // ------------------------------------------------------------ CT-9 全尺寸单步计时
  {
    const t = new CpuTrainer( UI_CONFIG, initWeights( UI_CONFIG, 4242 ) );
    const B = 8, T = UI_CONFIG.blockSize;
    const tokens = new Uint32Array( B * T );
    const targets = new Uint32Array( B * T );
    for ( let i = 0; i < tokens.length; i++ )
    {
      tokens[ i ] = Math.floor( rnd() * V );
      targets[ i ] = Math.floor( rnd() * V );
    }
    const opts = { lr: 3e-3, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
    // 预热（含编译）：首个 (B,T) 组合会触发 infer/plan/emit
    t.forward( tokens, B, T );
    t.backward( targets );
    t.step( opts );
    const N = 10;
    const t0 = performance.now();
    for ( let s = 0; s < N; s++ )
    {
      t.forward( tokens, B, T );
      await t.loss( targets );
      t.backward( targets );
      t.step( opts );
    }
    const ms = ( performance.now() - t0 ) / N;
    push( 'CT-9 · 全尺寸（B=8,T=32）单步耗时（UI 缩放步数的依据）', ms < 5000,
      `${ ms.toFixed( 0 ) } ms/步（前向+损失+反向+更新；编译缓存命中后）` );
  }

  return out;
}
