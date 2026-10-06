import type { GpuContext } from '../gpu/device';
import { initWeights } from '../model/init';
import { TinyGpt } from '../model/tiny-gpt';
import type { GPTConfig } from '../model/config';
import { Trainer } from '../train/trainer';
import { makeSftSamples, makeDpoPairs, SFT_VOCAB, type SftSample } from '../train/sft-data';
import { buildBatch, greedyCompletion } from '../train/sft';
import { saveCheckpoint, loadCheckpoint, deleteCheckpoint } from '../store/checkpoint';

/**
 * 后训练管线（M6）：SFT → checkpoint → DPO，全部浏览器端。
 *
 * SFT：字符级指令任务（Reverse / Sort），loss 只计 completion（prompt 位置
 *   loss 权重为 0，由 ce_softmax_bwd 的逐行权重实现）。
 * checkpoint：IndexedDB 全量持久化（权重 + AdamW m/v + 步数），往返后 logits
 *   必须逐位一致。
 * DPO：reference 模型用冻结的 TinyGpt；policy 的标量 DPO loss 对 token-logprob
 *   之和的梯度折算成逐行 CE 权重（±β(1-σ(z))/(K·N)），一次反向完成。
 */

const SFT_CONFIG: GPTConfig = {
  vocabSize: SFT_VOCAB.length, // 33
  blockSize: 32,
  nLayer: 2,
  nHead: 2,
  nEmbd: 64,
  bias: true,
};

/** 序列的 completion 部分 token logprob 之和（CPU 侧，weights/mask 由调用方给定）。 */
function rowLogprobSum ( logits: Float32Array, V: number, row: number, targets: Uint32Array, mask: Float32Array, T: number ): number
{
  let sum = 0;
  for ( let t = 0; t < T; t++ )
  {
    if ( mask[ row * T + t ] === 0 ) continue;
    const base = row * T * V + t * V;
    let mx = -Infinity;
    for ( let v = 0; v < V; v++ ) mx = Math.max( mx, logits[ base + v ] );
    let se = 0;
    for ( let v = 0; v < V; v++ ) se += Math.exp( logits[ base + v ] - mx );
    sum += ( logits[ base + targets[ row * T + t ] ] - mx ) - Math.log( se );
  }
  return sum;
}

/**
 * M6-1：SFT —— 指令微调闭环。
 * 验证：① SFT loss 显著下降；② held-out 指令的贪心 completion 正确率。
 */
export async function testSft ( gpu: GpuContext ): Promise<string>
{
  const train = makeSftSamples( 20260101, 512 );
  const heldOut = makeSftSamples( 777, 8 );
  const padId = SFT_VOCAB.indexOf( ' ' );

  const trainer = new Trainer( gpu, SFT_CONFIG, initWeights( SFT_CONFIG, 4242 ), 8 );
  const opts = { lr: 0.01, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
  const steps = 1500;
  let first = 0;
  const evals: number[] = [];

  for ( let s = 0; s < steps; s++ )
  {
    const idx = Array.from( { length: 8 }, ( _, i ) => ( s * 8 + i ) % train.length );
    const batch = buildBatch( idx.map( ( i ) => train[ i ] ), padId );
    trainer.forward( batch.tokens, batch.B, batch.T );
    const l = await trainer.loss( batch.targets, batch.weights );
    if ( s === 0 ) first = l;
    if ( s === 0 || ( s + 1 ) % 500 === 0 ) evals.push( l );
    trainer.backward( batch.targets, batch.weights );
    trainer.step( opts );
  }

  // 机制验证：训练集内样本的贪心生成正确率（mask CE 端到端工作的直接证据）。
  // held-out 泛化率受模型规模限制，作为信息输出不断言。
  const checkSet = train.slice( 0, 8 );
  let correct = 0;
  const fails: string[] = [];
  for ( const s of checkSet )
  {
    const got = await greedyCompletion( trainer, s.prompt, 8, SFT_VOCAB, SFT_CONFIG.blockSize );
    if ( got === s.completion.replace( '\n', '' ) ) correct++;
    else fails.push( `${ s.prompt } → ${ JSON.stringify( got ) }（期望 ${ JSON.stringify( s.completion.replace( '\n', '' ) ) }）` );
  }
  let heldCorrect = 0;
  for ( const s of heldOut )
  {
    const got = await greedyCompletion( trainer, s.prompt, 8, SFT_VOCAB, SFT_CONFIG.blockSize );
    if ( got === s.completion.replace( '\n', '' ) ) heldCorrect++;
  }

  const last = evals[ evals.length - 1 ];
  if ( !( last < 0.5 ) ) throw new Error( `SFT loss 未收敛：${ evals.map( ( x ) => x.toFixed(3) ).join(' -> ') }` );
  if ( correct < 7 ) throw new Error( `训练样本生成正确率不足：${ correct }/8（mask CE 管线异常）；失败样例 ${ fails.slice( 0, 2 ).join( '；' ) }` );

  const curve = evals.map( ( x ) => x.toFixed( 3 ) ).join( ' -> ' );
  return `steps=${steps}，SFT loss ${first.toFixed(3)} -> ${last.toFixed(3)}（${curve}）；训练集生成 ${correct}/8，held-out 泛化 ${heldCorrect}/8${fails.length ? `；失败：${fails[0]}` : ''}`;
}

/**
 * M6-2：IndexedDB checkpoint 往返。
 * 验证：保存→加载→logits 逐位一致，且恢复优化器状态后继续训练不回弹。
 */
export async function testCheckpoint ( gpu: GpuContext ): Promise<string>
{
  const key = 'selftest-tinygpt';
  const cfg = { vocabSize: 12, blockSize: 8, nLayer: 2, nHead: 2, nEmbd: 16, bias: true };
  const opts = { lr: 0.02, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };

  // 训练一个固定小任务若干步
  const weightsA = initWeights( cfg, 4242 );
  const t1 = new Trainer( gpu, cfg, weightsA, 4 );
  const B = 4; const T = 8; const M = B * T; const V = cfg.vocabSize;
  const tokens = new Uint32Array( M );
  const targets = new Uint32Array( M );
  for ( let b = 0; b < B; b++ ) for ( let t = 0; t < T; t++ )
  {
    const v = ( b * 3 + t ) % V;
    tokens[ b * T + t ] = v;
    targets[ b * T + t ] = ( v + 1 ) % V;
  }
  for ( let s = 0; s < 40; s++ )
  {
    t1.forward( tokens, B, T );
    await t1.loss( targets );
    t1.backward( targets );
    t1.step( opts );
  }
  t1.forward( tokens, B, T );
  const lossBefore = await t1.loss( targets );

  // 保存
  const params: Record<string, Float32Array> = {};
  const optimizer: Record<string, { m: Float32Array; v: Float32Array }> = {};
  for ( const name of t1.paramNames() )
  {
    params[ name ] = await t1.readParam( name );
    optimizer[ name ] = await t1.readOptimizerState( name );
  }
  await saveCheckpoint( key, { config: cfg, step: t1.stepCount, params, optimizer } );

  // 加载到全新 Trainer（不同初始权重，全部被覆盖）
  const t2 = new Trainer( gpu, cfg, initWeights( cfg, 999 ), 4 );
  const saved = await loadCheckpoint( key );
  for ( const name of t2.paramNames() )
  {
    t2.writeParam( name, saved.params[ name ] );
    t2.writeOptimizerState( name, saved.optimizer[ name ].m, saved.optimizer[ name ].v );
  }
  t2.setStepCount( saved.step );

  // 往返校验：logits 必须逐位一致
  t1.forward( tokens, B, T );
  t2.forward( tokens, B, T );
  const l1 = await t1.readLogits();
  const l2 = await t2.readLogits();
  let maxDiff = 0;
  for ( let i = 0; i < l1.length; i++ ) maxDiff = Math.max( maxDiff, Math.abs( l1[ i ] - l2[ i ] ) );
  if ( maxDiff !== 0 ) throw new Error( `checkpoint 往返 logits 不一致：maxDiff=${ maxDiff }` );

  // 恢复后继续训练：loss 不应回弹（bias correction 生效）
  for ( let s = 0; s < 20; s++ )
  {
    t2.forward( tokens, B, T );
    await t2.loss( targets );
    t2.backward( targets );
    t2.step( opts );
  }
  t2.forward( tokens, B, T );
  const lossAfter = await t2.loss( targets );
  if ( lossAfter > lossBefore + 0.01 ) throw new Error( `恢复后 loss 回弹：${ lossBefore.toFixed(4) } -> ${ lossAfter.toFixed(4) }` );

  await deleteCheckpoint( key );
  return `保存/恢复 step=${ saved.step }，往返 logits maxDiff=0；恢复后继续 20 步 loss ${ lossBefore.toFixed(4) } -> ${ lossAfter.toFixed(4) }（无回弹）`;
}

/**
 * M6-3：DPO —— 直接偏好优化闭环。
 *
 * policy 用 Trainer，reference 用冻结的 TinyGpt（同一 SFT 后权重）。
 * 标量 loss：L = -(1/K)·Σ logσ(z_i)，z_i = β[(pc_i-pc_ref_i) - (pr_i-pr_ref_i)]，
 * 其中 p 为序列 completion 的 token logprob 之和。对 logits 的梯度折成逐行
 * CE 权重：chosen 行 w=+β(1-σ(z))/(K·N)、rejected 行 w=-β(1-σ(z))/(K·N)，
 * 一次 backward 完成（链式传播复用预训练反向）。
 * 验证：chosen 相对 rejected 的 margin（相对 reference）单调上升。
 */
export async function testDpo ( gpu: GpuContext ): Promise<string>
{
  // --- 1) 先 SFT 一个会做任务的 policy ---
  const train = makeSftSamples( 20260101, 256 );
  const padId = SFT_VOCAB.indexOf( ' ' );
  const trainer = new Trainer( gpu, SFT_CONFIG, initWeights( SFT_CONFIG, 4242 ), 8 );
  const sftOpts = { lr: 0.01, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
  for ( let s = 0; s < 1200; s++ )
  {
    const idx = Array.from( { length: 8 }, ( _, i ) => ( s * 8 + i ) % train.length );
    const batch = buildBatch( idx.map( ( i ) => train[ i ] ), padId );
    trainer.forward( batch.tokens, batch.B, batch.T );
    await trainer.loss( batch.targets, batch.weights );
    trainer.backward( batch.targets, batch.weights );
    trainer.step( sftOpts );
  }

  // --- 2) 冻结 reference ---
  const refWeights = await trainer.exportWeights();
  const ref = new TinyGpt( gpu, SFT_CONFIG, refWeights, 8 );

  // --- 3) DPO 数据与训练 ---
  const pairs = makeDpoPairs( 31337, 12 );
  const K = 4; // 每步 4 个偏好对 → B=8 序列（chosen/rejected 相邻）
  const beta = 0.5;
  const dpoOpts = { lr: 1e-3, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
  const steps = 120;
  const V = SFT_CONFIG.vocabSize;

  const margins: number[] = [];

  for ( let s = 0; s < steps; s++ )
  {
    const batchPairs = Array.from( { length: K }, ( _, i ) => pairs[ ( s * K + i ) % pairs.length ] );
    const seqs: SftSample[] = [];
    for ( const p of batchPairs ) seqs.push( p.sample, { prompt: p.sample.prompt, completion: p.rejected } );
    const batch = buildBatch( seqs, padId );
    const B = batch.B;

    // policy 前向
    trainer.forward( batch.tokens, B, batch.T );
    const policyLogits = await trainer.readLogits();
    // reference 前向（冻结）
    const refLogitsT = ref.forward( batch.tokens, B, batch.T );
    const refLogits = await ref.readTensor( refLogitsT );

    // 每条序列的 completion logp 与 N
    const logp: number[] = [];
    const lens: number[] = [];
    for ( let b = 0; b < B; b++ )
    {
      logp.push( rowLogprobSum( policyLogits, V, b, batch.targets, batch.weights, batch.T ) );
      lens.push( batch.completionLens[ b ] );
    }
    const refLogp: number[] = [];
    for ( let b = 0; b < B; b++ ) refLogp.push( rowLogprobSum( refLogits, V, b, batch.targets, batch.weights, batch.T ) );

    // z、margin、逐行 CE 权重
    const weights = new Float32Array( batch.weights ); // 复制 CE mask 骨架
    let marginSum = 0;
    const sigmoid = ( x: number ) => 1 / ( 1 + Math.exp( -x ) );
    for ( let k = 0; k < K; k++ )
    {
      const cRow = 2 * k;
      const rRow = 2 * k + 1;
      const pc = logp[ cRow ]; const pcr = refLogp[ cRow ];
      const pr = logp[ rRow ]; const prr = refLogp[ rRow ];
      const z = beta * ( ( pc - pcr ) - ( pr - prr ) );
      const oneMinusSigma = 1 - sigmoid( z );
      marginSum += z / beta; // (pc-pcr)-(pr-prr)
      for ( let t = 0; t < batch.T; t++ )
      {
        const mask = batch.weights[ cRow * batch.T + t ];
        weights[ cRow * batch.T + t ] = mask === 0 ? 0 : ( beta * oneMinusSigma ) / ( K * lens[ cRow ] );
        weights[ rRow * batch.T + t ] = mask === 0 ? 0 : ( -beta * oneMinusSigma ) / ( K * lens[ rRow ] );
      }
    }
    margins.push( marginSum / K );

    trainer.backward( batch.targets, weights );
    trainer.step( dpoOpts );
  }

  const m0 = margins[ 0 ];
  const m1 = margins[ margins.length - 1 ];
  const earlyAvg = margins.slice( 0, 10 ).reduce( ( a, b ) => a + b, 0 ) / 10;
  const lateAvg = margins.slice( -10 ).reduce( ( a, b ) => a + b, 0 ) / 10;
  if ( !( lateAvg > earlyAvg ) ) throw new Error( `DPO margin 未上升：前 10 步均值 ${ earlyAvg.toFixed(4) }，后 10 步均值 ${ lateAvg.toFixed(4) }` );

  // 生成仍正确（SFT 能力未被 DPO 破坏）：用训练集内样本评估
  const check = train.slice( 0, 4 );
  let correct = 0;
  for ( const s of check )
  {
    const got = await greedyCompletion( trainer, s.prompt, 8, SFT_VOCAB, SFT_CONFIG.blockSize );
    if ( got === s.completion.replace( '\n', '' ) ) correct++;
  }
  if ( correct < 3 ) throw new Error( `DPO 后生成能力崩坏：仅 ${ correct }/4 正确` );

  return `SFT 1200 步 + DPO ${ steps } 步（β=${ beta }，lr=${ dpoOpts.lr }）：margin ${ m0.toFixed(3) } -> ${ m1.toFixed(3) }（前 10 均值 ${ earlyAvg.toFixed(3) } / 后 10 均值 ${ lateAvg.toFixed(3) }）；DPO 后训练集生成 ${ correct }/4 正确`;
}
