/**
 * 联邦训练核心的无头自检。
 *
 * 只用纯计算模块（protocol / corpus / engine），不碰 WebRTC、DOM、fetch，
 * 因此可以用 esbuild 打包后在 Node 里直接跑：`npm run verify:fed`。
 *
 * 检查的重点是「主张」而不是「代码没崩」：
 *   - 权重帧能不能无损往返、篡改能不能被发现
 *   - FedAvg 的加权是否严格等于手算
 *   - 同一 seed 的模型是否逐位一致（联邦的前提）
 *   - 真实的联邦训练，是否真的比「单节点独训」在共同探针上更好
 */
import { buildStoi, corpusDigest, encodeTo, pickProbe, shardText, vocabOf } from '../../fed/corpus';
import { TinyMlpEngine } from '../../fed/engine';
import {
  aggregateGlobal,
  decodeWeights,
  encodeWeights,
  fedAvg,
  l2Distance,
  weightsDigest,
  type AggregateSpec,
  type AggregateState,
  type MlpModelSpec,
  type NamedWeights,
} from '../../fed/protocol';

export interface CheckResult
{
  name: string;
  pass: boolean;
  detail: string;
}

const SPEC: MlpModelSpec = {
  engine: 'mlp',
  vocabSize: 0, // 运行时按真实字符表填
  ctx: 8,
  embDim: 16,
  hidden: 64,
  seed: 20261006,
};

function specFor ( vocabSize: number, seed = SPEC.seed ): MlpModelSpec
{
  return { ...SPEC, vocabSize, seed };
}

function digestOfWeights ( w: NamedWeights ): string
{
  return weightsDigest( w );
}

export async function runFedSelfCheck ( text: string ): Promise<CheckResult[]>
{
  const results: CheckResult[] = [];
  const check = ( name: string, pass: boolean, detail: string ): void =>
  {
    results.push( { name, pass, detail } );
  };

  const vocab = vocabOf( text );
  const stoi = buildStoi( vocab );
  const ids = encodeTo( text, stoi );
  const spec = specFor( vocab.length );

  // ---------------------------------------------------------------- 1. 编码
  {
    let back = '';
    for ( let i = 0; i < Math.min( 2000, ids.length ); i++ ) back += vocab[ ids[ i ] ];
    const expect = text.slice( 0, 2000 );
    check( '语料编码可逆', back === expect,
      `vocab=${ vocab.length }，chars=${ text.length }，前 2000 字符往返一致=${ back === expect }` );
  }

  // ---------------------------------------------------------------- 2. 权重帧往返
  {
    const a = new TinyMlpEngine( spec );
    const w = await a.getWeights();
    const buf = encodeWeights( w, { round: 1, peerId: 'x', samples: 1, tokens: 1, localLoss: 0, probeLoss: 0, deltaNorm: 0, digest: '' } );
    const dec = decodeWeights( buf );
    let identical = true;
    for ( const k of Object.keys( w ) )
    {
      const x = w[ k ];
      const y = dec.weights[ k ];
      if ( !y || x.length !== y.length ) { identical = false; break; }
      for ( let i = 0; i < x.length; i++ ) if ( x[ i ] !== y[ i ] ) { identical = false; break; }
      if ( !identical ) break;
    }
    check( '权重帧无损往返', identical && dec.meta !== null,
      `${ buf.byteLength } 字节，逐参数逐位一致=${ identical }，摘要=${ dec.digest }` );

    // 篡改 1 个字节应被摘要拦下
    const tampered = buf.slice( 0 );
    const view = new Uint8Array( tampered );
    view[ view.byteLength - 1 ] ^= 0xff;
    let caught = false;
    try { decodeWeights( tampered ); }
    catch { caught = true; }
    check( '篡改权重帧可被检出', caught, caught ? '末字节翻转后 decode 抛错（摘要不匹配）' : '未能检出篡改！' );

    // 摘要对同权重稳定
    const d2 = digestOfWeights( await a.getWeights() );
    check( '权重摘要稳定', d2 === digestOfWeights( await a.getWeights() ), `digest=${ d2 }` );
  }

  // ---------------------------------------------------------------- 3. FedAvg
  {
    const wa: NamedWeights = { p: Float32Array.from( [ 0, 2, 4 ] ) };
    const wb: NamedWeights = { p: Float32Array.from( [ 10, 10, 10 ] ) };
    const avg = fedAvg( [ wa, wb ], [ 1, 3 ] );
    const expect = [ 7.5, 8, 8.5 ];
    const ok = expect.every( ( v, i ) => Math.abs( avg.p[ i ] - v ) < 1e-6 );
    check( 'FedAvg 加权等于手算', ok, `[1,3] 权重 → [${ [ ...avg.p ].join( ', ' ) }]，期望 [${ expect.join( ', ' ) }]` );

    const same = fedAvg( [ wa, wb ], [ 1, 1 ] );
    check( 'FedAvg 等权', Math.abs( same.p[ 0 ] - 5 ) < 1e-6, `[${ [ ...same.p ].join( ', ' ) }]` );
  }

  // ---------------------------------------------------------------- 4. 模型确定性
  {
    const a = new TinyMlpEngine( spec );
    const b = new TinyMlpEngine( spec );
    const c = new TinyMlpEngine( specFor( vocab.length, SPEC.seed + 1 ) );
    const wa = await a.getWeights();
    const wb = await b.getWeights();
    const wc = await c.getWeights();
    const same = ( x: NamedWeights, y: NamedWeights ): boolean =>
      Object.keys( x ).every( ( k ) =>
      {
        const p = x[ k ];
        const q = y[ k ];
        if ( p.length !== q.length ) return false;
        for ( let i = 0; i < p.length; i++ ) if ( p[ i ] !== q[ i ] ) return false;
        return true;
      } );
    check( '同 seed 初始权重逐位一致', same( wa, wb ), `digest 相同=${ digestOfWeights( wa ) === digestOfWeights( wb ) }` );
    check( '异 seed 初始权重不同', !same( wa, wc ), `digest ${ digestOfWeights( wa ) } vs ${ digestOfWeights( wc ) }` );
    check( '参数量符合规格', a.paramCount() === vocab.length * 16 + 8 * 16 * 64 + 64 + 64 * vocab.length + vocab.length,
      `params=${ a.paramCount().toLocaleString() }` );
  }

  // ---------------------------------------------------------------- 5. 探针可复现
  {
    const probe = pickProbe( { name: 't', text, vocab }, 1024 );
    const probeIds = encodeTo( text.slice( probe.offset, probe.offset + probe.size ), stoi );
    const a = new TinyMlpEngine( spec );
    const b = new TinyMlpEngine( spec );
    b.setWeights( await a.getWeights() );
    const la = await a.evalAt( probeIds, spec.ctx, 256 );
    const lb = await b.evalAt( probeIds, spec.ctx, 256 );
    check( '共识探针评估可复现', Math.abs( la - lb ) === 0,
      `probe offset=${ probe.offset }，两次评估 loss=${ la.toFixed( 6 ) } / ${ lb.toFixed( 6 ) }` );
    check( '初始 loss 接近 ln(V)', Math.abs( la - Math.log( vocab.length ) ) < 0.25,
      `初始 loss=${ la.toFixed( 3 ) }，ln(${ vocab.length })=${ Math.log( vocab.length ).toFixed( 3 ) }` );
  }

  // ---------------------------------------------------------------- 6. 联邦 vs 单节点独训 / 两种聚合方式
  {
    const shards = 2;
    const rounds = 6;
    const stepsPerRound = 10;
    const lr = 0.05;
    const T = spec.ctx;
    const probe = pickProbe( { name: 't', text, vocab }, 1024 );
    const probeIds = encodeTo( text.slice( probe.offset, probe.offset + probe.size ), stoi );
    const pool = [ 0, 1 ].map( ( i ) => encodeTo( shardText( text, i, shards ), stoi ) );

    const init = new TinyMlpEngine( spec );
    const baseline = await init.evalAt( probeIds, T, 256 );

    /** 跑一轮完整的联合训练，聚合方式可插拔。返回最终探针 loss 与权重。 */
    const runFederated = async ( agg: AggregateSpec ): Promise<{ loss: number; weights: NamedWeights }> =>
    {
      const A = new TinyMlpEngine( spec );
      const B = new TinyMlpEngine( spec );
      const state: AggregateState = { momentum: null };
      let base = await A.getWeights();
      const sizes = [ stepsPerRound * 32, stepsPerRound * 32 ];
      for ( let r = 0; r < rounds; r++ )
      {
        for ( let s = 0; s < stepsPerRound; s++ )
        {
          await A.trainBatch( pool[ 0 ], 32, lr );
          await B.trainBatch( pool[ 1 ], 32, lr );
        }
        const global = aggregateGlobal( base, [ await A.getWeights(), await B.getWeights() ], sizes, agg, state );
        A.setWeights( global );
        B.setWeights( global );
        base = global;
      }
      return { loss: await A.evalAt( probeIds, T, 256 ), weights: await A.getWeights() };
    };

    const fed = await runFederated( { mode: 'fedavg', outerLr: 1, momentum: 0 } );
    const diloco = await runFederated( { mode: 'diloco', outerLr: 0.7, momentum: 0.9 } );

    // 单节点独训：同样总步数，只用自己那一片（对照）
    const soloA = new TinyMlpEngine( spec );
    const soloB = new TinyMlpEngine( spec );
    for ( let s = 0; s < rounds * stepsPerRound; s++ )
    {
      await soloA.trainBatch( pool[ 0 ], 32, lr );
      await soloB.trainBatch( pool[ 1 ], 32, lr );
    }
    const soloLossA = await soloA.evalAt( probeIds, T, 256 );
    const soloLossB = await soloB.evalAt( probeIds, T, 256 );
    const soloAvg = ( soloLossA + soloLossB ) / 2;

    check( '联邦训练确实降低了探针 loss', fed.loss < baseline - 0.5,
      `初始 ${ baseline.toFixed( 3 ) } → 联邦(FedAvg) ${ fed.loss.toFixed( 3 ) }（降 ${ ( baseline - fed.loss ).toFixed( 3 ) }）` );
    check( '联邦优于单节点独训', fed.loss < soloAvg,
      `联邦 ${ fed.loss.toFixed( 3 ) } vs 独训均值 ${ soloAvg.toFixed( 3 ) }（A ${ soloLossA.toFixed( 3 ) } / B ${ soloLossB.toFixed( 3 ) }）` );
    check( 'DiLoCo 式增量聚合同样有效', diloco.loss < baseline - 0.5,
      `FedAvg ${ fed.loss.toFixed( 3 ) } · DiLoCo(η=0.7,β=0.9) ${ diloco.loss.toFixed( 3 ) } · 基线 ${ baseline.toFixed( 3 ) }` );

    // 聚合确实在起作用（而非空转）
    const d = l2Distance( fed.weights, await soloA.getWeights() );
    check( '联邦模型与独训模型确有差异', d > 1e-3, `||w_fed - w_soloA||₂=${ d.toFixed( 4 ) }` );
  }

  // ---------------------------------------------------------------- 7. 聚合数学（纯函数）
  {
    const vecEq = ( x: Float32Array, y: Float32Array, tol = 1e-6 ): boolean =>
      x.length === y.length && [ ...x ].every( ( v, i ) => Math.abs( v - y[ i ] ) < tol );

    const base: NamedWeights = { p: Float32Array.from( [ 0, 0 ] ) };
    const wa: NamedWeights = { p: Float32Array.from( [ 1, 3 ] ) };
    const wb: NamedWeights = { p: Float32Array.from( [ 5, 1 ] ) };
    const direct = fedAvg( [ wa, wb ], [ 1, 1 ] );

    const s1: AggregateState = { momentum: null };
    const viaFed = aggregateGlobal( base, [ wa, wb ], [ 1, 1 ], { mode: 'fedavg', outerLr: 1, momentum: 0 }, s1 );
    const s2: AggregateState = { momentum: null };
    const viaDiloco = aggregateGlobal( base, [ wa, wb ], [ 1, 1 ], { mode: 'diloco', outerLr: 1, momentum: 0 }, s2 );
    check( '聚合骨架与直接 FedAvg 等价',
      vecEq( viaFed.p, direct.p ) && vecEq( viaDiloco.p, direct.p ),
      `直接 [${ [ ...direct.p ] }] · fedavg [${ [ ...viaFed.p ] }] · diloco(β=0,η=1) [${ [ ...viaDiloco.p ] }]` );

    // 外层动量应跨轮累积：M ← β·M + Δ̄
    const s3: AggregateState = { momentum: null };
    const agg: AggregateSpec = { mode: 'diloco', outerLr: 1, momentum: 0.9 };
    aggregateGlobal( base, [ wa, wb ], [ 1, 1 ], agg, s3 );
    const m1 = Float32Array.from( s3.momentum!.p );
    aggregateGlobal( base, [ wa, wb ], [ 1, 1 ], agg, s3 );
    const m2 = s3.momentum!.p;
    check( 'DiLoCo 外层动量跨轮累积', vecEq( m2, Float32Array.from( [ ...m1 ].map( ( v ) => 0.9 * v + v ) ) ),
      `第 1 轮 M=[${ [ ...m1 ] }] → 第 2 轮 M=[${ [ ...m2 ] }]（期望 1.9 倍）` );
  }

  // ---------------------------------------------------------------- 7. 分片覆盖
  {
    const parts = [ 0, 1, 2, 3 ].map( ( i ) => shardText( text, i, 4 ) );
    const joined = parts.join( '' );
    check( '分片拼回等于原文', joined === text,
      `4 片长度 ${ parts.map( ( p ) => p.length ).join( ' + ' ) } = ${ joined.length }` );
    check( '语料指纹稳定', corpusDigest( { name: 'a', text, vocab } ) === corpusDigest( { name: 'a', text, vocab } ),
      `digest=${ corpusDigest( { name: 'a', text, vocab } ) }` );
  }

  return results;
}
