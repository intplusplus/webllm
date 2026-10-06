/**
 * 联邦训练的端到端测试（无头，不依赖浏览器）。
 *
 * selfcheck.ts 测的是「零件」（权重帧、FedAvg、引擎收敛）。
 * 这里测的是**编排**：三个 FedNode 通过三条 LocalBus 连成一个房间，
 * 真跑一遍 开轮 → 本地训练 → 上报 → 探针校验 → FedAvg → 广播 → 账本 → 模型卡。
 *
 * 第三个节点是**故意作弊的**：它把上报的 probeLoss 改掉（模拟谎报）。
 * 期望结果是主机抓出来、剔除它、并如实记进账本 —— 这是信任层 v0 唯一的核心主张，
 * 必须有测试守住，否则它就只是一句宣传语。
 */
import { LocalBus } from './bus';
import { corpusDigest, pickProbe, type Corpus } from './corpus';
import { detectKind } from './capability';
import { FedNode, type ModelCard } from './node';
import {
  decodeWeights,
  encodeWeights,
  fingerprintOf,
  weightsDigest,
  type ControlMessage,
  type MlpModelSpec,
  type RoomManifest,
} from './protocol';
import type { PeerInfo } from './transport';

export interface E2EResult
{
  name: string;
  pass: boolean;
  detail: string;
}

const ROUNDS = 3;
const LOCAL_STEPS = 6;
const BATCH = 16;
const SHARDS = 4;
/** 从第几轮开始作弊 */
const CHEAT_FROM_ROUND = 2;

/**
 * 会篡改上报帧的节点：把 meta.probeLoss 抬高 1.0，模拟谎报。
 * 它从权重帧自带的 meta.round 判断当前轮次，因此不需要额外同步轮次状态。
 */
class TamperingBus extends LocalBus
{
  tamper = false;

  send ( peerId: string, data: string | ArrayBuffer ): boolean
  {
    if ( this.tamper && typeof data !== 'string' )
    {
      try
      {
        const dec = decodeWeights( data );
        if ( dec.meta && dec.meta.round >= CHEAT_FROM_ROUND )
        {
          const lying = { ...dec.meta, probeLoss: dec.meta.probeLoss + 1 };
          return super.send( peerId, encodeWeights( dec.weights, lying ) );
        }
      }
      catch { /* 不是权重帧就走原路 */ }
    }
    return super.send( peerId, data );
  }
}

function makeManifest ( corpus: Corpus, spec: MlpModelSpec, engineReason = '测试固定 mlp' ): RoomManifest
{
  const base = {
    roomId: 'e2e-room',
    taskName: '字符级语言模型 · 联邦预训练',
    taskBrief: '端到端测试',
    model: spec,
    engineReason,
    rounds: ROUNDS,
    localSteps: LOCAL_STEPS,
    batchSize: BATCH,
    lr: 0.05,
    shards: SHARDS,
    corpus: { name: corpus.name, digest: corpusDigest( corpus ), chars: corpus.text.length, vocab: corpus.vocab },
    probe: pickProbe( corpus, 1024 ),
    // 这个测试专门测「能不能抓住谎报」，所以交叉校验必须开
    crossCheck: true,
    aggregate: { mode: 'fedavg' as const, outerLr: 1, momentum: 0 },
    createdAt: Date.now(),
  };
  return { ...base, fingerprint: fingerprintOf( base ) };
}

interface NodeBox
{
  peer: PeerInfo;
  bus: LocalBus;
  node: FedNode | null;
  roundsSeen: number[];
  done: ModelCard | null;
  /** 收到的所有状态文案 —— 「开训后拒绝迟到节点」这类断言只能靠它验 */
  statuses: string[];
}

function makeBox (
  roomId: string,
  peer: PeerInfo,
  factory: ( opts: ConstructorParameters<typeof LocalBus>[ 0 ] ) => LocalBus,
): NodeBox
{
  const box: NodeBox = { peer, bus: null as unknown as LocalBus, node: null, roundsSeen: [], done: null, statuses: [] };
  box.bus = factory( {
    roomId,
    self: peer,
    onControl: ( from, msg ) => box.node?.onControl( from, msg as ControlMessage ),
    onBinary: ( from, buf ) => box.node?.onBinary( from, buf ),
    onPeerOpen: ( from ) => box.node?.onPeerOpen( from ),
    onPeerClose: ( from ) => box.node?.onPeerClose( from ),
    onRoster: ( peers ) => box.node?.onRoster( peers ),
    onStatus: ( s ) => box.statuses.push( s ),
  } );
  return box;
}

function makeEvents ( box: NodeBox )
{
  return {
    onStatus: ( t: string ) => { box.statuses.push( t ); },
    onLog: () => { /* 忽略 */ },
    onRoster: () => { /* 由 FedNode.onRoster 驱动，这里无需额外处理 */ },
    onManifest: () => { /* 断言直接读 node.currentManifest */ },
    onRound: ( stats: { round: number } ) => { box.roundsSeen.push( stats.round ); },
    onCurve: () => { /* 曲线与正确性无关 */ },
    onDone: ( card: ModelCard ) => { box.done = card; },
  };
}

export async function runFedE2E ( text: string ): Promise<{ results: E2EResult[]; card: ModelCard | null }>
{
  const results: E2EResult[] = [];
  const check = ( name: string, pass: boolean, detail: string ): void => { results.push( { name, pass, detail } ); };

  const corpus: Corpus = { name: 'E2E 语料（Tiny Shakespeare）', text, vocab: [ ...new Set( text ) ].sort() };
  const spec: MlpModelSpec = { engine: 'mlp', vocabSize: corpus.vocab.length, ctx: 8, embDim: 16, hidden: 64, seed: 777 };
  const manifest = makeManifest( corpus, spec );

  const device = { kind: detectKind(), webgpu: false, cores: 4, memoryGB: 0, ua: 'headless' };
  const peerA: PeerInfo = { peerId: 'aaaa0001', name: '主机', role: 'host', device, joinedAt: Date.now() };
  const peerB: PeerInfo = { peerId: 'bbbb0002', name: '好节点', role: 'peer', device, joinedAt: Date.now() };
  const peerC: PeerInfo = { peerId: 'cccc0003', name: '作弊节点', role: 'peer', device, joinedAt: Date.now() };

  const boxA = makeBox( manifest.roomId, peerA, ( o ) => new LocalBus( o ) );
  const boxB = makeBox( manifest.roomId, peerB, ( o ) => new LocalBus( o ) );
  const boxC = makeBox( manifest.roomId, peerC, ( o ) => new TamperingBus( o ) );
  const tamperBus = boxC.bus as TamperingBus;
  tamperBus.tamper = true; // 只对 round >= CHEAT_FROM_ROUND 生效

  boxA.node = new FedNode( { role: 'host', transport: boxA.bus, events: makeEvents( boxA ), corpus, contributionText: '', manifest } );
  boxB.node = new FedNode( { role: 'peer', transport: boxB.bus, events: makeEvents( boxB ), corpus, contributionText: 'oooo\n' } );
  boxC.node = new FedNode( { role: 'peer', transport: boxC.bus, events: makeEvents( boxC ), corpus, contributionText: '' } );

  await boxA.bus.connect();
  await boxB.bus.connect();
  await boxC.bus.connect();
  await delay( 80 ); // 让 present 握手走完

  check( '节点互相发现', boxA.bus.peerInfos.length === 2 && boxB.bus.peerInfos.length === 2,
    `主机看到 ${ boxA.bus.peerInfos.length } 个节点，B 看到 ${ boxB.bus.peerInfos.length } 个（期望各 2）` );

  const initialDigest = weightsDigest( await boxA.node.engineRef.getWeights() );

  let ok = false;
  try
  {
    await Promise.race( [
      boxA.node.startHost( ( engine, reason ) =>
      {
        // 无头 Node 环境必然拿不到 WebGPU 适配器，协商结果应该是 mlp。
        // 这里显式断言：如果哪天协商逻辑跑偏，要立刻炸出来，而不是默默换引擎。
        if ( engine !== 'mlp' ) throw new Error( `无头端到端测试预期协商出 mlp，实得 ${ engine }` );
        return makeManifest( corpus, spec, reason );
      } ),
      delay( 90000 ).then( () => { throw new Error( '端到端流程超时（90s）' ); } ),
    ] );
    await delay( 120 ); // 等最后一轮的广播与 round/close 送达各节点
    ok = true;
  }
  catch ( err )
  {
    check( '端到端流程完成', false, ( err as Error ).message );
  }

  const card = await boxA.node.buildCard( manifest );
  if ( ok ) check( '端到端流程完成', true, `${ ROUNDS } 轮跑完，主机账本共 ${ card.ledger.length } 条记录` );

  const ledger = card.ledger;
  const entryOf = ( peer: string, round: number ): typeof ledger[ number ] | undefined =>
    ledger.find( ( e ) => e.peerId === peer && e.round === round );

  // --- 任务分配 ---
  check( '节点收到任务清单', boxB.node.currentManifest !== null && boxC.node.currentManifest !== null,
    `分片：主机 #${ boxA.node.shard } / B #${ boxB.node.shard } / C #${ boxC.node.shard }` );
  check( '各节点分片互不重叠', new Set( [ boxA.node.shard, boxB.node.shard, boxC.node.shard ] ).size === 3,
    `分片 ${ boxA.node.shard } / ${ boxB.node.shard } / ${ boxC.node.shard }` );

  // --- 房主协调：引擎按全网能力协商 + 开训前就绪握手 ---
  const sealed = boxA.node.currentManifest;
  check( '引擎按全网能力下限协商（无头环境 → mlp）',
    sealed?.model.engine === 'mlp',
    `协商结果 ${ sealed?.model.engine } —— ${ sealed?.engineReason ?? '（清单里没写原因）' }` );
  check( '开训前所有节点完成就绪握手',
    boxA.node.readyIds.length === 2 && boxA.node.notReadyReasons.size === 0,
    `就绪 ${ boxA.node.readyIds.length } 个，未就绪 ${ boxA.node.notReadyReasons.size } 个` );

  // --- 账本完整性 ---
  const roundsRecorded = new Set( ledger.map( ( e ) => e.round ) );
  const peersRecorded = new Set( ledger.map( ( e ) => e.peerId ) );
  check( '账本覆盖每一轮每个节点',
    roundsRecorded.size === ROUNDS && peersRecorded.size === 3 && ledger.length === ROUNDS * 3,
    `轮次 ${ [ ...roundsRecorded ].join( ',' ) }，节点 ${ peersRecorded.size } 个，共 ${ ledger.length } 条（期望 ${ ROUNDS * 3 }）` );

  // --- 信任层：作弊必须被抓 ---
  const c1 = entryOf( peerC.peerId, 1 );
  const c2 = entryOf( peerC.peerId, 2 );
  const c3 = entryOf( peerC.peerId, 3 );
  check( '作弊节点在老实轮判为通过', c1?.verdict === 'ok',
    `第 1 轮 C：${ c1?.verdict } —— ${ c1?.note }` );
  check( '谎报 probeLoss 被探针复算抓出并剔除',
    c2?.verdict === 'suspect' && c3?.verdict === 'suspect',
    `第 2/3 轮 C：${ c2?.verdict } / ${ c3?.verdict } —— ${ c2?.note }` );
  const badShares = ledger.filter( ( e ) => e.verdict !== 'ok' ).reduce( ( a, e ) => a + e.share, 0 );
  check( '被剔除的节点份额为 0', badShares === 0, `异常记录份额合计 = ${ badShares }` );

  const bEntries = ledger.filter( ( e ) => e.peerId === peerB.peerId );
  check( '好节点全程通过', bEntries.length === ROUNDS && bEntries.every( ( e ) => e.verdict === 'ok' ),
    `B 共 ${ bEntries.length } 轮，全部通过=${ bEntries.every( ( e ) => e.verdict === 'ok' ) }` );

  // --- 聚合确实生效 ---
  const sharesOf = ( round: number ): number[] =>
    ledger.filter( ( e ) => e.round === round && e.verdict === 'ok' ).map( ( e ) => e.share );
  const r1 = sharesOf( 1 );
  const r2 = sharesOf( 2 );
  const r3 = sharesOf( 3 );
  const sum = ( a: number[] ): number => a.reduce( ( x, y ) => x + y, 0 );
  check( '每轮通过者份额合计为 1',
    [ r1, r2, r3 ].every( ( list ) => Math.abs( sum( list ) - 1 ) < 1e-9 ),
    `r1 ${ r1.length } 人 × ${ r1[ 0 ]?.toFixed( 3 ) }；r2 ${ r2.length } 人 × ${ r2[ 0 ]?.toFixed( 3 ) }；r3 ${ r3.length } 人 × ${ r3[ 0 ]?.toFixed( 3 ) }` );
  check( '剔除作弊节点后聚合范围随之收缩',
    r1.length === 3 && r2.length === 2 && r3.length === 2 &&
    Math.abs( r1[ 0 ] - 1 / 3 ) < 1e-9 && Math.abs( r2[ 0 ] - 0.5 ) < 1e-9,
    `r1 全员参与（3 人各 1/3）→ r2/r3 降至 2 人各 1/2` );

  // --- 训练有效 ---
  check( '联邦训练降低了探针 loss', card.finalProbeLoss < card.initialProbeLoss - 0.2,
    `初始 ${ card.initialProbeLoss.toFixed( 3 ) } → 最终 ${ card.finalProbeLoss.toFixed( 3 ) }` );

  // --- 全网一致 ---
  const digA = weightsDigest( await boxA.node.engineRef.getWeights() );
  const digB = weightsDigest( await boxB.node.engineRef.getWeights() );
  const digC = weightsDigest( await boxC.node.engineRef.getWeights() );
  check( '广播后所有节点收敛到同一份全局权重', digA === digB && digB === digC,
    `摘要 A=${ digA } B=${ digB } C=${ digC }` );
  check( '全局权重确实被更新（非空转）', digA !== initialDigest,
    `初始 ${ initialDigest } → 最终 ${ digA }` );

  // --- 收尾事件 ---
  check( '节点侧也收到完成事件并生成模型卡', boxB.done !== null && boxB.done.rounds === ROUNDS,
    boxB.done ? `B 的模型卡：${ boxB.done.rounds } 轮，最终 loss ${ boxB.done.finalProbeLoss.toFixed( 3 ) }` : 'B 未收到 onDone' );
  check( '模型卡记录了贡献者与流量', card.contributors.length === 3 && card.transportBytes > 0,
    `贡献者 ${ card.contributors.length } 人，传输 ${ ( card.transportBytes / 1024 ).toFixed( 1 ) } KB，参数量 ${ card.paramCount.toLocaleString() }` );

  // --- 开训后锁房：迟到的节点必须被明确拒绝，而不是静默旁观 ---
  // 没有这一条，就会出现「房主跑到第 3 轮、手机还停在第 0 轮」这种错位 ——
  // 用户看到的是「两边不同步」，而且不知道为什么。
  const peerD: PeerInfo = {
    peerId: 'dddd0004',
    name: '迟到节点',
    role: 'peer',
    device: { ...device, kind: 'Android 手机' },
    joinedAt: Date.now(),
  };
  const boxD = makeBox( manifest.roomId, peerD, ( o ) => new LocalBus( o ) );
  boxD.node = new FedNode( { role: 'peer', transport: boxD.bus, events: makeEvents( boxD ), corpus, contributionText: '' } );
  await boxD.bus.connect();
  await delay( 250 );
  check( '开训后加入的节点被明确拒绝，不会默默跟着跑',
    boxD.statuses.some( ( s ) => s.includes( '房间已锁定' ) ) && boxD.node.currentManifest === null,
    `节点最后状态「${ boxD.statuses[ boxD.statuses.length - 1 ] ?? '（无）' }」，清单=${ boxD.node.currentManifest ? '有' : '无' }` );
  boxD.bus.close();
  await delay( 40 );

  boxA.bus.close(); boxB.bus.close(); boxC.bus.close();
  await delay( 40 );
  return { results, card };
}

function delay ( ms: number ): Promise<void>
{
  return new Promise( ( r ) => globalThis.setTimeout( r, ms ) );
}
