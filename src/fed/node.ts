/**
 * 联邦训练节点 —— 公共训练网络的运行核心。
 *
 * 一个「训练房间」= 星形拓扑：
 *   汇聚节点（host）负责开轮、校验、FedAvg、广播全局权重；
 *   普通节点（peer）负责本地训练、上报权重、接收全局权重。
 *   P2P 通道内除了权重没有任何中心服务器参与。
 *
 * 一轮的完整时序：
 *   host  →  {round/open r}                广播
 *   各节点  本地 trainBatch × localSteps    在自己的数据分片上
 *   各节点 →  [权重帧(含 meta)]             上报 localLoss / probeLoss / Δ范数
 *   host    校验：用提交的权重在「共识探针」上复算 loss，比对节点自报值
 *   host    FedAvg（按样本数加权）→ 覆盖自己的权重 → 复算全局探针 loss
 *   host  →  [全局权重帧] + {round/close}   广播
 *
 * 「信任层 v0」做的四件事（对应 docs 里的 Phase 1）：
 *   1. 清单指纹 —— 所有节点对「训什么模型、用什么数据」达成一致，可事后复核
 *   2. 共识探针 —— 用固定数据窗口复算提交权重，能抓住伪造 loss / 掉包权重
 *   3. 贡献账本 —— 每节点每轮的样本数、token 数、Δ范数、份额、裁决，全部留痕
 *   4. 可复现性 —— 固定种子 + 确定性评估，同清单同贡献顺序 → 同一全局模型
 *
 * 局限（明确写在账本与模型卡里，不假装它是完整的）：FNV-1a 非密码学摘要、
 * 无位置无关验证、无质押与惩罚。完整方案见规划文档「信任层」路线图。
 */
import type { Corpus } from './corpus';
import { buildStoi, encodeTo, filterToVocab, shardText } from './corpus';
import { createEngine, type TrainEngine } from './engine';
import {
  decodeWeights,
  encodeWeights,
  fedAvg,
  l2Distance,
  weightsDigest,
  type ControlMessage,
  type LedgerEntry,
  type NamedWeights,
  type RoomManifest,
  type RoundStats,
  type WeightMeta,
} from './protocol';
import type { PeerInfo, RoomTransport } from './transport';

const PROBE_EVAL_COUNT = 256;
const VERIFY_TOL = 1e-3;
const ROUND_TIMEOUT_MS = 30000;

export interface ModelCard
{
  manifest: RoomManifest;
  startedAt: number;
  finishedAt: number;
  rounds: number;
  /** 初始（未训练）模型在探针上的 loss */
  initialProbeLoss: number;
  /** 第一轮聚合后 */
  firstProbeLoss: number;
  finalProbeLoss: number;
  paramCount: number;
  weightsDigest: string;
  transportBytes: number;
  ledger: LedgerEntry[];
  contributors: Array<{
    peerId: string;
    name: string;
    deviceKind: string;
    samples: number;
    tokens: number;
    rounds: number;
    ok: number;
    suspect: number;
    contributedChars: number;
  }>;
  note: string;
}

export interface NodeEvents
{
  onStatus: ( text: string ) => void;
  onLog: ( line: string ) => void;
  onRoster: ( peers: PeerInfo[], selfId: string ) => void;
  onManifest: ( manifest: RoomManifest, shardIndex: number ) => void;
  onRound: ( stats: RoundStats, totalRounds: number ) => void;
  onCurve: ( point: { round: number; local: number; global: number } ) => void;
  onDone: ( card: ModelCard ) => void;
}

export interface NodeOptions
{
  role: 'host' | 'peer';
  transport: RoomTransport;
  events: NodeEvents;
  corpus: Corpus;
  /** 用户贡献的文本（可为空）；并入本节点本地训练池 */
  contributionText: string;
  /** 仅 host：本地构建的清单 */
  manifest?: RoomManifest;
}

interface Submission
{
  peerId: string;
  name: string;
  deviceKind: string;
  weights: NamedWeights;
  meta: WeightMeta;
  bytes: number;
}

export class FedNode
{
  private readonly o: NodeOptions;
  private readonly stoi: Map<string, number>;

  /** 引擎在清单确定后构造：host 在构造期，peer 收到 assign 时。 */
  private engine!: TrainEngine;
  private readonly engineFactory = createEngine;
  private manifest: RoomManifest | null;
  private shardIndex = 0;
  // 显式标注为 Uint32Array：TS 5.7+ 起 TypedArray 带 ArrayBufferLike 泛型，
  // 不标注会被推断成过窄的 Uint32Array<ArrayBuffer>，接不住 encodeTo 的返回值。
  private trainIds: Uint32Array = new Uint32Array( 0 );
  private probeIds: Uint32Array = new Uint32Array( 0 );

  /** 上一轮结束时的全局权重（用于算 Δ 范数） */
  private prevGlobal: NamedWeights | null = null;
  private readonly submissions = new Map<string, Submission>();
  private readonly ledger: LedgerEntry[] = [];
  private readonly credits = new Map<string, { chars: number; digest: string }>();
  private rosterMap = new Map<string, PeerInfo>();
  private hostId: string | null = null;
  private assignCounter = 0;

  private startedAt = 0;
  private running = false;
  private stopped = false;
  private initialProbeLoss = 0;
  private firstProbeLoss = 0;
  private lastProbeLoss = 0;
  private transportBytes = 0;

  constructor ( opts: NodeOptions )
  {
    this.o = opts;
    this.manifest = opts.manifest ?? null;
    this.stoi = buildStoi( opts.corpus.vocab );
    if ( this.manifest ) this.adoptManifest( this.manifest, 0 );
  }

  get currentManifest (): RoomManifest | null { return this.manifest; }
  get vocab (): string[] { return this.o.corpus.vocab; }
  get shard (): number { return this.shardIndex; }
  get isRunning (): boolean { return this.running; }
  get ledgerEntries (): LedgerEntry[] { return this.ledger; }
  get selfId (): string { return this.o.transport.selfId; }
  get engineRef () { return this.engine; }

  // ---------------------------------------------------------------- 装配

  private adoptManifest ( m: RoomManifest, shardIndex: number ): void
  {
    this.manifest = m;
    this.shardIndex = shardIndex;
    this.engine = this.engineFactory( m.model );

    const shard = shardText( this.o.corpus.text, shardIndex, m.shards );
    const filtered = filterToVocab( this.o.contributionText || '', this.stoi );
    const pool = filtered.text.length > 64 ? `${ shard }${ filtered.text }` : shard;
    this.trainIds = encodeTo( pool, this.stoi );

    const probeText = this.o.corpus.text.slice( m.probe.offset, m.probe.offset + m.probe.size );
    this.probeIds = encodeTo( probeText, this.stoi );
    if ( this.probeIds.length < m.model.ctx + 32 ) throw new Error( '共识探针数据不足' );

    if ( filtered.text.length > 64 )
    {
      this.credits.set( this.selfId, { chars: filtered.text.length, digest: m.corpus.digest } );
      this.announceCredit();
    }

    this.o.events.onManifest( m, shardIndex );
    this.o.events.onLog(
      `本节点分片 #${ shardIndex }／${ m.shards }，训练池 ${ this.trainIds.length } 字符` +
      ( filtered.dropped > 0 ? `（贡献文本已过滤 ${ filtered.dropped } 个词表外字符）` : '' ),
    );
  }

  private announceCredit (): void
  {
    if ( this.o.role !== 'peer' ) return;
    const target = this.hostId ?? this.o.transport.openPeerIds[ 0 ];
    const c = this.credits.get( this.selfId );
    if ( target && c ) this.o.transport.sendControl( target, { t: 'credit', peerId: this.selfId, chars: c.chars, digest: c.digest } );
  }

  // ---------------------------------------------------------------- 传输回调

  onControl ( peerId: string, msg: ControlMessage ): void
  {
    if ( this.o.role === 'peer' )
    {
      this.hostId = peerId;
      switch ( msg.t )
      {
        case 'assign':
          if ( !this.manifest )
          {
            this.adoptManifest( msg.manifest, msg.shardIndex );
            this.o.events.onStatus( '已收到任务分配，等待主机开轮…' );
          }
          break;
        case 'round/open':
          void this.peerRound( msg.round );
          break;
        case 'round/close':
        {
          const roster = this.manifest?.rounds ?? 0;
          this.lastProbeLoss = msg.stats.globalLoss;
          if ( this.firstProbeLoss === 0 ) this.firstProbeLoss = msg.stats.globalLoss;
          this.ledger.push( ...msg.stats.entries );
          this.o.events.onRound( msg.stats, roster );
          if ( roster > 0 && msg.stats.round >= roster ) this.finish();
          break;
        }
        default:
          break;
      }
      return;
    }

    if ( msg.t === 'credit' )
    {
      this.credits.set( msg.peerId, { chars: msg.chars, digest: msg.digest } );
      this.o.events.onLog( `${ msg.peerId.slice( 0, 6 ) } 贡献数据 ${ msg.chars } 字符` );
    }
  }

  onBinary ( peerId: string, buf: ArrayBuffer ): void
  {
    this.transportBytes += buf.byteLength;

    if ( this.o.role === 'peer' )
    {
      try
      {
        const dec = decodeWeights( buf );
        this.engine.setWeights( dec.weights );
        this.prevGlobal = dec.weights;
      }
      catch ( err )
      {
        this.o.events.onLog( `接收全局权重失败：${ ( err as Error ).message }` );
      }
      return;
    }

    try
    {
      const dec = decodeWeights( buf );
      if ( !dec.meta ) throw new Error( '提交帧缺少元信息' );
      const peer = this.rosterMap.get( peerId );
      this.submissions.set( peerId, {
        peerId,
        name: peer?.name ?? peerId.slice( 0, 6 ),
        deviceKind: peer?.device.kind ?? '未知',
        weights: dec.weights,
        meta: dec.meta,
        bytes: buf.byteLength,
      } );
    }
    catch ( err )
    {
      this.o.events.onLog( `来自 ${ peerId.slice( 0, 6 ) } 的权重帧无法解析：${ ( err as Error ).message }` );
      const peer = this.rosterMap.get( peerId );
      this.submissions.set( peerId, {
        peerId,
        name: peer?.name ?? peerId.slice( 0, 6 ),
        deviceKind: peer?.device.kind ?? '未知',
        weights: {},
        meta: { round: 0, peerId, samples: 0, tokens: 0, localLoss: 0, probeLoss: 0, deltaNorm: 0, digest: '' },
        bytes: buf.byteLength,
      } );
    }
  }

  onPeerOpen ( peerId: string ): void
  {
    this.o.events.onLog( `P2P 通道就绪：${ peerId.slice( 0, 6 ) }` );
    if ( this.o.role === 'peer' )
    {
      this.hostId = peerId;
      this.announceCredit();
      return;
    }
    // 主机：新节点一上线就下发任务分配
    this.sendAssign( peerId );
  }

  onPeerClose ( peerId: string ): void
  {
    this.submissions.delete( peerId );
  }

  onRoster ( peers: PeerInfo[] ): void
  {
    this.rosterMap = new Map( peers.map( ( p ) => [ p.peerId, p ] ) );
    this.o.events.onRoster( peers, this.selfId );
  }

  private sendAssign ( peerId: string ): void
  {
    const m = this.manifest;
    if ( !m ) return;
    const span = Math.max( 1, m.shards - 1 );
    const idx = ( this.assignCounter % span ) + 1;
    this.assignCounter += 1;
    this.o.transport.sendControl( peerId, { t: 'assign', shardIndex: idx, manifest: m } );
  }

  // ---------------------------------------------------------------- 本地训练

  private async localPhase ( round: number ): Promise<{ weights: NamedWeights; meta: WeightMeta }>
  {
    const m = this.manifest;
    if ( !m ) throw new Error( 'localPhase: 清单尚未确定' );
    const before = this.prevGlobal ?? this.engine.getWeights();

    let loss = 0;
    let tokens = 0;
    for ( let s = 0; s < m.localSteps; s++ )
    {
      const r = await this.engine.trainBatch( this.trainIds, m.batchSize, m.lr );
      loss += r.loss;
      tokens += r.tokens;
    }
    loss /= m.localSteps;

    const weights = this.engine.getWeights();
    const probeLoss = await this.engine.evalAt( this.probeIds, m.model.ctx, PROBE_EVAL_COUNT );
    const meta: WeightMeta = {
      round,
      peerId: this.selfId,
      samples: m.localSteps * m.batchSize,
      tokens,
      localLoss: loss,
      probeLoss,
      deltaNorm: l2Distance( weights, before ),
      digest: weightsDigest( weights ),
    };
    return { weights, meta };
  }

  // ---------------------------------------------------------------- 主机主循环

  async startHost (): Promise<void>
  {
    if ( this.o.role !== 'host' ) throw new Error( 'startHost: 仅主机可用' );
    if ( this.running || !this.manifest ) return;
    this.running = true;
    this.stopped = false;
    this.startedAt = Date.now();
    const m = this.manifest;

    // 初始（未训练）模型基线
    this.initialProbeLoss = await this.engine.evalAt( this.probeIds, m.model.ctx, PROBE_EVAL_COUNT );
    this.o.events.onLog( `初始模型探针 loss = ${ this.initialProbeLoss.toFixed( 3 ) }（未训练的基线）` );

    for ( const p of this.rosterMap.values() ) if ( p.role === 'peer' ) this.sendAssign( p.peerId );

    this.o.events.onStatus( `主机启动：共 ${ m.rounds } 轮，每轮每节点本地 ${ m.localSteps } 步 × ${ m.batchSize } 序列` );

    try
    {
      for ( let r = 1; r <= m.rounds; r++ )
      {
        if ( this.stopped ) break;
        await this.hostRound( r );
      }
      if ( !this.stopped ) this.finish();
    }
    catch ( err )
    {
      this.o.events.onStatus( `主机循环中断：${ ( err as Error ).message }` );
    }
    finally
    {
      this.running = false;
    }
  }

  private async hostRound ( round: number ): Promise<void>
  {
    const m = this.manifest!;
    const t0 = performance.now();
    this.submissions.clear();

    this.o.transport.broadcast( JSON.stringify( { t: 'round/open', round } satisfies ControlMessage ) );

    const mine = await this.localPhase( round );
    this.o.events.onStatus( `第 ${ round }/${ m.rounds } 轮 · 本机完成（loss ${ mine.meta.localLoss.toFixed( 3 ) }），等待节点上报…` );

    const peers = [ ...this.rosterMap.values() ].filter( ( p ) => p.role === 'peer' );
    const deadline = performance.now() + ROUND_TIMEOUT_MS;
    while ( this.submissions.size < peers.length && performance.now() < deadline && !this.stopped )
    {
      await sleep( 120 );
    }

    const entries: LedgerEntry[] = [];
    const accepted: NamedWeights[] = [ mine.weights ];
    const sizes: number[] = [ mine.meta.samples ];

    const mineVerdict = await this.verify( mine.meta, mine.weights );
    entries.push( this.entryFor( round, this.selfId, '本机（主机）', this.deviceKindOf( this.selfId ), mine.meta, mineVerdict.ok, mineVerdict.note ) );

    for ( const p of peers )
    {
      const sub = this.submissions.get( p.peerId );
      if ( !sub || Object.keys( sub.weights ).length === 0 )
      {
        entries.push( {
          round, peerId: p.peerId, name: p.name, deviceKind: p.device.kind,
          samples: 0, tokens: 0, probeLoss: 0, localLoss: 0, deltaNorm: 0, share: 0,
          verdict: 'timeout', note: '本轮未上报（超时或断连），已从聚合中剔除',
        } );
        continue;
      }
      const v = await this.verify( sub.meta, sub.weights );
      entries.push( this.entryFor( round, p.peerId, sub.name, sub.deviceKind, sub.meta, v.ok, v.note ) );
      if ( v.ok )
      {
        accepted.push( sub.weights );
        sizes.push( sub.meta.samples );
      }
    }

    const global = fedAvg( accepted, sizes );
    this.engine.setWeights( global );

    const probe = await this.engine.evalAt( this.probeIds, m.model.ctx, PROBE_EVAL_COUNT );
    const prevProbe = this.lastProbeLoss === 0 ? probe : this.lastProbeLoss;
    if ( this.firstProbeLoss === 0 ) this.firstProbeLoss = probe;
    this.lastProbeLoss = probe;
    this.prevGlobal = global;

    const totalSamples = sizes.reduce( ( a, b ) => a + b, 0 );
    for ( const e of entries ) e.share = totalSamples > 0 && e.verdict === 'ok' ? e.samples / totalSamples : 0;
    this.ledger.push( ...entries );

    const globalBuf = encodeWeights( global );
    const fanout = Math.max( 1, peers.length );
    this.transportBytes += globalBuf.byteLength * fanout;
    this.o.transport.broadcast( globalBuf );

    let submitted = 0;
    for ( const s of this.submissions.values() ) submitted += s.bytes;

    const stats: RoundStats = {
      round,
      globalLoss: probe,
      prevGlobalLoss: prevProbe,
      online: peers.length + 1,
      aggregated: accepted.length,
      bytes: globalBuf.byteLength * fanout + submitted,
      elapsedMs: performance.now() - t0,
      entries,
    };
    this.o.transport.broadcast( JSON.stringify( { t: 'round/close', stats } satisfies ControlMessage ) );

    this.o.events.onRound( stats, m.rounds );
    this.o.events.onCurve( { round, local: mine.meta.localLoss, global: probe } );
    this.o.events.onStatus(
      `第 ${ round }/${ m.rounds } 轮完成 · 全局 loss ${ probe.toFixed( 3 ) } · ` +
      `聚合 ${ accepted.length }/${ peers.length + 1 } 节点 · ${ stats.elapsedMs.toFixed( 0 ) }ms`,
    );
  }

  /** 用提交的权重在共识探针上复算，与节点自报值比对。 */
  private async verify ( meta: WeightMeta, weights: NamedWeights ): Promise<{ ok: boolean; note: string }>
  {
    const m = this.manifest!;
    try
    {
      const probeEngine = this.engineFactory( m.model );
      probeEngine.setWeights( weights );
      const recomputed = await probeEngine.evalAt( this.probeIds, m.model.ctx, PROBE_EVAL_COUNT );
      const diff = Math.abs( recomputed - meta.probeLoss );
      const tol = VERIFY_TOL * Math.max( 1, Math.abs( meta.probeLoss ) );
      if ( diff > tol )
      {
        return { ok: false, note: `探针复算 ${ recomputed.toFixed( 4 ) } 与自报 ${ meta.probeLoss.toFixed( 4 ) } 不符（Δ=${ diff.toFixed( 4 ) }），剔除` };
      }
      return { ok: true, note: `探针复算通过（Δ=${ diff.toExponential( 1 ) }）` };
    }
    catch ( err )
    {
      return { ok: false, note: `校验失败：${ ( err as Error ).message }` };
    }
  }

  private entryFor (
    round: number, peerId: string, name: string, deviceKind: string,
    meta: WeightMeta, ok: boolean, note: string,
  ): LedgerEntry
  {
    return {
      round, peerId, name, deviceKind,
      samples: meta.samples, tokens: meta.tokens,
      probeLoss: meta.probeLoss, localLoss: meta.localLoss, deltaNorm: meta.deltaNorm,
      share: 0, verdict: ok ? 'ok' : 'suspect', note,
    };
  }

  private deviceKindOf ( peerId: string ): string
  {
    return this.rosterMap.get( peerId )?.device.kind ?? detectKind();
  }

  // ---------------------------------------------------------------- 节点轮次

  private async peerRound ( round: number ): Promise<void>
  {
    if ( this.running || !this.manifest ) return;
    if ( this.startedAt === 0 ) this.startedAt = Date.now();
    this.running = true;
    try
    {
      const mine = await this.localPhase( round );
      const buf = encodeWeights( mine.weights, mine.meta );
      this.transportBytes += buf.byteLength;
      const target = this.hostId ?? this.o.transport.openPeerIds[ 0 ];
      if ( !target )
      {
        this.o.events.onLog( `第 ${ round } 轮：与主机的通道已断开，跳过上报` );
        return;
      }
      this.o.transport.send( target, buf );
      this.o.events.onStatus( `第 ${ round }/${ this.manifest.rounds } 轮 · 已上报（本地 loss ${ mine.meta.localLoss.toFixed( 3 ) }）` );
      this.o.events.onCurve( { round, local: mine.meta.localLoss, global: this.lastProbeLoss } );
    }
    catch ( err )
    {
      this.o.events.onLog( `第 ${ round } 轮本地训练失败：${ ( err as Error ).message }` );
    }
    finally
    {
      this.running = false;
    }
  }

  // ---------------------------------------------------------------- 收尾

  stop (): void
  {
    this.stopped = true;
    this.o.events.onStatus( '已请求停止（当前轮结束后退出）' );
  }

  private finish (): void
  {
    if ( !this.manifest ) return;
    const card = this.buildCard( this.manifest );
    this.o.events.onDone( card );
    this.o.events.onStatus( `训练完成 · 全局探针 loss ${ card.finalProbeLoss.toFixed( 3 ) }` );
  }

  buildCard ( m: RoomManifest ): ModelCard
  {
    const byPeer = new Map<string, ModelCard[ 'contributors' ][ number ]>();
    const ensure = ( peerId: string, name: string, deviceKind: string ): ModelCard[ 'contributors' ][ number ] =>
    {
      let c = byPeer.get( peerId );
      if ( !c )
      {
        c = { peerId, name, deviceKind, samples: 0, tokens: 0, rounds: 0, ok: 0, suspect: 0, contributedChars: this.credits.get( peerId )?.chars ?? 0 };
        byPeer.set( peerId, c );
      }
      return c;
    };
    for ( const e of this.ledger )
    {
      const c = ensure( e.peerId, e.name, e.deviceKind );
      c.rounds += 1;
      c.samples += e.samples;
      c.tokens += e.tokens;
      if ( e.verdict === 'ok' ) c.ok += 1;
      else if ( e.verdict === 'suspect' ) c.suspect += 1;
    }
    return {
      manifest: m,
      startedAt: this.startedAt,
      finishedAt: Date.now(),
      rounds: m.rounds,
      initialProbeLoss: this.initialProbeLoss,
      firstProbeLoss: this.firstProbeLoss,
      finalProbeLoss: this.lastProbeLoss,
      paramCount: this.engine.paramCount(),
      weightsDigest: weightsDigest( this.engine.getWeights() ),
      transportBytes: this.transportBytes,
      ledger: this.ledger,
      contributors: [ ...byPeer.values() ],
      note: '信任层 v0：清单指纹 + 共识探针复算 + 贡献账本。摘要算法为 FNV-1a（非密码学安全），尚无质押与惩罚机制。',
    };
  }

  /** 当前全局权重的可下载字节。 */
  exportGlobalWeights (): ArrayBuffer | null
  {
    try
    {
      return encodeWeights( this.engine.getWeights() );
    }
    catch
    {
      return null;
    }
  }
}

function sleep ( ms: number ): Promise<void>
{
  return new Promise( ( r ) => window.setTimeout( r, ms ) );
}

/** 根据 UA 粗判设备类别（仅用于展示）。 */
export function detectKind (): string
{
  const ua = navigator.userAgent;
  if ( /Android/i.test( ua ) ) return 'Android';
  if ( /iPhone|iPad|iPod/i.test( ua ) ) return 'iOS';
  if ( /Macintosh/i.test( ua ) ) return 'macOS';
  if ( /Windows/i.test( ua ) ) return 'Windows';
  if ( /Linux/i.test( ua ) ) return 'Linux';
  return '未知设备';
}
