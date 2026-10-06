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
import { detectKind, gpuIfReady } from './capability';
import {
  aggregateGlobal,
  decodeWeights,
  encodeWeights,
  l2Distance,
  specContext,
  weightsDigest,
  type AggregateState,
  type ControlMessage,
  type EngineId,
  type LedgerEntry,
  type NamedWeights,
  type RoomManifest,
  type RoundStats,
  type WeightMeta,
} from './protocol';
import type { PeerInfo, Transport } from './transport';

const PROBE_EVAL_COUNT = 256;
const VERIFY_TOL = 1e-3;
const ROUND_TIMEOUT_MS = 30000;
/**
 * 开训前的「就绪」窗口。房主下发任务后等每个节点回 ready。
 * 手机首次编译 WGSL 不慢（实测 6 轮 6 秒量级），25 秒足够宽松。
 */
const READY_TIMEOUT_MS = 25000;

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
  transport: Transport;
  events: NodeEvents;
  corpus: Corpus;
  /** 用户贡献的文本（可为空）；并入本节点本地训练池 */
  contributionText: string;
  /** 仅 host：本地构建的清单 */
  manifest?: RoomManifest;
  /**
   * 仅房主：引擎策略。
   *   `auto` —— 全网每台设备都拿得到 WebGPU 适配器才用 GPU，只要有一个不行就回退 CPU
   *   `cpu`  —— 一律 CPU（兼容性最好）
   * 真正的抉择发生在开训那一刻，由 negotiateEngine 依据**全网**能力做，
   * 而不是建房时凭房主一己之力拍脑袋。
   */
  enginePolicy?: 'auto' | 'cpu';
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
  private readonly enginePolicy: 'auto' | 'cpu';

  /** 引擎在清单确定后构造：host 在构造期，peer 收到 assign 时。 */
  private engine!: TrainEngine;
  private readonly engineFactory = createEngine;
  private manifest: RoomManifest | null;
  private shardIndex = 0;
  // 显式标注为 Uint32Array：TS 5.7+ 起 TypedArray 带 ArrayBufferLike 泛型，
  // 不标注会被推断成过窄的 Uint32Array<ArrayBuffer>，接不住 encodeTo 的返回值。
  private trainIds: Uint32Array = new Uint32Array( 0 );
  private probeIds: Uint32Array = new Uint32Array( 0 );

  /** 上一轮结束时的全局权重（用于算 Δ 范数、以及聚合的基准点） */
  private prevGlobal: NamedWeights | null = null;
  /** 聚合器状态（DiLoCo 的外层动量）。 */
  private readonly aggState: AggregateState = { momentum: null };
  private readonly submissions = new Map<string, Submission>();
  private readonly ledger: LedgerEntry[] = [];
  private readonly credits = new Map<string, { chars: number; digest: string }>();
  private rosterMap = new Map<string, PeerInfo>();
  private hostId: string | null = null;
  private assignCounter = 0;
  /** 每个节点被分到的数据分片。记住它，这样重发 assign 时用的是同一片。 */
  private readonly assignedShards = new Map<string, number>();

  // ---- 开训前的协调状态（房主侧）----
  /**
   * 房间已开训。之后加入的节点一律拒绝并说明原因 ——
   * 否则会出现「房主已经在跑第 3 轮、手机刚到第 1 轮」这种错位。
   */
  private sealed = false;
  private currentRound = 0;
  /** 已确认「模型建好、随时能训」的节点 */
  private readonly readyPeers = new Set<string>();
  /** 明确回了不可用的节点 → 原因 */
  private readonly notReady = new Map<string, string>();
  /** 训练中反复无响应的节点：只容忍一次，之后不再为它空等 */
  private readonly offlinePeers = new Set<string>();

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
    this.enginePolicy = opts.enginePolicy ?? 'auto';
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
  /** 房间是否已开训（开训后加入的节点会被拒）。 */
  get isSealed (): boolean { return this.sealed; }
  get openedRound (): number { return this.currentRound; }
  /** 已确认可用的节点 id。 */
  get readyIds (): string[] { return [ ...this.readyPeers ]; }
  /** 未就绪的节点 → 原因（界面直接展示，别让它变成黑箱）。 */
  get notReadyReasons (): Map<string, string> { return this.notReady; }

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
    if ( this.probeIds.length < specContext( m.model ) + 32 ) throw new Error( '共识探针数据不足' );

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
          if ( this.manifest )
          {
            // 重发的 assign（房主等就绪时会补发）：模型早就建好了，补回一次 ready 就行
            this.sendReady( true );
            break;
          }
          try
          {
            this.adoptManifest( msg.manifest, msg.shardIndex );
            this.sendReady( true );
            this.o.events.onStatus( '任务已收到、模型已建好，等待房主开轮…' );
          }
          catch ( err )
          {
            // 最常见的失败：房间用 WebGPU 引擎，而本机没有可用的 WebGPU。
            // 权重形状不同，根本没法混训 —— 所以必须把原因回报房主，
            // 而不是自己报个错就算完（房主据此把它排除，而不是每轮白等超时）。
            const detail = ( err as Error ).message;
            this.sendReady( false, detail );
            this.o.events.onStatus( `本机无法参与本次训练：${ detail }` );
            this.o.events.onLog( `收到任务，但本机构造模型失败：${ detail }。已回报房主。` );
          }
          break;
        case 'busy':
          this.o.events.onStatus( `房间已锁定（房主正在第 ${ msg.round } 轮训练），本机未参与` );
          this.o.events.onLog( '加入得太晚了：房主已经锁定房间开始训练。请等这一轮结束后，让房主重新创建房间再加入。' );
          break;
        case 'round/open':
          // 没有清单 = 本机没被纳入本次训练，不要跟着跑
          if ( !this.manifest ) break;
          void this.peerRound( msg.round );
          break;
        case 'round/close':
        {
          const roster = this.manifest?.rounds ?? 0;
          this.lastProbeLoss = msg.stats.globalLoss;
          if ( this.firstProbeLoss === 0 ) this.firstProbeLoss = msg.stats.globalLoss;
          this.ledger.push( ...msg.stats.entries );
          this.o.events.onRound( msg.stats, roster );
          if ( roster > 0 && msg.stats.round >= roster ) void this.finish();
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
      this.o.events.onLog( `${ this.nameOf( msg.peerId ) } 贡献数据 ${ msg.chars } 字符` );
      return;
    }

    if ( msg.t === 'ready' )
    {
      if ( msg.ok )
      {
        if ( !this.readyPeers.has( msg.peerId ) )
        {
          this.readyPeers.add( msg.peerId );
          this.o.events.onLog( `${ this.nameOf( msg.peerId ) } 已就绪（引擎 ${ msg.engine }）` );
        }
      }
      else if ( !this.notReady.has( msg.peerId ) )
      {
        this.notReady.set( msg.peerId, msg.reason ?? '节点自报无法构建模型' );
        this.o.events.onLog( `${ this.nameOf( msg.peerId ) } 自报不可用：${ msg.reason ?? '未说明原因' }` );
      }
      this.o.events.onRoster( [ ...this.rosterMap.values() ], this.selfId );
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
    this.o.events.onLog( `P2P 通道就绪：${ this.nameOf( peerId ) }` );
    // 通道就绪会改变节点表里的「状态」列，必须重绘一次
    // （onRoster 只在成员变化时触发，早于通道打开，否则界面会一直显示"连接中"）
    this.o.events.onRoster( [ ...this.rosterMap.values() ], this.selfId );
    if ( this.o.role === 'peer' )
    {
      this.hostId = peerId;
      this.announceCredit();
      return;
    }
    // 房主侧：
    // 已经开训之后才上线的节点直接拒绝 —— 否则就是「房主跑到第 3 轮、手机还停在第 0 轮」。
    if ( this.sealed )
    {
      this.o.transport.sendControl( peerId, { t: 'busy', round: this.currentRound } satisfies ControlMessage );
      this.o.events.onLog( `${ this.nameOf( peerId ) } 在开训后才加入，已告知本机未参与` );
      return;
    }
    // 开训前上线的节点**先不下发任务**：任务要等房主点「开始训练」、
    // 按全网能力协商出引擎之后，才统一下发。这是「房主调配」的关键。
    this.o.events.onLog( `${ this.nameOf( peerId ) } 已加入，等待房主开始训练` );
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
    let idx = this.assignedShards.get( peerId );
    if ( idx === undefined )
    {
      idx = ( this.assignCounter % span ) + 1;
      this.assignCounter += 1;
      this.assignedShards.set( peerId, idx );
    }
    this.o.transport.sendControl( peerId, { t: 'assign', shardIndex: idx, manifest: m } );
  }

  /** 节点侧：回报「模型已建好 / 建不出来」。房主靠它决定谁能进训练。 */
  private sendReady ( ok: boolean, reason?: string ): void
  {
    const target = this.hostId ?? this.o.transport.openPeerIds[ 0 ];
    if ( !target ) return;
    const msg: ControlMessage = {
      t: 'ready',
      peerId: this.selfId,
      ok,
      engine: this.manifest?.model.engine ?? 'mlp',
      ...( reason ? { reason } : {} ),
    };
    this.o.transport.sendControl( target, msg );
  }

  /** 房间内的普通节点（不含房主自己）。 */
  private hostPeers (): PeerInfo[]
  {
    return [ ...this.rosterMap.values() ].filter( ( p ) => p.role === 'peer' );
  }

  private nameOf ( peerId: string ): string
  {
    return this.rosterMap.get( peerId )?.name ?? peerId.slice( 0, 6 );
  }

  // ---------------------------------------------------------------- 本地训练

  /**
   * 探针评估的样本量。GPU 引擎每批都要回读 logits，代价比 CPU 高，
   * 所以用量少一些 —— 它只用于「展示全局 loss 走势」，不需要很高精度。
   */
  private probeCount (): number
  {
    return this.engine.spec.engine === 'gpu-tinygpt' ? 64 : PROBE_EVAL_COUNT;
  }

  private async evalProbe (): Promise<number>
  {
    const m = this.manifest;
    if ( !m ) throw new Error( 'evalProbe: 清单尚未确定' );
    return this.engine.evalAt( this.probeIds, specContext( m.model ), this.probeCount() );
  }

  private async localPhase ( round: number ): Promise<{ weights: NamedWeights; meta: WeightMeta; base: NamedWeights }>
  {
    const m = this.manifest;
    if ( !m ) throw new Error( 'localPhase: 清单尚未确定' );
    // base = 本轮开始时的共同出发点。主机是上一轮广播的全局权重；
    // 还没广播过（第一轮或单节点）时，就是本机当前权重。聚合要用它做基准。
    const base = this.prevGlobal ?? await this.engine.getWeights();

    let loss = 0;
    let tokens = 0;
    for ( let s = 0; s < m.localSteps; s++ )
    {
      const r = await this.engine.trainBatch( this.trainIds, m.batchSize, m.lr );
      loss += r.loss;
      tokens += r.tokens;
    }
    loss /= m.localSteps;

    const weights = await this.engine.getWeights();
    // 关掉交叉校验时不跑探针 —— 省掉每个节点每轮的一次额外评估（手机尤其有感）
    const probeLoss = m.crossCheck ? await this.evalProbe() : 0;
    const meta: WeightMeta = {
      round,
      peerId: this.selfId,
      samples: m.localSteps * m.batchSize,
      tokens,
      localLoss: loss,
      probeLoss,
      deltaNorm: l2Distance( weights, base ),
      digest: weightsDigest( weights ),
    };
    return { weights, meta, base };
  }

  // ---------------------------------------------------------------- 主机主循环

  /**
   * 房主开训 —— 整个「调配」过程都收敛在这里：
   *
   *   1. 按**全网**能力协商引擎（不是凭房主一己之力拍脑袋）
   *   2. 用协商结果重建清单并下发（参数全部由房主定义）
   *   3. 等每个节点回 ready（没回话的会补发 assign），只有确认能训的才纳入
   *   4. 锁定房间，然后逐轮训练
   *
   * 为什么必须有第 3 步：没有它，房主只知道「对方连着」，不知道「对方训得动」，
   * 于是只能靠每轮 30–60 秒超时去发现 —— 那正是「两边不同步 + 主机报超时」的成因。
   *
   * buildFinalManifest 由界面提供：参数与语料在界面手里，
   * 编排层只负责「算出该用哪个引擎、以及为什么」。
   */
  async startHost (
    buildFinalManifest: ( engine: EngineId, reason: string ) => RoomManifest,
  ): Promise<void>
  {
    if ( this.o.role !== 'host' ) throw new Error( 'startHost: 仅房主可用' );
    if ( this.running || this.sealed ) return;
    this.running = true;
    this.stopped = false;
    this.startedAt = Date.now();

    const peers = this.hostPeers();

    // ---- 1 + 2. 协商引擎并按结果重建清单 ----
    const { engine, reason } = this.negotiateEngine( peers );
    const m = buildFinalManifest( engine, reason );
    this.adoptManifest( m, 0 );
    this.o.events.onLog( `引擎协商结果：${ engine } —— ${ reason }` );

    // ---- 3. 下发 + 等就绪 ----
    this.o.events.onStatus( `房主已下发任务（${ engine }），等待 ${ peers.length } 个节点就绪…` );
    for ( const p of peers ) this.sendAssign( p.peerId );
    await this.waitReady( peers );

    const ready = peers.filter( ( p ) => this.readyPeers.has( p.peerId ) );
    const skipped = peers.filter( ( p ) => !this.readyPeers.has( p.peerId ) );
    for ( const p of skipped )
    {
      this.o.events.onLog(
        `${ p.name }（${ p.device.kind }）未纳入本次训练：${ this.notReady.get( p.peerId ) ?? '未就绪' }`,
      );
    }

    // 房间里明明有节点、却一个都没就绪 —— 直接中止。
    // 让房主自己跑完会制造「看起来在联训、其实手机全程没参与」的假象，
    // 那种错位比明确失败更糟。
    if ( ready.length === 0 && peers.length > 0 )
    {
      this.running = false;
      this.o.events.onStatus( `已中止：${ peers.length } 个节点全部未能就绪` );
      this.o.events.onLog(
        '排查顺序：① 节点房间 ID 是否与房主一致；② 节点页面的状态栏/日志有没有报错；' +
        '③ 节点是否点了「加入房间」。',
      );
      return;
    }

    // ---- 4. 锁定房间 ----
    this.sealed = true;
    this.o.events.onRoster( [ ...this.rosterMap.values() ], this.selfId );
    this.o.events.onStatus(
      `房间已锁定：房主 + ${ ready.length } 个节点参与，共 ${ m.rounds } 轮` +
      ( skipped.length > 0 ? ` · ${ skipped.length } 个节点未纳入` : '' ),
    );

    this.initialProbeLoss = await this.evalProbe();
    this.o.events.onLog( `初始模型探针 loss = ${ this.initialProbeLoss.toFixed( 3 ) }（未训练的基线）` );
    if ( m.aggregate.mode === 'diloco' ) this.aggState.momentum = null;

    try
    {
      for ( let r = 1; r <= m.rounds; r++ )
      {
        if ( this.stopped ) break;
        await this.hostRound( r );
      }
      if ( !this.stopped ) await this.finish();
    }
    catch ( err )
    {
      this.o.events.onStatus( `房主循环中断：${ ( err as Error ).message }` );
    }
    finally
    {
      this.running = false;
    }
  }

  /**
   * 引擎协商：取**能力下限**。
   *
   * 规则很简单 —— 只要房间里有一台设备跑不了 WebGPU，全网就用 CPU。
   * 因为引擎必须全网一致（权重形状不同没法聚合），而 CPU 引擎在任何设备上都能跑，
   * 所以「回退 CPU」永远安全；反过来则会让那台设备彻底出局。
   *
   * 判据用 `gpuOk`（真拿到适配器）而不是 `webgpu`（navigator.gpu 存在）：
   * 后者在下述情况会骗人 —— 有接口但拿不到适配器（驱动旧 / chrome://gpu 里被禁 /
   * 远程桌面里没有 GPU）。宁可保守。
   */
  private negotiateEngine ( peers: PeerInfo[] ): { engine: EngineId; reason: string }
  {
    if ( this.enginePolicy === 'cpu' )
    {
      return { engine: 'mlp', reason: '房主指定「只用 CPU」' };
    }
    if ( gpuIfReady() === null )
    {
      return { engine: 'mlp', reason: `房主本机（${ detectKind() }）没有可用的 WebGPU 适配器` };
    }
    const noGpu = peers.filter( ( p ) => p.device.gpuOk !== true );
    if ( noGpu.length > 0 )
    {
      const who = noGpu.map( ( p ) => `${ p.name }（${ p.device.kind }）` ).join( '、' );
      return { engine: 'mlp', reason: `${ who } 没有可用的 WebGPU 适配器 → 按能力下限全网回退 CPU` };
    }
    return { engine: 'gpu-tinygpt', reason: '全网所有设备都能用 WebGPU' };
  }

  /** 等节点就绪；没回话的每 5 秒补发一次 assign（覆盖页面刷新/首帧丢失）。 */
  private async waitReady ( peers: PeerInfo[] ): Promise<void>
  {
    if ( peers.length === 0 ) return;
    const settled = (): number =>
      peers.filter( ( p ) => this.readyPeers.has( p.peerId ) || this.notReady.has( p.peerId ) ).length;

    const deadline = performance.now() + READY_TIMEOUT_MS;
    let nextNudge = performance.now() + 5000;
    while ( performance.now() < deadline && !this.stopped && settled() < peers.length )
    {
      if ( performance.now() >= nextNudge )
      {
        for ( const p of peers )
        {
          if ( !this.readyPeers.has( p.peerId ) && !this.notReady.has( p.peerId ) ) this.sendAssign( p.peerId );
        }
        nextNudge = performance.now() + 5000;
        this.o.events.onStatus( `等待节点就绪…（${ this.readyPeers.size }/${ peers.length }）` );
      }
      await sleep( 100 );
    }

    for ( const p of peers )
    {
      if ( !this.readyPeers.has( p.peerId ) && !this.notReady.has( p.peerId ) )
      {
        this.notReady.set( p.peerId, `就绪超时（${ READY_TIMEOUT_MS / 1000 } 秒内未回应）` );
      }
    }
  }

  private async hostRound ( round: number ): Promise<void>
  {
    const m = this.manifest!;
    const t0 = performance.now();
    this.currentRound = round;
    this.submissions.clear();

    // 只跟「开训前已确认就绪」且「训练中没掉线」的节点打交道。
    // 未就绪的节点在开训前已经说明过原因了，不能再让它们每轮拖一次 30 秒超时 ——
    // 那正是「房主显示手机超时」的来源。
    const peers = this.hostPeers().filter(
      ( p ) => this.readyPeers.has( p.peerId ) && !this.offlinePeers.has( p.peerId ),
    );

    this.o.transport.broadcast( JSON.stringify( { t: 'round/open', round } satisfies ControlMessage ) );

    const mine = await this.localPhase( round );
    this.o.events.onStatus( `第 ${ round }/${ m.rounds } 轮 · 本机完成（loss ${ mine.meta.localLoss.toFixed( 3 ) }），等待节点上报…` );

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
        // 容忍一次（网络抖动），但之后不再为它空等 —— 否则每轮都要白等 30 秒
        this.offlinePeers.add( p.peerId );
        entries.push( {
          round, peerId: p.peerId, name: p.name, deviceKind: p.device.kind,
          samples: 0, tokens: 0, probeLoss: 0, localLoss: 0, deltaNorm: 0, share: 0,
          verdict: 'timeout',
          note: '本轮未上报，已标为离线；后续轮次不再等待，避免每轮空等超时',
        } );
        this.o.events.onLog( `${ p.name } 本轮未上报 → 标为离线，后续轮次不再等它` );
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

    // 开训前就没就绪的节点：只在第 1 轮记一条，把原因留在账本里，
    // 这样模型卡能解释「为什么这次只有两台机器在训」。
    if ( round === 1 )
    {
      for ( const p of this.hostPeers() )
      {
        if ( this.readyPeers.has( p.peerId ) ) continue;
        entries.push( {
          round, peerId: p.peerId, name: p.name, deviceKind: p.device.kind,
          samples: 0, tokens: 0, probeLoss: 0, localLoss: 0, deltaNorm: 0, share: 0,
          verdict: 'timeout',
          note: `未纳入本次训练：${ this.notReady.get( p.peerId ) ?? '未就绪' }`,
        } );
      }
    }

    // ---- 聚合：两种模式共用同一个「基准点 + 加权增量」骨架（数学在 protocol.ts） ----
    const global = aggregateGlobal( mine.base, accepted, sizes, m.aggregate, this.aggState );
    this.engine.setWeights( global );

    const probe = await this.evalProbe();
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
    if ( !m.crossCheck ) return { ok: true, note: '未开启交叉校验（自用设备，省掉一轮评估）' };
    try
    {
      const probeEngine = this.engineFactory( m.model );
      probeEngine.setWeights( weights );
      const recomputed = await probeEngine.evalAt( this.probeIds, specContext( m.model ), this.probeCount() );
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

  private async finish (): Promise<void>
  {
    if ( !this.manifest ) return;
    const card = await this.buildCard( this.manifest );
    this.o.events.onDone( card );
    this.o.events.onStatus( `训练完成 · 全局探针 loss ${ card.finalProbeLoss.toFixed( 3 ) }` );
  }

  async buildCard ( m: RoomManifest ): Promise<ModelCard>
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
      weightsDigest: weightsDigest( await this.engine.getWeights() ),
      transportBytes: this.transportBytes,
      ledger: this.ledger,
      contributors: [ ...byPeer.values() ],
      note: '清单指纹 + 贡献账本 + 可复现（固定种子 + 确定性评估）。摘要算法为 FNV-1a（非密码学安全）。',
    };
  }

  /** 当前全局权重的可下载字节。 */
  async exportGlobalWeights (): Promise<ArrayBuffer | null>
  {
    try
    {
      return encodeWeights( await this.engine.getWeights() );
    }
    catch
    {
      return null;
    }
  }
}

function sleep ( ms: number ): Promise<void>
{
  // 用 globalThis 而非 window：同一份编排代码要能在浏览器与 Node（无头端到端测试）里跑
  return new Promise( ( r ) => globalThis.setTimeout( r, ms ) );
}
