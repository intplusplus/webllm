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
/**
 * 单轮等提交的窗口。同一 WiFi 直连很快，12 秒足够。
 * 关键是不为「已经不在的节点」空等 —— 链路一断立刻从等待集移除。
 */
const ROUND_TIMEOUT_MS = 12000;
/**
 * 开训前给节点一点就绪时间，让「两边都点完了再开始」这条常见路径能一起起步。
 * 但**绝不阻塞**：窗口过了就先开跑，晚到的节点从下一轮开始参与。
 */
const READY_WINDOW_MS = 10000;
/** 连续漏报这么多轮后不再为它等待（重新就绪会自动恢复）。 */
const MAX_MISSES = 3;
/**
 * 单轮本地训练的时间预算（毫秒）。
 *
 * 为什么必须有它：`trainBatch` 是一整段**同步**计算，一步算多久主线程就卡多久。
 * PC 上 CPU 引擎「中」档一步 ≈ 67ms（实测），手机慢 3–10 倍 → 300–700ms，
 * 一轮 20 步就是 6–14 秒页面完全无响应，浏览器直接判「页面无响应」。
 * 有了预算，慢设备少跑几步、**如实上报样本数**（FedAvg 按实际样本数加权），
 * 比把界面卡死好得多。
 */
const ROUND_BUDGET_MS = 2500;

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

  // ---- 动态成员（房主侧）----
  /**
   * 已开训到第几轮（0 = 还没开始）。
   * 注意：**它不是一道锁** —— 晚到的节点照样能加入，只是从下一轮开始参与。
   */
  private currentRound = 0;
  /**
   * 房主手上的「当前全局权重」。
   * 晚到的节点必须拿它当训练起点 —— 否则它会拿自己那份随机初始权重当基准，
   * 算出来的增量是错的，聚合进去直接把全局模型带偏。这是「随时加入」能成立的
   * 唯一前提，漏了它就是隐性污染。
   */
  private currentGlobal: NamedWeights | null = null;
  /** 已确认「模型建好、随时能训」的节点 */
  private readonly readyPeers = new Set<string>();
  /** 明确回了不可用的节点 → 原因 */
  private readonly notReady = new Map<string, string>();
  /** 链路已断 / 连续漏报的节点：暂时不为它等待，重新就绪即恢复 */
  private readonly offlinePeers = new Set<string>();
  /** 连续漏报轮数。漏报是常态（网络抖动），所以只累计、不永久拉黑。 */
  private readonly misses = new Map<string, number>();

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
  /** 房主侧：训练是否已经开跑（晚到的节点会从下一轮加入，不会被拒）。 */
  get hasStarted (): boolean { return this.currentRound > 0; }
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
          // 同一份清单的重发（房主等就绪时会补发）：补回一次 ready 就行。
          // 用清单指纹判断 —— 重开一场训练时清单可能换规格（hidden/seed 等），
          // 拿旧引擎假装就绪会直接把聚合炸掉，必须重建。
          if ( this.manifest && this.manifest.fingerprint === msg.manifest.fingerprint )
          {
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
        case 'sync':
          // 房主把「当前全局权重」推给中途加入的本机（紧随其后就是一帧二进制权重）
          if ( msg.finished )
          {
            this.o.events.onStatus( `训练已结束（共 ${ msg.round } 轮）· 你加入晚了，这一轮没赶上` );
            this.o.events.onLog( '房主已把最终全局权重同步给本机。训练已经收尾，等房主重新开始就能参与。' );
          }
          else
          {
            this.o.events.onStatus( `已对齐到第 ${ msg.round } 轮 · 从下一轮开始参与训练` );
            this.o.events.onLog( `房主正在同步第 ${ msg.round } 轮的全局权重，本机从下一轮开始参与` );
          }
          break;
        case 'round/open':
          // 还没有清单 = 本机还没被房主纳入，别跟着跑（等 assign 来了再说）
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
        const isNew = !this.readyPeers.has( msg.peerId );
        this.readyPeers.add( msg.peerId );
        this.notReady.delete( msg.peerId );
        this.offlinePeers.delete( msg.peerId ); // 重新就绪 = 恢复参与
        this.misses.set( msg.peerId, 0 );
        if ( isNew )
        {
          this.o.events.onLog( `${ this.nameOf( msg.peerId ) } 已就绪（引擎 ${ msg.engine }）` );
          // 已经开训之后才加入的节点，必须先拿到「当前全局权重」才能算对增量
          this.pushCurrentGlobal( msg.peerId );
        }
      }
      else if ( !this.notReady.has( msg.peerId ) )
      {
        this.notReady.set( msg.peerId, msg.reason ?? '节点自报无法构建模型' );
        this.readyPeers.delete( msg.peerId );
        this.o.events.onLog( `${ this.nameOf( msg.peerId ) } 自报不可用：${ msg.reason ?? '未说明原因' }` );
        if ( /WebGPU|引擎|构造/.test( msg.reason ?? '' ) )
        {
          this.o.events.onLog(
            '提示：引擎写进清单后全网不可变。若要让这类设备参与，请点「停止」，' +
            '然后以「只用 CPU」重新创建房间再开始训练。',
          );
        }
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
    // 房主侧：谁都可以随时来，**没有任何时间门槛**。
    if ( this.currentRound > 0 )
    {
      // 训练已经在跑（或已经跑过）：立刻下发任务。
      // 它就绪之后会收到当前全局权重，从下一轮开始参与 —— 不需要房主做任何操作。
      this.o.events.onLog( `${ this.nameOf( peerId ) } 在训练进行中加入（当前第 ${ this.currentRound } 轮），已下发任务` );
      this.sendAssign( peerId );
      return;
    }
    this.o.events.onLog( `${ this.nameOf( peerId ) } 已加入，等待房主开始训练` );
  }

  onPeerClose ( peerId: string ): void
  {
    this.submissions.delete( peerId );
    if ( this.o.role !== 'host' ) return;
    // 立刻从等待集移除 —— 否则每轮都要为一个已经不在的节点空等超时
    this.offlinePeers.add( peerId );
    this.o.events.onLog( `${ this.nameOf( peerId ) } 已断开，本轮不再等它（随时重连即可继续参与）` );
    this.o.events.onRoster( [ ...this.rosterMap.values() ], this.selfId );
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

  /**
   * 把「当前全局权重」推给一个中途加入的节点。
   *
   * 这是「随时加入、不影响训练」能成立的**唯一前提**：新节点必须从同一个点继续，
   * 否则它算出的 Δ 是相对自己那份随机初始权重的，聚合进去会直接把全局模型带偏
   * —— 而且这是隐性污染，账本上看不出来。
   */
  private pushCurrentGlobal ( peerId: string ): void
  {
    if ( this.currentRound === 0 || !this.currentGlobal ) return;
    // 训练已经收尾才加入的节点，必须明确告诉它「你来晚了」，不能让它干等永远不会来的下一轮
    const finished = !this.running;
    this.o.transport.sendControl( peerId, {
      t: 'sync', round: this.currentRound, ...( finished ? { finished: true } : {} ),
    } satisfies ControlMessage );
    const buf = encodeWeights( this.currentGlobal );
    this.transportBytes += buf.byteLength;
    this.o.transport.send( peerId, buf );
    this.o.events.onLog(
      finished
        ? `${ this.nameOf( peerId ) } 是训练结束后才加入的，已把最终全局权重给它（并告知已结束）`
        : `已把第 ${ this.currentRound } 轮的全局权重同步给 ${ this.nameOf( peerId ) }`,
    );
  }

  /**
   * 本轮的等待集：已就绪、没掉线、且不是连续漏报过多。
   *
   * 刻意**不**把「最近漏报过的节点」永久拉黑 —— 漏报是常态（WiFi 抖动、手机切后台），
   * 记满 MAX_MISSES 才暂时不等；任何一次 ready 都会把它清零恢复。
   */
  private waitSet (): PeerInfo[]
  {
    return this.hostPeers().filter( ( p ) =>
      this.readyPeers.has( p.peerId )
      && !this.offlinePeers.has( p.peerId )
      && ( this.misses.get( p.peerId ) ?? 0 ) < MAX_MISSES );
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
    let steps = 0;
    const roundStart = performance.now();

    for ( let s = 0; s < m.localSteps; s++ )
    {
      const r = await this.engine.trainBatch( this.trainIds, m.batchSize, m.lr );
      loss += r.loss;
      tokens += r.tokens;
      steps += 1;

      // **必须主动让出主线程**。trainBatch 是一整段同步计算 —— 纯 JS 引擎一步
      // 可能要几十到几百毫秒，GPU 引擎也有回读等待。不让出的话，页面在这段时间里
      // 完全无响应，用户看到的就是「点开始之后卡死了」。
      await sleep( 0 );

      // 时间预算：慢设备（手机）跑不动房主定义的步数时，少跑几步、如实上报样本数。
      // 这不是偷工减料 —— FedAvg 按实际样本数加权，少跑的节点权重自然小；
      // 比「把主线程卡死几分钟」好得多。
      if ( performance.now() - roundStart >= ROUND_BUDGET_MS ) break;
    }
    loss /= steps;

    const weights = await this.engine.getWeights();
    // 关掉交叉校验时不跑探针 —— 省掉每个节点每轮的一次额外评估（手机尤其有感）
    const probeLoss = m.crossCheck ? await this.evalProbe() : 0;
    const meta: WeightMeta = {
      round,
      peerId: this.selfId,
      samples: steps * m.batchSize,
      tokens,
      localLoss: loss,
      probeLoss,
      deltaNorm: l2Distance( weights, base ),
      digest: weightsDigest( weights ),
    };

    if ( steps < m.localSteps )
    {
      this.o.events.onLog(
        `本机本轮跑了 ${ steps }/${ m.localSteps } 步（耗时 ${ ( performance.now() - roundStart ).toFixed( 0 ) } ms，` +
        `预算 ${ ROUND_BUDGET_MS } ms）—— 设备较慢，为避免卡死界面而截断；样本数已如实上报`,
      );
    }

    return { weights, meta, base };
  }

  // ---------------------------------------------------------------- 主机主循环

  /**
   * 房主开训。就四步，且**没有任何时间门槛**：
   *
   *   1. 按当前在线设备的能力协商引擎（取能力下限）
   *   2. 按协商结果重建清单并下发（参数全部由房主定义）
   *   3. 给一小段就绪窗口，让「两边都准备好了」这条常见路径能一起起步 —— 但到点就开跑
   *   4. 逐轮训练；中途谁进来就从下一轮参与，谁走了就当轮不等它
   *
   * 与上一版的区别：这里**不再锁房间**。锁会把「晚点加入」变成「被拒绝」，
   * 而那正是用户最想要的用法（第二个手机、刷新页面、中途开一台）。正确做法不是拒绝，
   * 而是让新成员对齐到当前全局权重（pushCurrentGlobal）。
   *
   * buildFinalManifest 由界面提供：参数与语料在界面手里，
   * 编排层只负责「算出该用哪个引擎、以及为什么」。
   */
  async startHost (
    buildFinalManifest: ( engine: EngineId, reason: string ) => RoomManifest,
  ): Promise<void>
  {
    if ( this.o.role !== 'host' ) throw new Error( 'startHost: 仅房主可用' );
    if ( this.running )
    {
      this.o.events.onLog( '训练已在进行中 —— 无需重复开始' );
      return;
    }
    // 上一场已跑完（currentRound>0 且不在跑）：重开一场，而不是静默忽略。
    // 曾经这里直接 return，用户视角就是「点开始没反应 / 训练完不能再训」。
    if ( this.currentRound > 0 ) this.resetSession();
    this.running = true;
    this.stopped = false;
    this.startedAt = Date.now();

    // ---- 1 + 2. 协商引擎，并按结果重建清单 ----
    const { engine, reason } = this.negotiateEngine( this.hostPeers() );
    const m = buildFinalManifest( engine, reason );
    this.adoptManifest( m, 0 );
    this.o.events.onLog( `引擎协商结果：${ engine } —— ${ reason }` );
    // 告诉信令服务器「这个房间已经开始训练了」，别的设备在列表里能看到引擎与状态
    this.o.transport.announce?.( { taskName: m.taskName, engine, started: true } );

    // ---- 3. 下发任务 + 一小段就绪窗口（不阻塞） ----
    for ( const p of this.hostPeers() ) this.sendAssign( p.peerId );
    this.o.events.onStatus( `任务已下发（引擎 ${ engine }）· 等待节点就绪，最多 ${ READY_WINDOW_MS / 1000 } 秒` );
    await this.readyWindow();
    this.o.events.onStatus(
      `开始训练：房主 + ${ this.waitSet().length } 个节点参与，共 ${ m.rounds } 轮` +
      ( this.notReady.size > 0 ? ` · ${ this.notReady.size } 个节点暂不可用` : '' ),
    );

    try
    {
      // 探针评估也属于本场训练：失败要走统一的 catch（之前在 try 外，异常会静默）
      this.initialProbeLoss = await this.evalProbe();
      this.o.events.onLog( `初始模型探针 loss = ${ this.initialProbeLoss.toFixed( 3 ) }（未训练的基线）` );
      if ( m.aggregate.mode === 'diloco' ) this.aggState.momentum = null;

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
   * 规则：
   *   - 房间里有**一台已知设备**跑不了 WebGPU → 全网用 CPU
   *   - **开训时房间里还没有其它设备 → 也用 CPU**
   *
   * 第二条是修出来的教训：房主先点「开始训练」、手机后加入，是「随时加入」承诺下
   * 最常见的路径。若那一刻房间里只有房主自己（PC 有 WebGPU）就把引擎定成 GPU，
   * 之后加入的手机会因构造不出引擎被挡在门外 —— 「随时加入」形同虚设。
   * 引擎一旦写进清单就不可更改（权重形状不兼容），所以**开训那一刻还看不见的设备
   * 也必须被考虑到**，宁可保守。
   *
   * 判据用 `gpuOk`（真拿到适配器）而不是 `webgpu`（navigator.gpu 存在）：
   * 后者会骗人 —— 有接口但拿不到适配器（驱动旧 / chrome://gpu 被禁 / 远程桌面无 GPU）。
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
    if ( peers.length === 0 )
    {
      return {
        engine: 'mlp',
        reason: '开训时房间里还没有其它设备 —— 为让手机等设备之后能随时加入，保守选用 CPU',
      };
    }
    const noGpu = peers.filter( ( p ) => p.device.gpuOk !== true );
    if ( noGpu.length > 0 )
    {
      const who = noGpu.map( ( p ) => `${ p.name }（${ p.device.kind }）` ).join( '、' );
      return { engine: 'mlp', reason: `${ who } 没有可用的 WebGPU 适配器 → 按能力下限全网回退 CPU` };
    }
    return {
      engine: 'gpu-tinygpt',
      reason: `当前 ${ peers.length } 台设备都能用 WebGPU（之后加入的设备也必须支持，否则无法参与）`,
    };
  }

  /**
   * 开训前的就绪窗口。
   *
   * 给「两边都点完了再开始」这条常见路径一个一起起步的机会；到点就开跑，
   * **不等齐** —— 晚到的节点从下一轮加入即可。
   * 没回话的每 3 秒补发一次 assign（覆盖页面刷新、首帧丢失）。
   */
  private async readyWindow (): Promise<void>
  {
    const deadline = performance.now() + READY_WINDOW_MS;
    let nextNudge = performance.now() + 3000;
    while ( performance.now() < deadline && !this.stopped )
    {
      const pending = this.hostPeers().filter(
        ( p ) => !this.readyPeers.has( p.peerId ) && !this.notReady.has( p.peerId ),
      );
      if ( pending.length === 0 ) break;
      if ( performance.now() >= nextNudge )
      {
        for ( const p of pending ) this.sendAssign( p.peerId );
        nextNudge = performance.now() + 3000;
      }
      await sleep( 100 );
    }

    for ( const p of this.hostPeers() )
    {
      if ( !this.readyPeers.has( p.peerId ) && !this.notReady.has( p.peerId ) )
      {
        this.o.events.onLog( `${ this.nameOf( p.peerId ) } 尚未就绪，先开跑 —— 它随时可以加入并参与后续轮次` );
      }
    }
  }

  private async hostRound ( round: number ): Promise<void>
  {
    const m = this.manifest!;
    const t0 = performance.now();
    this.currentRound = round;
    this.submissions.clear();

    // 本轮的等待集：已就绪、没掉线、且不是连续漏报过多（见 waitSet）。
    // 注意这是**每轮重新算**的 —— 所以中途加入的人下一轮就进得来。
    const peers = this.waitSet();

    this.o.transport.broadcast( JSON.stringify( { t: 'round/open', round } satisfies ControlMessage ) );

    const mine = await this.localPhase( round );
    this.o.events.onStatus( `第 ${ round }/${ m.rounds } 轮 · 本机完成（loss ${ mine.meta.localLoss.toFixed( 3 ) }），等待节点上报…` );

    // 等齐就走；但只要「还没交的节点」变空（比如中途断连被移出等待集）或到点，
    // 立刻收尾 —— 绝不为已经不在的人空等。
    const pending = (): PeerInfo[] =>
      peers.filter( ( p ) => !this.submissions.has( p.peerId ) && !this.offlinePeers.has( p.peerId ) );
    const deadline = performance.now() + ROUND_TIMEOUT_MS;
    while ( pending().length > 0 && performance.now() < deadline && !this.stopped )
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
        // 漏报是常态（WiFi 抖动、手机切后台），所以只累计次数，不永久拉黑：
        // 记满 MAX_MISSES 才暂时不等，任何一次 ready 都会清零恢复。
        const n = ( this.misses.get( p.peerId ) ?? 0 ) + 1;
        this.misses.set( p.peerId, n );
        entries.push( {
          round, peerId: p.peerId, name: p.name, deviceKind: p.device.kind,
          samples: 0, tokens: 0, probeLoss: 0, localLoss: 0, deltaNorm: 0, share: 0,
          verdict: 'timeout',
          note: `本轮未上报（连续 ${ n } 次）` +
            ( n >= MAX_MISSES ? '，已暂时不再为它等待（重新就绪即恢复）' : '，下一轮仍会等它'),
        } );
        this.o.events.onLog( `${ p.name } 本轮未上报（连续 ${ n } 次）` );
        continue;
      }
      this.misses.set( p.peerId, 0 ); // 有回应就清零
      this.offlinePeers.delete( p.peerId );
      const v = await this.verify( sub.meta, sub.weights );
      entries.push( this.entryFor( round, p.peerId, sub.name, sub.deviceKind, sub.meta, v.ok, v.note ) );
      if ( v.ok )
      {
        accepted.push( sub.weights );
        sizes.push( sub.meta.samples );
      }
    }

    // 还没就绪的节点：只在第 1 轮记一条，把原因留在账本里，
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
          note: `第 1 轮时尚未就绪：${ this.notReady.get( p.peerId ) ?? '还在准备' }（随时可加入）`,
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
    // 记下当前全局权重：中途加入的节点靠它对齐起点（见 pushCurrentGlobal）
    this.currentGlobal = global;

    const totalSamples = sizes.reduce( ( a, b ) => a + b, 0 );
    for ( const e of entries ) e.share = totalSamples > 0 && e.verdict === 'ok' ? e.samples / totalSamples : 0;
    this.ledger.push( ...entries );

    const globalBuf = encodeWeights( global );
    const fanout = Math.max( 1, peers.length );
    this.transportBytes += globalBuf.byteLength * fanout;
    // 大帧走背压广播并 **await**：保证权重帧先于紧随其后的 round/close
    // （异步分块下也不能乱序）。同步连发会撑爆 RTCDataChannel 发送队列，
    // 2026-10-06 联调实测把房主循环炸断过（房主循环中断：send queue is full）。
    await this.o.transport.broadcastBinary( globalBuf );

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

  /**
   * 重开一场训练前清掉上一场的轮次状态。
   *
   * 保留：名单（rosterMap）、语料贡献（credits）、分片分配（assignedShards）——
   * 这些与「哪一场」无关。清掉：轮次、账本、探针基线、聚合状态、就绪表 ——
   * 就绪表必须清：新清单可能换规格，节点要按新清单重新回报 ready。
   */
  private resetSession (): void
  {
    this.currentRound = 0;
    this.prevGlobal = null;
    this.currentGlobal = null;
    this.aggState.momentum = null;
    this.initialProbeLoss = 0;
    this.firstProbeLoss = 0;
    this.lastProbeLoss = 0;
    this.transportBytes = 0;
    this.ledger.length = 0;
    this.submissions.clear();
    this.misses.clear();
    this.offlinePeers.clear();
    this.readyPeers.clear();
    this.notReady.clear();
    this.stopped = false;
    this.o.events.onLog( '重开一场训练：上一场状态已清空' );
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
