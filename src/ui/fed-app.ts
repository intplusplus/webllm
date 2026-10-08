/**
 * 公共训练网络 · 联邦联训 Demo 界面。
 *
 * 一个页面同时承载两个角色：
 *   - 房间创建者 = 汇聚节点（host，写清单、开轮、校验、FedAvg）
 *   - 加入者 = 普通节点（peer，本地训练、上报、接收全局权重）
 *
 * 界面顺序刻意跟「一轮训练的心智模型」一致：
 *   连接 → 任务清单 → 我的数据 → 节点表 → 训练与曲线 → 账本 → 成果
 */
import './fed.css';
import { corpusDigest, loadBuiltinCorpus, pickProbe, type Corpus } from '../fed/corpus';
import {
  fingerprintOf,
  MODEL_PRESETS,
  specLabel,
  type AggregateSpec,
  type ControlMessage,
  type DevCap,
  type LedgerEntry,
  type ModelPreset,
  type ModelSpec,
  type RoomManifest,
  type RoundStats,
} from '../fed/protocol';
import { RoomTransport, type PeerInfo, type Transport } from '../fed/transport';
import { LocalBus } from '../fed/bus';
import {
  detectKind,
  probeCapability,
  type Capability,
} from '../fed/capability';
import { FedNode, type ModelCard, type NodeEvents } from '../fed/node';

// ------------------------------------------------------------------ 小工具

function el<K extends keyof HTMLElementTagNameMap> (
  tag: K, cls?: string, text?: string,
): HTMLElementTagNameMap[ K ]
{
  const n = document.createElement( tag );
  if ( cls ) n.className = cls;
  if ( text !== undefined ) n.textContent = text;
  return n;
}

function randomId (): string
{
  const b = crypto.getRandomValues( new Uint8Array( 6 ) );
  return [ ...b ].map( ( x ) => x.toString( 16 ).padStart( 2, '0' ) ).join( '' );
}

function fmtBytes ( n: number ): string
{
  if ( n < 1024 ) return `${ n } B`;
  if ( n < 1024 * 1024 ) return `${ ( n / 1024 ).toFixed( 1 ) } KB`;
  return `${ ( n / 1024 / 1024 ).toFixed( 2 ) } MB`;
}

function shardOf ( corpus: Corpus, manifest: RoomManifest, index: number ): string
{
  const per = Math.floor( corpus.text.length / manifest.shards );
  const start = index * per;
  const end = index === manifest.shards - 1 ? corpus.text.length : start + per;
  return `${ start.toLocaleString() } – ${ end.toLocaleString() }`;
}

/**
 * 本机能力快照 —— 它会被上传到房间，**房主据此协商全网引擎**。
 * 所以 `gpuOk` 必须如实：有 `navigator.gpu` 不代表拿得到适配器，
 * 而房主只会认 `gpuOk`。拿不准就当 false（最坏结果是全网跑 CPU，不会出错）。
 */
function detectDevice ( cap: Capability | null ): DevCap
{
  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    kind: detectKind(),
    webgpu: typeof navigator.gpu !== 'undefined',
    gpuOk: cap?.adapterOk ?? false,
    secureContext: cap?.secureContext ?? window.isSecureContext,
    cores: nav.hardwareConcurrency ?? 0,
    memoryGB: nav.deviceMemory ?? 0,
    ua: navigator.userAgent,
  };
}

function defaultSignalUrl (): string
{
  const q = qp( 'signal' );
  if ( q ) return q;
  const host = location.hostname || '127.0.0.1';
  // 网关部署（如公网发布）：HTTPS 且无显式端口时，信令挂在同源 /signal 路径下。
  // 默认 :5180 在公网不可达（托管平台只暴露 443），是「创建失败：连接信令服务器
  // 超时」的常见成因。本地开发（vite :5173 / 局域网 IP）仍走 :5180 默认。
  if ( location.protocol === 'https:' && !location.port )
    return `wss://${ host }/signal`;
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${ proto }//${ host }:5180`;
}

/** 读取 URL 查询参数 —— 用于分享/复现一份房间配置，也让界面可被自动化测试。 */
function qp ( name: string ): string | null
{
  return new URLSearchParams( location.search ).get( name );
}

// ------------------------------------------------------------------ 主界面

/**
 * 房间号 = 房间的**密码学身份**：128 位随机数（crypto.getRandomValues），
 * base32 风格去掉易混字符。同名房间因此必然是不同房间 —— 身份由随机性保证，
 * 不由名字保证。注意：这是「不可猜」的识别，还不是「需要口令」的鉴权；
 * 鉴权（加入口令 / 签名）是下一步。
 */
function roomCode (): string
{
  const alphabet = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';
  const bytes = crypto.getRandomValues( new Uint8Array( 10 ) );
  return [ ...bytes ].map( ( b ) => alphabet[ b % alphabet.length ] ).join( '' );
}

export function renderFedApp ( root: HTMLElement, view: 'host' | 'join' ): void
{
  // ---------- 状态 ----------
  let corpus: Corpus | null = null;
  let node: FedNode | null = null;
  let transport: Transport | null = null;
  let manifest: RoomManifest | null = null;
  let role: 'host' | 'peer' | null = null;
  let busy = false;
  let ready = false;
  let self: PeerInfo | null = null;
  let cap: Capability | null = null;

  const curveLocal: Array<{ x: number; y: number }> = [];
  const curveGlobal: Array<{ x: number; y: number }> = [];
  const logs: string[] = [];

  // ---------- 屏幕唤醒锁 ----------
  // 双端联调实测（2026-10-06）：手机默认灭屏后 Chrome 会**冻结页面**，节点从此静默
  // 停止上报，主机等超时后按「掉线止损」继续跑 —— 用户看到的就是「手机一灭屏就掉队」。
  // 训练期间持有 screen 唤醒锁，保住这颗页面的执行权。需要安全上下文（localhost/https 都有）。
  let wakeLock: { release: () => Promise<void> } | null = null;
  let trainingActive = false;

  async function holdWakeLock (): Promise<void>
  {
    trainingActive = true;
    const nav = navigator as Navigator & {
      wakeLock?: { request: ( type: string ) => Promise<{ release: () => Promise<void> }> };
    };
    try
    {
      if ( nav.wakeLock && !wakeLock ) wakeLock = await nav.wakeLock.request( 'screen' );
    }
    catch { /* 被拒绝或不支持：不影响训练本身，只是可能被系统冻结 */ }
  }

  function releaseWakeLock (): void
  {
    trainingActive = false;
    try { void wakeLock?.release().catch( () => {} ); }
    catch { /* 忽略 */ }
    wakeLock = null;
  }

  // 唤醒锁在页面隐藏时会自动释放 —— 重新可见且训练还在进行就补锁
  document.addEventListener( 'visibilitychange', () =>
  {
    if ( document.visibilityState === 'visible' && trainingActive && !wakeLock ) void holdWakeLock();
  } );

  // ---------- 骨架 ----------
  const head = el( 'div', 'fed-head' );
  head.append( el( 'h1', undefined, view === 'host' ? '房主 · 创建并设计训练房间' : '训练节点 · 加入房间' ) );
  head.append( el( 'p', undefined, view === 'host'
    ? '你是房主：任务、模型、参数都由你定义，写进清单下发给所有节点。成员随时可加入退出，不影响训练。'
    : '加入一个房间参与联邦训练。你的数据不出本地，只交换权重。' ) );
  root.append( head );

  // 1 · 连接
  const cardConnect = el( 'section', 'fed-card' );
  const h1 = el( 'h2' ); h1.append( el( 'span', 'idx', '1' ), el( 'span', undefined, view === 'host' ? '创建房间' : '选择房间' ) );
  cardConnect.append( h1 );

  // 开放房间列表：加入方的第一入口。数据来自信令服务器（只读展示），3 秒自动刷新。
  const btnRooms = el( 'button', 'fed-btn', '刷新开放房间' ) as HTMLButtonElement;
  const roomsBox = el( 'div', 'fed-roomlist' );
  if ( view === 'join' )
  {
    cardConnect.append( btnRooms );
    cardConnect.append( roomsBox );
  }

  const rowRoom = el( 'div', 'fed-row' );
  rowRoom.append( el( 'label', undefined, view === 'host' ? '房间号（自动生成）' : '房间号' ) );
  const inRoom = el( 'input', 'fed-input' ) as HTMLInputElement;
  // 默认给一个随机短号：让每个房主都开**自己的**房间，
  // 别让所有人都挤在同一个写死的默认号里（两台手机抢一个房间的混乱就是这么来的）。
  inRoom.value = qp( 'room' ) ?? roomCode();
  rowRoom.append( inRoom );
  cardConnect.append( rowRoom );

  const rowMode = el( 'div', 'fed-row' );
  rowMode.append( el( 'label', undefined, '联机方式' ) );
  const inMode = el( 'select', 'fed-input' ) as HTMLSelectElement;
  for ( const [ value, text ] of [
    [ 'webrtc', '跨设备（WebRTC，需信令服务器）' ],
    [ 'local', '本机多标签页（免服务器）' ],
  ] as Array<[ string, string ]> )
  {
    const opt = el( 'option' ) as HTMLOptionElement;
    opt.value = value;
    opt.textContent = text;
    inMode.append( opt );
  }
  inMode.value = qp( 'mode' ) === 'local' ? 'local' : 'webrtc';
  rowMode.append( inMode );
  cardConnect.append( rowMode );

  const modeHint = el( 'p', 'hint', '' );
  cardConnect.append( modeHint );

  const rowSig = el( 'div', 'fed-row' );
  rowSig.append( el( 'label', undefined, '信令地址' ) );
  const inSignal = el( 'input', 'fed-input' ) as HTMLInputElement;
  inSignal.value = defaultSignalUrl();
  rowSig.append( inSignal );
  cardConnect.append( rowSig );

  const rowName = el( 'div', 'fed-row' );
  rowName.append( el( 'label', undefined, '我的名字' ) );
  const inName = el( 'input', 'fed-input' ) as HTMLInputElement;
  {
    // 默认名字按设备类型猜一个，省用户一步
    const kind = detectDevice( null ).kind;
    inName.value = kind === 'Android' || kind === 'iOS' ? '我的手机' : '我的电脑';
  }
  rowName.append( inName );
  cardConnect.append( rowName );

  const devInfo = el( 'p', 'hint' );
  {
    const d = detectDevice( null );
    devInfo.textContent = `本机：${ d.kind } · ${ d.cores } 核 · 内存约 ${ d.memoryGB || '?' } GB · WebGPU ${ d.webgpu ? 'API 可用' : 'API 不可用' }`;
  }
  cardConnect.append( devInfo );

  // 能力诊断：WebGPU 能不能用，取决于「安全上下文」和「能否拿到适配器」两件事，
  // 两件都可能失败且失败原因不同，所以如实逐项显示 + 给出可执行建议。
  const capBox = el( 'div', 'fed-capbox', '正在探测设备能力…' );
  cardConnect.append( capBox );

  // 引擎策略 —— 只有房主有意义。
  // 刻意**不提供「只用 WebGPU」**：那是唯一的坑 —— 手机走 http 拿不到适配器时
  // 房主一旦强行指定 GPU，手机会被彻底挡在门外。宁可只留「自动/只用 CPU」。
  const rowEngine = el( 'div', 'fed-row' );
  rowEngine.append( el( 'label', undefined, '引擎策略（房主）' ) );
  const inEngine = el( 'select', 'fed-input' ) as HTMLSelectElement;
  for ( const [ value, text ] of [
    [ 'auto', '自动 —— 全网都支持 WebGPU 才用 GPU，否则回退 CPU' ],
    [ 'cpu', '只用 CPU —— 兼容性最好' ],
  ] as Array<[ string, string ]> )
  {
    const opt = el( 'option' ) as HTMLOptionElement;
    opt.value = value;
    opt.textContent = text;
    inEngine.append( opt );
  }
  inEngine.value = 'auto';
  rowEngine.append( inEngine );
  cardConnect.append( rowEngine );

  const engineHint = el( 'p', 'hint' );
  cardConnect.append( engineHint );

  const rowBtns = el( 'div', 'fed-row' );
  const btnHost = el( 'button', 'fed-btn primary', '创建训练房间（主机）' ) as HTMLButtonElement;
  const btnJoin = el( 'button', 'fed-btn', '加入房间（节点）' ) as HTMLButtonElement;
  const btnLeave = el( 'button', 'fed-btn danger', '断开' ) as HTMLButtonElement;
  btnLeave.disabled = true;
  rowBtns.append( btnHost, btnJoin, btnLeave );
  cardConnect.append( rowBtns );

  const status = el( 'div', 'fed-status', '空闲 —— 先选一种角色：创建房间，或用手机加入' );
  cardConnect.append( status );
  root.append( cardConnect );

  // 2 · 任务清单
  const cardTask = el( 'section', 'fed-card' );
  const h2 = el( 'h2' ); h2.append( el( 'span', 'idx', '2' ), el( 'span', undefined, '任务清单（房间宪法）' ) );
  cardTask.append( h2 );
  const taskBody = el( 'div' );
  taskBody.append( el( 'p', 'hint', '尚未加入任何房间。清单确定后，所有节点对「训什么模型、用什么数据」达成一致。' ) );
  cardTask.append( taskBody );
  root.append( cardTask );

  // 3 · 数据
  const cardData = el( 'section', 'fed-card' );
  const h3 = el( 'h2' ); h3.append( el( 'span', 'idx', '3' ), el( 'span', undefined, '数据' ) );
  cardData.append( h3 );
  cardData.append( el( 'p', 'hint',
    '任务数据由房间提供（内置语料按分片切给每个节点）。你也可以粘一段自己的文本贡献进来 —— 加入房间时生效，会并入你的本地训练池，只参与本地训练，不上传。' ) );
  const inContribution = el( 'textarea', 'fed-input' ) as HTMLTextAreaElement;
  inContribution.placeholder = '（可选）粘贴一段你自己的文本，作为本节点的额外训练数据…';
  cardData.append( inContribution );
  const dataNote = el( 'p', 'hint', '' );
  cardData.append( dataNote );
  root.append( cardData );

  // 4 · 节点
  const cardPeers = el( 'section', 'fed-card' );
  const h4 = el( 'h2' ); h4.append( el( 'span', 'idx', '4' ), el( 'span', undefined, '节点' ) );
  cardPeers.append( h4 );
  const peerWrap = el( 'div', 'table-wrap' );
  cardPeers.append( peerWrap );
  root.append( cardPeers );

  // 5 · 训练（这里就是「房间设计」：房主定引擎、模型规模、数据量、超参）
  const cardTrain = el( 'section', 'fed-card' );
  const h5 = el( 'h2' ); h5.append( el( 'span', 'idx', '5' ), el( 'span', undefined, '房间设计' ) );
  cardTrain.append( h5 );
  const designHint = el( 'p', 'hint',
    '由房主定义，开训时写进清单下发给所有节点。'
    + '每台设备每轮最多花 2.5 秒做本地训练：跑不完就少跑几步、如实上报样本数（聚合按实际样本数加权），'
    + '不会为追赶参数把界面卡死。' );
  cardTrain.append( designHint );

  const rowTask = el( 'div', 'fed-row' );
  rowTask.append( el( 'label', undefined, '任务名' ) );
  const inTask = el( 'input', 'fed-input' ) as HTMLInputElement;
  inTask.value = qp( 'task' ) ?? '字符级语言模型 · 联邦预训练';
  rowTask.append( inTask );
  cardTrain.append( rowTask );

  const rowPreset = el( 'div', 'fed-row' );
  rowPreset.append( el( 'label', undefined, '模型规模' ) );
  const inPreset = el( 'select', 'fed-input' ) as HTMLSelectElement;
  for ( const [ value, p ] of Object.entries( MODEL_PRESETS ) )
  {
    const opt = el( 'option' ) as HTMLOptionElement;
    opt.value = value;
    opt.textContent = p.label;
    inPreset.append( opt );
  }
  inPreset.value = qp( 'preset' ) ?? 'medium';
  rowPreset.append( inPreset );
  cardTrain.append( rowPreset );

  const rowParams = el( 'div', 'fed-row' );
  // ?quick=1 给出一个跑得快的小配置（演示/自动化测试用）
  const quick = qp( 'quick' ) !== null;
  const inRounds = numInput( '轮次', qp( 'rounds' ) ?? ( quick ? '6' : '30' ) );
  const inSteps = numInput( '每轮步数', qp( 'steps' ) ?? ( quick ? '8' : '20' ) );
  const inBatch = numInput( '批大小', qp( 'batch' ) ?? ( quick ? '16' : '32' ) );
  const inLr = numInput( '学习率', qp( 'lr' ) ?? '0.02' );
  const inShards = numInput( '分片数', qp( 'shards' ) ?? '4' );
  for ( const [ label, input ] of [ inRounds, inSteps, inBatch, inLr, inShards ] )
  {
    const wrap = el( 'div', 'fed-row' );
    wrap.style.flex = '1 1 92px';
    wrap.style.margin = '0';
    wrap.append( el( 'label', undefined, label ), input );
    rowParams.append( wrap );
  }
  cardTrain.append( rowParams );

  const rowAgg = el( 'div', 'fed-row' );
  rowAgg.append( el( 'label', undefined, '聚合方式' ) );
  const inAgg = el( 'select', 'fed-input' ) as HTMLSelectElement;
  for ( const [ value, text ] of [
    [ 'fedavg', 'FedAvg：直接平均权重（小规模实测更优，默认）' ],
    [ 'diloco', 'DiLoCo：平均「参数增量」+ 外层动量（大 H、多轮时更省通信）' ],
  ] as Array<[ string, string ]> )
  {
    const opt = el( 'option' ) as HTMLOptionElement;
    opt.value = value;
    opt.textContent = text;
    inAgg.append( opt );
  }
  inAgg.value = 'fedavg';
  rowAgg.append( inAgg );
  cardTrain.append( rowAgg );
  cardTrain.append( el( 'p', 'hint',
    '两者共用同一个「基准点 + 加权增量」骨架：FedAvg 就是 DiLoCo 在 β=0、η=1 时的特例。' +
    '本仓库自检实测（2 节点、6 轮、每轮 10 步、MLP 小模型）：FedAvg 2.824 vs DiLoCo(η=0.7,β=0.9) 3.310 —— ' +
    '规模这么小时 FedAvg 反而更好；DiLoCo 的收益要点是大 H（每轮本地步数多）、多轮、链路慢。' ) );

  const rowOpts = el( 'div', 'fed-row' );
  const crossLabel = el( 'label', 'fed-check' );
  const inCross = el( 'input' ) as HTMLInputElement;
  inCross.type = 'checkbox';
  crossLabel.append( inCross, document.createTextNode( '开启交叉校验（用提交的权重复算探针 loss，比对自报值）' ) );
  rowOpts.append( crossLabel );
  cardTrain.append( rowOpts );
  cardTrain.append( el( 'p', 'hint',
    '都是自己的设备时不需要防作弊，关掉交叉校验能省掉每个节点每轮的一次额外评估（手机上尤其有感）。' ) );

  const rowTrainBtns = el( 'div', 'fed-row' );
  const btnStart = el( 'button', 'fed-btn primary', '▶ 开始训练' ) as HTMLButtonElement;
  const btnStop = el( 'button', 'fed-btn danger', '■ 停止' ) as HTMLButtonElement;
  btnStart.disabled = true;
  btnStop.disabled = true;
  rowTrainBtns.append( btnStart, btnStop );
  cardTrain.append( rowTrainBtns );

  const bar = el( 'div', 'bar' );
  const barFill = el( 'i' );
  bar.append( barFill );
  cardTrain.append( bar );

  const chart = el( 'canvas', 'fed-chart' ) as HTMLCanvasElement;
  cardTrain.append( chart );
  const legend = el( 'div', 'legend' );
  legend.append( legendDot( '#4c8dff', '全局探针 loss' ), legendDot( '#3fb950', '本节点本地 loss' ) );
  cardTrain.append( legend );

  const logPre = el( 'pre', 'fed-log', '日志…' );
  cardTrain.append( logPre );
  root.append( cardTrain );

  function numInput ( label: string, value: string ): [ string, HTMLInputElement ]
  {
    const i = el( 'input', 'fed-input' ) as HTMLInputElement;
    i.type = 'number';
    i.value = value;
    i.style.flex = '1 1 40px';
    i.style.minWidth = '44px';
    return [ label, i ];
  }

  // 6 · 账本
  const cardLedger = el( 'section', 'fed-card' );
  const h6 = el( 'h2' ); h6.append( el( 'span', 'idx', '6' ), el( 'span', undefined, '贡献账本（信任层 v0）' ) );
  cardLedger.append( h6 );
  cardLedger.append( el( 'p', 'hint',
    '主机用「共识探针」对每个节点提交的权重复算 loss，与节点自报值比对；不符者剔除并留痕。以下是最近一轮的明细。' ) );
  const ledgerWrap = el( 'div', 'table-wrap' );
  cardLedger.append( ledgerWrap );
  root.append( cardLedger );

  // 7 · 成果
  const cardOut = el( 'section', 'fed-card' );
  const h7 = el( 'h2' ); h7.append( el( 'span', 'idx', '7' ), el( 'span', undefined, '成果' ) );
  cardOut.append( h7 );
  const outBody = el( 'div' );
  outBody.append( el( 'p', 'hint', '训练完成后，可以下载模型卡与全局权重，或直接用训好的模型续写。' ) );
  cardOut.append( outBody );

  const rowGen = el( 'div', 'fed-row' );
  const inGen = el( 'input', 'fed-input' ) as HTMLInputElement;
  inGen.placeholder = '输入一个前缀，模型会贪心续写 80 个字符';
  inGen.value = 'The ';
  const btnGen = el( 'button', 'fed-btn', '续写' ) as HTMLButtonElement;
  btnGen.disabled = true;
  rowGen.append( inGen, btnGen );
  cardOut.append( rowGen );
  const genOut = el( 'pre', 'fed-log', '（续写结果会显示在这里）' );
  cardOut.append( genOut );
  root.append( cardOut );

  function legendDot ( color: string, label: string ): HTMLElement
  {
    const s = el( 'span' );
    const dot = el( 'span', 'dot' ) as HTMLElement;
    dot.style.background = color;
    s.append( dot, document.createTextNode( label ) );
    return s;
  }

  // ---------- 渲染辅助 ----------

  function syncMode (): void
  {
    const local = inMode.value === 'local';
    inSignal.disabled = local;
    inSignal.style.opacity = local ? '0.45' : '1';
    inSignal.parentElement?.querySelector( 'label' )?.setAttribute( 'style', local ? 'opacity:0.45' : '' );
    modeHint.textContent = local
      ? '本机模式：同一个浏览器再开一个标签页，填同一个房间 ID，两边就能互相训练。全程走 BroadcastChannel，不需要任何服务器。'
      : '跨设备模式：手机与电脑在同一 WiFi 下，手机打开终端里打印的局域网地址，填同一个房间 ID 加入即可。需要信令服务器（npm run demo 会一起起）。';
    // 本机总线没有服务器，自然没有「房间列表」；房主页也不需要它（房间号是它自己生成的）
    const showRooms = view === 'join' && inMode.value !== 'local';
    btnRooms.style.display = showRooms ? '' : 'none';
    roomsBox.style.display = showRooms ? '' : 'none';
    if ( showRooms ) void refreshRooms();
    else roomsBox.innerHTML = '';
    renderCapability( cap, null );
    syncRole();
  }

  /**
   * 角色决定权限。
   * 房主：定义引擎策略与全部训练参数 —— 它们写进清单，由房主下发。
   * 节点：这些项一律只读。反正改了也不会生效（清单是房主给的），
   *       明确关掉比让人白改一通、然后纳闷「为什么没生效」好得多。
   */
  function syncRole (): void
  {
    const isPeer = role === 'peer';
    const inputs: Array<HTMLInputElement | HTMLSelectElement> = [
      inRounds[ 1 ], inSteps[ 1 ], inBatch[ 1 ], inLr[ 1 ], inShards[ 1 ], inAgg, inCross, inEngine, inPreset, inTask,
    ];
    for ( const i of inputs )
    {
      i.disabled = isPeer;
      i.style.opacity = isPeer ? '0.5' : '1';
    }
    engineHint.textContent = role === null
      ? '引擎由房主在「开始训练」时按全网能力协商：只要有一台设备跑不了 WebGPU，全网就用 CPU。'
      : isPeer
        ? '你是参与节点：引擎与全部训练参数由房主下发，本机只上报能力并执行。'
        : '你是房主：引擎策略与全部参数由你定义；开训时会按全网能力再复核一次引擎，然后下发给各节点。';
  }

  /** 信令地址 → HTTP 地址（同一个服务器，同一个端口，只是协议不同）。 */
  function signalHttpBase (): string
  {
    return inSignal.value.replace( /^ws/, 'http' ).replace( /\/+$/, '' );
  }

  interface RoomBrief
  {
    roomId: string;
    hasHost: boolean;
    meta: { taskName?: string; engine?: string; preset?: string } | null;
    peers: Array<{ name: string; kind?: string; role: string; gpuOk?: boolean }>;
  }

  /**
   * 拉开放房间列表。
   *
   * 为什么值得做：加入方最自然的动作是「看看有哪些房间、点一个进去」，
   * 而不是「猜房主用了什么号再手打一遍」。信令服务器本来就认识所有房间，
   * 这里只是把名单读出来给人看 —— 它仍然不做任何房间逻辑。
   */
  async function refreshRooms (): Promise<void>
  {
    roomsBox.innerHTML = '';
    roomsBox.append( el( 'div', 'hint', '正在拉取房间列表…' ) );
    try
    {
      const res = await fetch( `${ signalHttpBase() }/rooms` );
      if ( !res.ok ) throw new Error( `HTTP ${ res.status }` );
      const data = await res.json() as { rooms: RoomBrief[] };
      roomsBox.innerHTML = '';
      const open = data.rooms.filter( ( r ) => r.peers.length > 0 );
      if ( open.length === 0 )
      {
        roomsBox.append( el( 'div', 'hint', '现在没有开放的房间。在任一台设备上点「创建训练房间」即可。' ) );
        return;
      }
      for ( const r of open )
      {
        const item = el( 'button', 'fed-roomitem' ) as HTMLButtonElement;
        const parts = [
          r.meta?.taskName ?? '未命名任务',
          `引擎 ${ r.meta?.engine ?? '?' }`,
          `${ r.peers.length } 人`,
          r.hasHost ? '房主在' : '无房主',
        ];
        item.append( el( 'b', undefined, r.roomId ) );
        item.append( el( 'span', 'hint', '  ' + parts.join(' · ') ) );
        item.onclick = () =>
        {
          inRoom.value = r.roomId;
          addLog( `已选择房间 ${ r.roomId }（${ r.meta?.taskName ?? '' }）` );
        };
        roomsBox.append( item );
      }
    }
    catch ( err )
    {
      roomsBox.innerHTML = '';
      roomsBox.append( el( 'div', 'hint', `拿不到房间列表（信令服务器没开？）：${ ( err as Error ).message }` ) );
    }
  }

  /**
   * 向信令服务器上报房间的「自我介绍」，供其它设备在列表里看到。
   * 只是只读展示；房间的一切逻辑仍由房主自己驱动。
   */
  function announceRoom ( engine: string ): void
  {
    if ( !manifest ) return;
    transport?.announce?.( {
      taskName: manifest.taskName,
      engine,
      preset: inPreset.value,
      mode: inMode.value,
    } );
  }

  /**
   * 换引擎时给一套合适的默认超参。
   * MLP 是纯 JS、可以吃大 batch 和学习率；WebGPU tiny-GPT 反之（AdamW 对 lr 敏感得多）。
   * URL 里显式给了的参数不覆盖。
   */
  function applyEngineDefaults (): void
  {
    const gpu = inEngine.value !== 'cpu' && cap?.adapterOk === true;
    const quick = qp( 'quick' ) !== null;
    const set = ( input: HTMLInputElement, name: string, value: string ): void =>
    {
      if ( qp( name ) === null ) input.value = value;
    };
    set( inBatch[ 1 ], 'batch', gpu ? '8' : ( quick ? '16' : '32' ) );
    set( inLr[ 1 ], 'lr', gpu ? '0.003' : '0.02' );
    set( inSteps[ 1 ], 'steps', gpu ? '8' : ( quick ? '8' : '20' ) );
    set( inRounds[ 1 ], 'rounds', quick ? '6' : ( gpu ? '12' : '30' ) );
  }

  function setStatus ( t: string ): void
  {
    status.textContent = t;
  }

  function addLog ( line: string ): void
  {
    const ts = new Date().toLocaleTimeString( 'zh-CN', { hour12: false } );
    logs.push( `[${ ts }] ${ line }` );
    if ( logs.length > 200 ) logs.splice( 0, logs.length - 200 );
    logPre.textContent = logs.join( '\n' );
    logPre.scrollTop = logPre.scrollHeight;
  }

  function renderCapability ( c: Capability | null, error: string | null ): void
  {
    capBox.innerHTML = '';
    if ( error )
    {
      capBox.append( el( 'div', 'bad', `能力探测失败：${ error }` ) );
      return;
    }
    if ( !c )
    {
      capBox.textContent = '正在探测设备能力…';
      return;
    }
    const line = ( label: string, ok: boolean, text: string ): void =>
    {
      const d = el( 'div' );
      d.append( el( 'span', ok ? 'good' : 'bad', ok ? '✓ ' : '✗ ' ) );
      d.append( el( 'b', undefined, label + '：' ) );
      d.append( document.createTextNode( text ) );
      capBox.append( d );
    };
    line( '安全上下文', c.secureContext, c.secureContext ? '是（https 或 localhost）' : '否 —— 浏览器会禁掉 WebGPU' );
    line( 'navigator.gpu', c.hasWebGpuApi, c.hasWebGpuApi ? '存在' : '不存在' );
    line( 'GPU 适配器', c.adapterOk, c.adapterOk
      ? `${ c.vendor || '?' } / ${ c.architecture || '?' } / ${ c.device || '?' }`
      : ( c.adapterError ?? '未取得' ) );
    if ( c.adapterOk )
    {
      line( 'shader-f16', c.hasF16, c.hasF16 ? '支持' : '不支持（只能 fp32）' );
      line( 'maxBufferSize', true, `${ ( c.maxBufferSize / 1024 / 1024 / 1024 ).toFixed( 2 ) } GiB` );
    }
    const usable = c.adapterOk;
    const want = inEngine.value;
    // 引擎最终由房主决定：节点只上报能力，不自己挑。
    // 房主会取「能力下限」—— 只要有一台设备跑不了 WebGPU，全网就用 CPU。
    const willUse = role === 'peer'
      ? '由房主决定（本机能力已上报，房主按全网能力协商）'
      : want === 'cpu'
        ? 'mlp（CPU）—— 房主指定'
        : usable ? 'gpu-tinygpt（WebGPU）—— 开训时还要按全网能力复核' : 'mlp（CPU 回退）';
    const pick = el( 'div' );
    pick.append( el( 'b', undefined, role === 'peer' ? '本机使用的引擎：' : '本机倾向使用的引擎：' ) );
    pick.append( document.createTextNode( willUse ) );
    capBox.append( pick );

    if ( c.advise.length > 0 )
    {
      const a = el( 'div', 'advise' );
      for ( const t of c.advise ) a.append( el( 'div', undefined, '• ' + t ) );
      capBox.append( a );
    }
  }

  function renderManifest ( m: RoomManifest, shardIndex: number ): void
  {
    taskBody.innerHTML = '';
    const dl = el( 'dl', 'kv' );
    const add = ( k: string, v: string ): void =>
    {
      dl.append( el( 'dt', undefined, k ), el( 'dd', undefined, v ) );
    };
    add( '任务', m.taskName );
    add( '清单指纹', m.fingerprint );
    add( '模型', `${ specLabel( m.model ) } · vocab ${ m.model.vocabSize } · seed ${ m.model.seed }` );
    if ( m.engineReason ) add( '引擎为何这么选', m.engineReason );
    add( '轮次 / 每轮步数', `${ m.rounds } × ${ m.localSteps } 步 × ${ m.batchSize } 序列` );
    add( '学习率', String( m.lr ) );
    add( '聚合', m.aggregate.mode === 'diloco'
      ? `DiLoCo（外层 lr ${ m.aggregate.outerLr }，动量 ${ m.aggregate.momentum }）`
      : 'FedAvg（直接平均权重）' );
    add( '交叉校验', m.crossCheck ? '开启' : '关闭（自用设备）' );
    add( '分片', `${ m.shards } 片 · 我是 #${ shardIndex }` );
    add( '语料', `${ m.corpus.name } · ${ m.corpus.chars.toLocaleString() } 字符` );
    add( '语料指纹', m.corpus.digest );
    add( '共识探针', `offset ${ m.probe.offset.toLocaleString() } · ${ m.probe.size } 字符` );
    taskBody.append( dl );

    if ( corpus )
    {
      const mine = corpusDigest( corpus );
      if ( mine !== m.corpus.digest )
      {
        const w = el( 'div', 'warnbox',
          `⚠ 本地语料指纹 ${ mine } 与房间不一致（${ m.corpus.digest }）。可能加载了不同的数据文件，请勿参与本轮训练。` );
        taskBody.append( w );
      }
      dataNote.textContent = `数据来源：任务语料分片 #${ shardIndex }（${ shardOf( corpus, m, shardIndex ) } 字符区间）` +
        ( inContribution.value.trim().length > 0 ? ` + 我贡献的 ${ inContribution.value.trim().length } 字符` : '' );
    }
  }

  function renderRoster ( peers: PeerInfo[], selfId: string ): void
  {
    peerWrap.innerHTML = '';
    const t = el( 'table', 'fed-table' );
    const thead = el( 'thead' );
    const hr = el( 'tr' );
    for ( const h of [ '名字', '设备', 'WebGPU', '角色', 'ID', '状态', '就绪' ] ) hr.append( el( 'th', undefined, h ) );
    thead.append( hr );
    t.append( thead );
    const tb = el( 'tbody' );

    const add = ( p: PeerInfo, isSelf: boolean ): void =>
    {
      const tr = el( 'tr' );
      tr.append( el( 'td', undefined, p.name + ( isSelf ? '（我）' : '' ) ) );
      tr.append( el( 'td', undefined, p.device?.kind ?? '?' ) );
      // 房主协商引擎就看这一列；「API 有但没适配器」是真实存在的情况，要区分开
      const gpuTd = el( 'td' );
      if ( isSelf )
      {
        gpuTd.append( el( 'span', 'tag', cap?.adapterOk ? '可用' : '不可用' ) );
      }
      else if ( p.device?.gpuOk === true )
      {
        gpuTd.append( el( 'span', 'tag ok', '可用' ) );
      }
      else
      {
        gpuTd.append( el( 'span', 'tag warn', p.device?.webgpu ? 'API 有·无适配器' : '无' ) );
      }
      tr.append( gpuTd );
      const roleTd = el( 'td' );
      roleTd.append( el( 'span', `tag ${ p.role === 'host' ? 'host' : '' }`, p.role === 'host' ? '房主/汇聚' : '节点' ) );
      tr.append( roleTd );
      tr.append( el( 'td', undefined, p.peerId.slice( 0, 8 ) ) );
      const stTd = el( 'td' );
      const open = p.peerId === selfId || ( transport?.openPeerIds.includes( p.peerId ) ?? false );
      stTd.append( el( 'span', `tag ${ open ? 'ok' : 'warn' }`, open ? '通道就绪' : '连接中' ) );
      tr.append( stTd );

      // 就绪：只有房主等得到这个信息，所以只在房主侧显示真实值
      const readyTd = el( 'td' );
      if ( role !== 'host' || !node )
      {
        readyTd.append( el( 'span', 'tag', '—' ) );
      }
      else if ( isSelf )
      {
        readyTd.append( el( 'span', 'tag host', '房主' ) );
      }
      else
      {
        const why = node.notReadyReasons.get( p.peerId );
        const isReady = node.readyIds.includes( p.peerId );
        const tag = el( 'span', `tag ${ isReady ? 'ok' : why ? 'warn' : '' }`,
          isReady ? '已就绪' : why ? '未就绪' : '等待' );
        if ( why ) tag.title = why;
        // 未就绪的原因写在旁边，别让人去翻日志
        if ( why ) { readyTd.append( tag, el( 'span', 'hint', ` ${ why }` ) ); }
        else readyTd.append( tag );
      }
      tr.append( readyTd );
      tb.append( tr );
    };

    if ( self ) add( self, true );
    for ( const p of peers ) if ( p.peerId !== selfId ) add( p, false );
    t.append( tb );

    const others = peers.filter( ( p ) => p.peerId !== selfId ).length;
    if ( role === 'host' )
    {
      btnStart.disabled = busy || manifest === null;
      peerWrap.append( t );
      const note = el( 'p', 'hint', `房间内其他节点：${ others } 个。可以等手机加入后再开始，也可以单机先跑通。` );
      peerWrap.append( note );
      return;
    }
    peerWrap.append( t );
  }

  function renderRound ( stats: RoundStats, total: number ): void
  {
    barFill.style.width = `${ Math.min( 100, ( stats.round / Math.max( 1, total ) ) * 100 ).toFixed( 1 ) }%`;
    ledgerWrap.innerHTML = '';
    const t = el( 'table', 'fed-table' );
    const thead = el( 'thead' );
    const hr = el( 'tr' );
    for ( const h of [ '轮', '节点', '设备', '样本', '探针 loss', 'Δ范数', '份额', '裁决' ] ) hr.append( el( 'th', undefined, h ) );
    thead.append( hr );
    t.append( thead );
    const tb = el( 'tbody' );
    for ( const e of stats.entries )
    {
      const tr = el( 'tr' );
      tr.append( el( 'td', undefined, String( e.round ) ) );
      tr.append( el( 'td', undefined, e.name ) );
      tr.append( el( 'td', undefined, e.deviceKind ) );
      tr.append( el( 'td', undefined, e.samples ? String( e.samples ) : '—' ) );
      tr.append( el( 'td', undefined, e.probeLoss ? e.probeLoss.toFixed( 3 ) : '—' ) );
      tr.append( el( 'td', undefined, e.deltaNorm ? e.deltaNorm.toFixed( 2 ) : '—' ) );
      tr.append( el( 'td', undefined, `${ ( e.share * 100 ).toFixed( 0 ) }%` ) );
      const v = el( 'td' );
      v.append( el( 'span', `tag ${ verdictClass( e ) }`, verdictText( e ) ) );
      v.title = e.note;
      tr.append( v );
      tb.append( tr );
    }
    t.append( tb );
    ledgerWrap.append( t );

    const sum = el( 'p', 'hint',
      `第 ${ stats.round } 轮 · 全局探针 loss ${ stats.prevGlobalLoss.toFixed( 3 ) } → ${ stats.globalLoss.toFixed( 3 ) } · ` +
      `在线 ${ stats.online } · 参与聚合 ${ stats.aggregated } · 本轮流量 ${ fmtBytes( stats.bytes ) } · 耗时 ${ stats.elapsedMs.toFixed( 0 ) }ms` );
    ledgerWrap.append( sum );
  }

  function verdictClass ( e: LedgerEntry ): string
  {
    if ( e.verdict === 'ok' ) return 'ok';
    if ( e.verdict === 'suspect' ) return 'bad';
    return 'warn';
  }

  function verdictText ( e: LedgerEntry ): string
  {
    if ( e.verdict === 'ok' ) return '通过';
    if ( e.verdict === 'suspect' ) return '异常·已剔除';
    return '超时·已剔除';
  }

  // ---------- 曲线 ----------

  function drawChart (): void
  {
    const dpr = Math.min( 2, window.devicePixelRatio || 1 );
    const cssW = chart.clientWidth || 520;
    const cssH = 180;
    if ( chart.width !== Math.round( cssW * dpr ) )
    {
      chart.width = Math.round( cssW * dpr );
      chart.height = Math.round( cssH * dpr );
    }
    const ctx = chart.getContext( '2d' );
    if ( !ctx ) return;
    ctx.setTransform( dpr, 0, 0, dpr, 0, 0 );
    const W = cssW;
    const H = cssH;
    const pad = 32;
    ctx.clearRect( 0, 0, W, H );
    ctx.fillStyle = '#0f141c';
    ctx.fillRect( 0, 0, W, H );

    const all = [ ...curveGlobal, ...curveLocal ];
    if ( all.length < 2 )
    {
      ctx.fillStyle = '#8b98a9';
      ctx.font = '12px monospace';
      ctx.fillText( '等待训练数据…', pad, H / 2 );
      return;
    }
    const maxY = Math.max( 0.5, ...all.map( ( p ) => p.y ) ) * 1.05;
    const maxX = Math.max( ...all.map( ( p ) => p.x ) );
    const px = ( x: number ): number => pad + ( ( W - pad - 10 ) * x ) / Math.max( 1, maxX );
    const py = ( y: number ): number => H - 22 - ( ( H - 44 ) * y ) / maxY;

    ctx.strokeStyle = '#1e2635';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for ( const gy of [ 0, 0.25, 0.5, 0.75, 1 ] )
    {
      const y = py( gy * maxY );
      ctx.moveTo( pad, y );
      ctx.lineTo( W - 10, y );
    }
    ctx.stroke();
    ctx.fillStyle = '#8b98a9';
    ctx.font = '10px monospace';
    ctx.fillText( maxY.toFixed( 2 ), 3, py( maxY ) + 9 );
    ctx.fillText( '0', 3, py( 0 ) + 4 );
    ctx.fillText( `轮次 ${ maxX }`, W - 58, H - 6 );

    const draw = ( pts: Array<{ x: number; y: number }>, color: string ): void =>
    {
      if ( pts.length < 2 ) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      pts.forEach( ( p, i ) =>
      {
        if ( i === 0 ) ctx.moveTo( px( p.x ), py( p.y ) );
        else ctx.lineTo( px( p.x ), py( p.y ) );
      } );
      ctx.stroke();
    };
    draw( curveGlobal, '#4c8dff' );
    draw( curveLocal, '#3fb950' );
  }

  window.addEventListener( 'resize', drawChart );

  // ---------- 事件桥 ----------

  const events: NodeEvents = {
    onStatus: setStatus,
    onLog: addLog,
    onRoster: ( peers, selfId ) => renderRoster( peers, selfId ),
    onManifest: ( m, shardIndex ) =>
    {
      // 房主在开训时、节点在收到 assign 时都会走到这 —— 训练一开始就保住屏幕
      void holdWakeLock();
      renderManifest( m, shardIndex );
    },
    onRound: ( stats, total ) =>
    {
      // 节点侧拿不到主机的 onCurve，用 round/close 里的全局 loss 补曲线
      if ( role === 'peer' ) curveGlobal.push( { x: stats.round, y: stats.globalLoss } );
      renderRound( stats, total );
      drawChart();
    },
    onCurve: ( p ) =>
    {
      curveLocal.push( { x: p.round, y: p.local } );
      if ( p.global > 0 ) curveGlobal.push( { x: p.round, y: p.global } );
      drawChart();
    },
    onDone: ( c ) =>
    {
      releaseWakeLock();
      renderCard( c );
      setBusy( false );
      // 房主视角：训练完一场后可以随时再开一场（按钮文案随之变化）
      if ( role === 'host' || view === 'host' )
      {
        btnStart.textContent = '▶ 再次训练';
        btnStart.disabled = false;
      }
    },
  };

  // ---------- 成果渲染 ----------

  function renderCard ( c: ModelCard ): void
  {
    outBody.innerHTML = '';
    const dl = el( 'dl', 'kv' );
    const add = ( k: string, v: string ): void => { dl.append( el( 'dt', undefined, k ), el( 'dd', undefined, v ) ); };
    const secs = ( c.finishedAt - c.startedAt ) / 1000;
    add( '耗时', `${ secs.toFixed( 1 ) }s` );
    add( '参数量', c.paramCount.toLocaleString() );
    add( '探针 loss', `${ c.initialProbeLoss > 0 ? c.initialProbeLoss.toFixed( 3 ) : '—' }（初始） → ${ c.finalProbeLoss.toFixed( 3 ) }（最终）` );
    add( '全局权重摘要', c.weightsDigest );
    add( '传输总量', fmtBytes( c.transportBytes ) );
    outBody.append( dl );

    const ct = el( 'table', 'fed-table' );
    const thead = el( 'thead' );
    const hr = el( 'tr' );
    for ( const h of [ '贡献者', '设备', '参与轮次', '累计样本', '通过/异常', '贡献数据' ] ) hr.append( el( 'th', undefined, h ) );
    thead.append( hr );
    ct.append( thead );
    const tb = el( 'tbody' );
    for ( const p of c.contributors )
    {
      const tr = el( 'tr' );
      tr.append( el( 'td', undefined, p.name ) );
      tr.append( el( 'td', undefined, p.deviceKind ) );
      tr.append( el( 'td', undefined, String( p.rounds ) ) );
      tr.append( el( 'td', undefined, p.samples.toLocaleString() ) );
      tr.append( el( 'td', undefined, `${ p.ok } / ${ p.suspect }` ) );
      tr.append( el( 'td', undefined, p.contributedChars ? `${ p.contributedChars } 字符` : '—' ) );
      tb.append( tr );
    }
    ct.append( tb );
    outBody.append( ct );

    const row = el( 'div', 'fed-row' );
    const btnCard = el( 'button', 'fed-btn', '⬇ 下载模型卡 JSON' ) as HTMLButtonElement;
    const btnW = el( 'button', 'fed-btn', '⬇ 下载全局权重' ) as HTMLButtonElement;
    btnCard.onclick = () =>
    {
      const blob = new Blob( [ JSON.stringify( c, null, 2 ) ], { type: 'application/json' } );
      downloadBlob( blob, `model-card-${ c.manifest.roomId }.json` );
    };
    btnW.onclick = () =>
    {
      void ( async () =>
      {
        const buf = await node?.exportGlobalWeights();
        if ( !buf ) return;
        downloadBlob( new Blob( [ buf ], { type: 'application/octet-stream' } ), `global-weights-${ c.manifest.roomId }.wlfd` );
      } )();
    };
    row.append( btnCard, btnW );
    outBody.append( row );
    outBody.append( el( 'p', 'hint', c.note ) );
    btnGen.disabled = false;
  }

  function downloadBlob ( blob: Blob, name: string ): void
  {
    const url = URL.createObjectURL( blob );
    const a = el( 'a' ) as HTMLAnchorElement;
    a.href = url;
    a.download = name;
    a.click();
    window.setTimeout( () => URL.revokeObjectURL( url ), 4000 );
  }

  // ---------- 角色切换 ----------

  function setBusy ( v: boolean ): void
  {
    busy = v;
    btnStart.disabled = v || manifest === null || role !== 'host';
    btnStop.disabled = !v || role !== 'host';
    btnHost.disabled = v || transport !== null || !ready;
    btnJoin.disabled = v || transport !== null || !ready;
    btnLeave.disabled = transport === null;
  }

  /** 语料与能力探测都完成后才允许建房/加入 —— 否则 createEngine 拿不到 GPU 上下文。 */
  function markReady (): void
  {
    ready = true;
    setBusy( busy );
  }

  /**
   * 构建房间清单。
   *
   * `engineOverride` 由 FedNode 在**开训那一刻**给出 —— 它按全网能力协商出唯一
   * 可行的引擎。不传时（建房时的预览清单）先按本机能力猜一个，
   * 点「开始训练」时会被协商结果覆盖。
   */
  function buildManifest (
    corpusRef: Corpus,
    engineOverride?: 'mlp' | 'gpu-tinygpt',
    engineReason?: string,
  ): RoomManifest
  {
    // 引擎必须全网一致：权重形状不同根本没法聚合。
    const useGpu = engineOverride
      ? engineOverride === 'gpu-tinygpt'
      : ( inEngine.value !== 'cpu' && cap?.adapterOk === true );

    // 模型规模由房主在「房间设计」里选。两种引擎各有一套对应规格，
    // 但**同一档位**的语义是一致的：small=快速验证，medium=能看清 loss 在动，large=认真观察。
    const preset: ModelPreset = ( inPreset.value in MODEL_PRESETS )
      ? ( inPreset.value as ModelPreset )
      : 'medium';
    const dims = MODEL_PRESETS[ preset ];

    const model: ModelSpec = useGpu
      ? {
        engine: 'gpu-tinygpt',
        vocabSize: corpusRef.vocab.length,
        ...dims.gpu,
        seed: 20261006,
      }
      : {
        engine: 'mlp',
        vocabSize: corpusRef.vocab.length,
        ...dims.mlp,
        seed: 20261006,
      };

    const aggregate: AggregateSpec = {
      mode: inAgg.value === 'fedavg' ? 'fedavg' : 'diloco',
      outerLr: 1,
      momentum: 0.9,
    };

    const base = {
      roomId: inRoom.value.trim() || 'room',
      taskName: inTask.value.trim() || '字符级语言模型 · 联邦预训练',
      taskBrief: '各节点在自己的语料分片上本地训练，按样本数加权聚合，合出一个全局语言模型。',
      model,
      engineReason: engineReason
        ?? ( useGpu
          ? '建房时按本机能力初选 WebGPU（开训时会按全网能力复核）'
          : '建房时按本机能力初选 CPU（开训时会按全网能力复核）' ),
      rounds: Math.max( 1, Number( inRounds[ 1 ].value ) || 30 ),
      localSteps: Math.max( 1, Number( inSteps[ 1 ].value ) || 20 ),
      batchSize: Math.max( 1, Number( inBatch[ 1 ].value ) || 32 ),
      lr: Math.max( 1e-5, Number( inLr[ 1 ].value ) || 0.02 ),
      shards: Math.max( 1, Number( inShards[ 1 ].value ) || 4 ),
      corpus: {
        name: corpusRef.name,
        digest: corpusDigest( corpusRef ),
        chars: corpusRef.text.length,
        vocab: corpusRef.vocab,
      },
      probe: pickProbe( corpusRef, 1024 ),
      crossCheck: inCross.checked,
      aggregate,
      createdAt: Date.now(),
    };
    return { ...base, fingerprint: fingerprintOf( base ) };
  }

  async function connect ( wantRole: 'host' | 'peer' ): Promise<void>
  {
    if ( !corpus ) throw new Error( '语料尚未加载完成' );
    if ( inShards[ 1 ].value === '' )
    {
      inShards[ 1 ].value = '4';
    }
    const roomId = inRoom.value.trim();
    if ( !roomId ) throw new Error( '请填写房间 ID' );

    self = {
      peerId: randomId(),
      name: inName.value.trim() || '未命名设备',
      role: wantRole,
      device: detectDevice( cap ),
      joinedAt: Date.now(),
    };
    role = wantRole;
    syncRole();
    addLog( `以 ${ wantRole === 'host' ? '房主' : '参与节点' } 身份加入房间 ${ roomId }（我的 ID ${ self.peerId.slice( 0, 8 ) }）` );

    const callbacks = {
      onControl: ( peerId: string, msg: unknown ): void => node?.onControl( peerId, msg as ControlMessage ),
      onBinary: ( peerId: string, buf: ArrayBuffer ): void => node?.onBinary( peerId, buf ),
      onPeerOpen: ( peerId: string ): void => node?.onPeerOpen( peerId ),
      onPeerClose: ( peerId: string ): void => node?.onPeerClose( peerId ),
      onRoster: ( peers: PeerInfo[] ): void => node?.onRoster( peers ),
      onStatus: ( s: string ): void => setStatus( s ),
    };

    const useLocal = inMode.value === 'local';
    const t: Transport = useLocal
      ? new LocalBus( { roomId, self, ...callbacks } )
      : new RoomTransport( { signalUrl: inSignal.value.trim(), roomId, self, ...callbacks } );
    transport = t;

    // 顺序很重要：先建节点、再连通道。
    // 反过来的话，本机通道的 assign 可能在 FedNode 存在之前就送达而被丢掉 ——
    // 症状是节点收不到任务清单，训练时一声不响（主机每轮白等 30s）。
    const m = wantRole === 'host' ? buildManifest( corpus ) : undefined;
    if ( m ) manifest = m;
    node = new FedNode( {
      role: wantRole,
      transport: t,
      events,
      corpus,
      contributionText: inContribution.value,
      manifest: m,
      // 引擎策略只在房主侧有意义：真正的抉择发生在开训时，按**全网**能力协商
      enginePolicy: inEngine.value === 'cpu' ? 'cpu' : 'auto',
    } );

    addLog( useLocal ? '使用本机总线（BroadcastChannel），无需信令服务器' : `使用 WebRTC，信令 ${ inSignal.value.trim() }` );
    setBusy( true );
    btnLeave.disabled = false;
    try
    {
      await t.connect();
    }
    catch ( err )
    {
      resetConnection();
      throw err;
    }
    setBusy( false );

    if ( wantRole === 'host' )
    {
      addLog( `清单指纹 ${ m!.fingerprint }（模型 vocab ${ m!.model.vocabSize } · 分片 ${ m!.shards }）` );
      announceRoom( m!.model.engine );
      setStatus( `房间已创建（房间号 ${ roomId }）—— 加入方在「刷新开放房间」里点一下就能进；然后你点「开始训练」` );
    }
    else
    {
      setStatus( '已加入，等待房主下发任务…（房主点「开始训练」后你才会收到清单）' );
    }
    renderRoster( t.peerInfos, self.peerId );
  }

  function resetConnection (): void
  {
    node?.stop();
    releaseWakeLock();
    transport?.close();
    transport = null;
    node = null;
    manifest = null;
    role = null;
    self = null;
    setBusy( false );
    btnStart.disabled = true;
    btnGen.disabled = true;
    syncRole();
  }

  // ---------- 按钮 ----------

  btnHost.onclick = () =>
  {
    void ( async () =>
    {
      try { await connect( 'host' ); }
      catch ( err ) { setStatus( `创建失败：${ ( err as Error ).message }` ); addLog( `创建失败：${ ( err as Error ).message }` ); setBusy( false ); }
    } )();
  };
  btnJoin.onclick = () =>
  {
    void ( async () =>
    {
      try { await connect( 'peer' ); }
      catch ( err ) { setStatus( `加入失败：${ ( err as Error ).message }` ); addLog( `加入失败：${ ( err as Error ).message }` ); setBusy( false ); }
    } )();
  };
  btnLeave.onclick = () =>
  {
    resetConnection();
    setStatus( '已断开' );
    addLog( '已断开与房间的连接' );
  };
  btnRooms.onclick = () => void refreshRooms();
  btnStart.onclick = () =>
  {
    if ( !node ) return;
    setBusy( true );
    curveLocal.length = 0;
    curveGlobal.length = 0;
    // 引擎在**这一刻**才最终确定：房主按房间里所有节点的能力协商（见 FedNode.negotiateEngine），
    // 只要有一台设备跑不了 WebGPU，全网就回退 CPU。
    addLog( '开始训练：房主按全网能力协商引擎 → 下发任务 → 等各节点就绪 → 开轮' );
    node
      .startHost( ( engine, reason ) => buildManifest( corpus!, engine, reason ) )
      .catch( ( err ) =>
      {
        // 开训失败绝不能静默 —— 用户视角的「点了没反应」就是这么来的
        const msg = ( err as Error ).message ?? String( err );
        addLog( `开训失败：${ msg }` );
        setStatus( `开训失败：${ msg }` );
      } )
      .finally( () => setBusy( false ) );
  };
  btnStop.onclick = () =>
  {
    releaseWakeLock();
    node?.stop();
  };
  btnGen.onclick = () =>
  {
    void ( async () =>
    {
      const m = node?.currentManifest;
      if ( !node || !m ) return;
      const vocab = node.vocab;
      const stoi = new Map( vocab.map( ( c, i ) => [ c, i ] ) );
      const prefix: number[] = [];
      for ( const ch of inGen.value ) if ( stoi.has( ch ) ) prefix.push( stoi.get( ch )! );
      btnGen.disabled = true;
      try
      {
        const ids = await node.engineRef.sample( prefix, 80 );
        genOut.textContent = inGen.value + ids.map( ( i ) => vocab[ i ] ?? '' ).join( '' );
      }
      catch ( err )
      {
        genOut.textContent = `续写失败：${ ( err as Error ).message }`;
      }
      finally
      {
        btnGen.disabled = false;
      }
    } )();
  };

  // ---------- 启动 ----------

  // 按视图裁剪：房主页和节点页的职责本来就不同，混在一页里只会让人不知道该点哪个。
  if ( view === 'join' )
  {
    // 加入方不需要「房间设计」（那是房主的事），也不该看到开始/停止
    cardTrain.remove();
    btnHost.style.display = 'none';
    btnStart.style.display = 'none';
    btnStop.style.display = 'none';
  }
  else
  {
    // 房主不需要「挑房间」—— 房间号是它自己生成的
    btnRooms.style.display = 'none';
    roomsBox.style.display = 'none';
    btnJoin.style.display = 'none';
  }
  // 两页互通：谁都能当房主，也能随时换角色
  {
    const switchBar = el( 'p', 'hint' );
    const a = document.createElement( 'a' );
    // 跨页带上当前查询参数（signal / room 等）——不然公网部署下换页后
    // 信令地址会退回 :5180 默认，又是连不上。
    a.href = `/pages/${ view === 'host' ? 'join' : 'host' }.html${ location.search }`;
    a.textContent = view === 'host' ? '→ 切换到加入页（浏览开放房间）' : '→ 切换到房主页（创建并设计房间）';
    switchBar.append( a );
    head.append( switchBar );
  }
  // 开放房间列表 3 秒自动刷新 —— 不然房主建好房了，加入方看到的还是旧的空列表。
  // 这正是「根本不知道加入哪个房间」的成因：列表只拉了一次。
  const roomPoll = window.setInterval( () =>
  {
    if ( role !== null || view !== 'join' || inMode.value === 'local' ) return;
    void refreshRooms();
  }, 3000 );
  window.addEventListener( 'beforeunload', () => window.clearInterval( roomPoll ) );

  drawChart();
  setBusy( false );
  syncMode();
  inMode.onchange = syncMode;
  inEngine.onchange = () =>
  {
    applyEngineDefaults();
    renderCapability( cap, null );
  };

  void ( async () =>
  {
    let corpusOk = false;
    try
    {
      corpus = await loadBuiltinCorpus();
      dataNote.textContent = `任务语料已加载：${ corpus.name } · ${ corpus.text.length.toLocaleString() } 字符 · 字符表 ${ corpus.vocab.length }`;
      addLog( `内置语料就绪：${ corpus.text.length } 字符，字符表 ${ corpus.vocab.length } 个（含大小写、标点、换行）` );
      corpusOk = true;
    }
    catch ( err )
    {
      setStatus( `语料加载失败：${ ( err as Error ).message }` );
    }

    // 能力探测（同时把 WebGPU 上下文初始化好，createEngine 之后才拿得到）
    try
    {
      cap = await probeCapability();
      addLog(
        `设备能力：安全上下文=${ cap.secureContext }，navigator.gpu=${ cap.hasWebGpuApi }，` +
        `适配器=${ cap.adapterOk ? `可用（${ cap.device || cap.vendor || '?' }）` : `不可用：${ cap.adapterError ?? '未知' }` }`,
      );
      for ( const t of cap.advise ) addLog( '建议：' + t );
    }
    catch ( err )
    {
      renderCapability( null, ( err as Error ).message );
      addLog( `能力探测失败：${ ( err as Error ).message }` );
    }

    applyEngineDefaults();
    renderCapability( cap, null );
    if ( corpusOk ) markReady();
  } )();
}
