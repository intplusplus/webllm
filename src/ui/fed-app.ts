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
  type ControlMessage,
  type DevCap,
  type LedgerEntry,
  type ModelSpec,
  type RoomManifest,
  type RoundStats,
} from '../fed/protocol';
import { RoomTransport, type PeerInfo } from '../fed/transport';
import { detectKind, FedNode, type ModelCard, type NodeEvents } from '../fed/node';

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

function detectDevice (): DevCap
{
  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    kind: detectKind(),
    webgpu: typeof navigator.gpu !== 'undefined',
    cores: nav.hardwareConcurrency ?? 0,
    memoryGB: nav.deviceMemory ?? 0,
    ua: navigator.userAgent,
  };
}

function defaultSignalUrl (): string
{
  const q = new URLSearchParams( location.search ).get( 'signal' );
  if ( q ) return q;
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${ proto }//${ location.hostname || '127.0.0.1' }:5180`;
}

// ------------------------------------------------------------------ 主界面

export function renderFedApp ( root: HTMLElement ): void
{
  // ---------- 状态 ----------
  let corpus: Corpus | null = null;
  let node: FedNode | null = null;
  let transport: RoomTransport | null = null;
  let manifest: RoomManifest | null = null;
  let role: 'host' | 'peer' | null = null;
  let busy = false;
  let self: PeerInfo | null = null;

  const curveLocal: Array<{ x: number; y: number }> = [];
  const curveGlobal: Array<{ x: number; y: number }> = [];
  const logs: string[] = [];

  // ---------- 骨架 ----------
  const head = el( 'div', 'fed-head' );
  head.append( el( 'h1', undefined, '公共训练网络 · 联邦联训 Demo' ) );
  head.append( el( 'p', undefined,
    '浏览器开箱参与：各设备在自己的数据上训练，通过 P2P 交换权重做联邦平均，数据不出本地。' ) );
  root.append( head );

  // 1 · 连接
  const cardConnect = el( 'section', 'fed-card' );
  const h1 = el( 'h2' ); h1.append( el( 'span', 'idx', '1' ), el( 'span', undefined, '连接房间' ) );
  cardConnect.append( h1 );

  const rowRoom = el( 'div', 'fed-row' );
  rowRoom.append( el( 'label', undefined, '房间 ID' ) );
  const inRoom = el( 'input', 'fed-input' ) as HTMLInputElement;
  inRoom.value = 'wifi-lab';
  rowRoom.append( inRoom );
  cardConnect.append( rowRoom );

  const rowSig = el( 'div', 'fed-row' );
  rowSig.append( el( 'label', undefined, '信令地址' ) );
  const inSignal = el( 'input', 'fed-input' ) as HTMLInputElement;
  inSignal.value = defaultSignalUrl();
  rowSig.append( inSignal );
  cardConnect.append( rowSig );

  const rowName = el( 'div', 'fed-row' );
  rowName.append( el( 'label', undefined, '我的名字' ) );
  const inName = el( 'input', 'fed-input' ) as HTMLInputElement;
  inName.value = detectDevice().kind === 'Android' || detectDevice().kind === 'iOS' ? '我的手机' : '我的电脑';
  rowName.append( inName );
  cardConnect.append( rowName );

  const devInfo = el( 'p', 'hint' );
  {
    const d = detectDevice();
    devInfo.textContent = `本机：${ d.kind } · ${ d.cores } 核 · WebGPU ${ d.webgpu ? '可用' : '不可用（本 demo 默认引擎不需要它）' }`;
  }
  cardConnect.append( devInfo );

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

  // 5 · 训练
  const cardTrain = el( 'section', 'fed-card' );
  const h5 = el( 'h2' ); h5.append( el( 'span', 'idx', '5' ), el( 'span', undefined, '训练' ) );
  cardTrain.append( h5 );

  const rowParams = el( 'div', 'fed-row' );
  const inRounds = numInput( '轮次', '30' );
  const inSteps = numInput( '每轮步数', '20' );
  const inBatch = numInput( '批大小', '32' );
  const inLr = numInput( '学习率', '0.02' );
  const inShards = numInput( '分片数', '4' );
  for ( const [ label, input ] of [ inRounds, inSteps, inBatch, inLr, inShards ] )
  {
    const wrap = el( 'div', 'fed-row' );
    wrap.style.flex = '1 1 92px';
    wrap.style.margin = '0';
    wrap.append( el( 'label', undefined, label ), input );
    rowParams.append( wrap );
  }
  cardTrain.append( rowParams );

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
    add( '模型', `${ m.model.engine } · vocab ${ m.model.vocabSize } · ctx ${ m.model.ctx } · emb ${ m.model.embDim } · hidden ${ m.model.hidden } · seed ${ m.model.seed }` );
    add( '轮次 / 每轮步数', `${ m.rounds } × ${ m.localSteps } 步 × ${ m.batchSize } 序列` );
    add( '学习率', String( m.lr ) );
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
    for ( const h of [ '名字', '设备', '角色', 'ID', '状态' ] ) hr.append( el( 'th', undefined, h ) );
    thead.append( hr );
    t.append( thead );
    const tb = el( 'tbody' );

    const add = ( p: PeerInfo, isSelf: boolean ): void =>
    {
      const tr = el( 'tr' );
      tr.append( el( 'td', undefined, p.name + ( isSelf ? '（我）' : '' ) ) );
      tr.append( el( 'td', undefined, p.device?.kind ?? '?' ) );
      const roleTd = el( 'td' );
      roleTd.append( el( 'span', `tag ${ p.role === 'host' ? 'host' : '' }`, p.role === 'host' ? '主机/汇聚' : '节点' ) );
      tr.append( roleTd );
      tr.append( el( 'td', undefined, p.peerId.slice( 0, 8 ) ) );
      const stTd = el( 'td' );
      const open = p.peerId === selfId || ( transport?.openPeerIds.includes( p.peerId ) ?? false );
      stTd.append( el( 'span', `tag ${ open ? 'ok' : 'warn' }`, open ? '通道就绪' : '连接中' ) );
      tr.append( stTd );
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
    onManifest: ( m, shardIndex ) => renderManifest( m, shardIndex ),
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
      renderCard( c );
      setBusy( false );
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
      const buf = node?.exportGlobalWeights();
      if ( !buf ) return;
      downloadBlob( new Blob( [ buf ], { type: 'application/octet-stream' } ), `global-weights-${ c.manifest.roomId }.wlfd` );
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
    btnHost.disabled = v || transport !== null;
    btnJoin.disabled = v || transport !== null;
    btnLeave.disabled = transport === null;
  }

  function buildManifest ( corpusRef: Corpus ): RoomManifest
  {
    const model: ModelSpec = {
      engine: 'mlp',
      vocabSize: corpusRef.vocab.length,
      ctx: 8,
      embDim: 16,
      hidden: 64,
      seed: 20261006,
    };
    const base = {
      roomId: inRoom.value.trim() || 'room',
      taskName: '字符级语言模型 · 联邦预训练',
      taskBrief: '各节点在自己的语料分片上本地训练，按样本数加权做 FedAvg，合出一个全局语言模型。',
      model,
      rounds: Math.max( 1, Number( inRounds[ 1 ].value ) || 30 ),
      localSteps: Math.max( 1, Number( inSteps[ 1 ].value ) || 20 ),
      batchSize: Math.max( 1, Number( inBatch[ 1 ].value ) || 32 ),
      lr: Math.max( 1e-4, Number( inLr[ 1 ].value ) || 0.02 ),
      shards: Math.max( 1, Number( inShards[ 1 ].value ) || 4 ),
      corpus: {
        name: corpusRef.name,
        digest: corpusDigest( corpusRef ),
        chars: corpusRef.text.length,
        vocab: corpusRef.vocab,
      },
      probe: pickProbe( corpusRef, 1024 ),
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
      device: detectDevice(),
      joinedAt: Date.now(),
    };
    role = wantRole;
    addLog( `以 ${ wantRole === 'host' ? '主机' : '节点' } 身份加入房间 ${ roomId }（我的 ID ${ self.peerId.slice( 0, 8 ) }）` );

    const t = new RoomTransport( {
      signalUrl: inSignal.value.trim(),
      roomId,
      self,
      onControl: ( peerId, msg ) => node?.onControl( peerId, msg as unknown as ControlMessage ),
      onBinary: ( peerId, buf ) => node?.onBinary( peerId, buf ),
      onPeerOpen: ( peerId ) => node?.onPeerOpen( peerId ),
      onPeerClose: ( peerId ) => node?.onPeerClose( peerId ),
      onRoster: ( peers ) => node?.onRoster( peers ),
      onStatus: ( s ) => setStatus( s ),
    } );
    transport = t;
    setBusy( true );
    btnLeave.disabled = false;
    await t.connect();

    const m = wantRole === 'host' ? buildManifest( corpus ) : undefined;
    if ( m ) manifest = m;
    node = new FedNode( {
      role: wantRole,
      transport: t,
      events,
      corpus,
      contributionText: inContribution.value,
      manifest: m,
    } );
    setBusy( false );

    if ( wantRole === 'host' )
    {
      addLog( `清单指纹 ${ m!.fingerprint }（模型 vocab ${ m!.model.vocabSize } · 分片 ${ m!.shards }）` );
      setStatus( '房间已创建 —— 手机在同一 WiFi 下打开本页并加入，然后点「开始训练」' );
    }
    else
    {
      setStatus( '已加入，等待主机下发任务清单…' );
    }
    renderRoster( t.peerInfos, self.peerId );
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
    node?.stop();
    transport?.close();
    transport = null;
    node = null;
    manifest = null;
    role = null;
    self = null;
    setBusy( false );
    btnStart.disabled = true;
    btnGen.disabled = true;
    setStatus( '已断开' );
    addLog( '已断开与房间的连接' );
  };
  btnStart.onclick = () =>
  {
    if ( !node ) return;
    setBusy( true );
    curveLocal.length = 0;
    curveGlobal.length = 0;
    void node.startHost().finally( () => setBusy( false ) );
  };
  btnStop.onclick = () => node?.stop();
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

  drawChart();
  setBusy( false );

  void ( async () =>
  {
    try
    {
      corpus = await loadBuiltinCorpus();
      dataNote.textContent = `任务语料已加载：${ corpus.name } · ${ corpus.text.length.toLocaleString() } 字符 · 字符表 ${ corpus.vocab.length }`;
      addLog( `内置语料就绪：${ corpus.text.length } 字符，字符表 ${ corpus.vocab.length } 个（含大小写、标点、换行）` );
    }
    catch ( err )
    {
      setStatus( `语料加载失败：${ ( err as Error ).message }` );
      btnHost.disabled = true;
      btnJoin.disabled = true;
    }
  } )();
}
