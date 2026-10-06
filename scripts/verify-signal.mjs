/**
 * 信令服务器的集成自检。
 *
 * 手写的 WebSocket（握手 / 掩码 / 帧解析）是这个工程里最容易出错的一块，
 * 所以单独跑一遍真实链路：起服务 → 两个客户端加入同一房间 → 断言
 * joined / peer-joined / signal 转发 / peer-left 四个环节都正确。
 *
 * 客户端用 Node 22 内置的全局 WebSocket，不是同一份代码，避免自证。
 *
 * 用法： npm run verify:signal
 */
import net from 'node:net';
import crypto from 'node:crypto';
import { startSignalServer } from './signal-server.mjs';

const PORT = Number( process.env.SIGNAL_TEST_PORT || 5199 );
const URL = `ws://127.0.0.1:${ PORT }`;

const results = [];
const check = ( name, pass, detail ) => results.push( { name, pass, detail } );

/** 带消息队列的客户端，便于顺序 await。 */
function makeClient ( label )
{
  const ws = new WebSocket( URL );
  const queue = [];
  const waiters = [];
  ws.addEventListener( 'message', ( ev ) =>
  {
    const msg = JSON.parse( String( ev.data ) );
    const w = waiters.shift();
    if ( w ) w( msg );
    else queue.push( msg );
  } );
  return {
    ws,
    label,
    open: () => new Promise( ( res, rej ) =>
    {
      ws.addEventListener( 'open', () => res(), { once: true } );
      ws.addEventListener( 'error', ( e ) => rej( new Error( `${ label } 连接失败：${ e.message ?? 'unknown' }` ) ), { once: true } );
    } ),
    next: ( timeoutMs = 4000 ) => new Promise( ( res, rej ) =>
    {
      if ( queue.length > 0 ) { res( queue.shift() ); return; }
      const timer = setTimeout( () => rej( new Error( `${ label } 等待消息超时` ) ), timeoutMs );
      waiters.push( ( m ) => { clearTimeout( timer ); res( m ); } );
    } ),
    send: ( obj ) => ws.send( JSON.stringify( obj ) ),
    close: () => ws.close(),
  };
}

const { server } = await startSignalServer( PORT );
const peerA = { peerId: 'aaaa1111', name: 'PC', role: 'host', device: { kind: 'Windows' }, joinedAt: 1 };
const peerB = { peerId: 'bbbb2222', name: 'Phone', role: 'peer', device: { kind: 'Android' }, joinedAt: 2 };

try
{
  const A = makeClient( 'A' );
  await A.open();
  A.send( { t: 'join', roomId: 'test-room', peer: peerA } );
  const aJoined = await A.next();
  check( '握手与 join 回执', aJoined.t === 'joined' && Array.isArray( aJoined.peers ) && aJoined.peers.length === 0,
    `A 收到 ${ aJoined.t }，房间内已有 ${ aJoined.peers?.length } 人` );

  const B = makeClient( 'B' );
  await B.open();
  B.send( { t: 'join', roomId: 'test-room', peer: peerB } );
  const bJoined = await B.next();
  check( '后加入者能看到已有成员', bJoined.t === 'joined' && bJoined.peers.length === 1 && bJoined.peers[ 0 ].peerId === peerA.peerId,
    `B 收到 joined，peers=[${ bJoined.peers.map( ( p ) => p.name ).join( ',' ) }]` );

  const aNotice = await A.next();
  check( '已有成员收到 peer-joined', aNotice.t === 'peer-joined' && aNotice.peer.peerId === peerB.peerId,
    `A 收到 ${ aNotice.t }：${ aNotice.peer?.name }` );

  B.send( { t: 'signal', to: peerA.peerId, data: { sdp: { type: 'offer', sdp: 'v=0-fake' } } } );
  const relayed = await A.next();
  check( 'signal 精确转发', relayed.t === 'signal' && relayed.from === peerB.peerId && relayed.data?.sdp?.sdp === 'v=0-fake',
    `A 收到来自 ${ relayed.from?.slice( 0, 8 ) } 的 ${ relayed.data?.sdp?.type }` );

  B.send( { t: 'signal', to: 'nosuchpeer', data: { ice: {} } } );
  const err = await B.next();
  check( '转发到不存在的节点会报错', err.t === 'error', `B 收到：${ err.message }` );

  // 大帧：验证 126/65536 两种长度分支（用一个长字符串撑过 64KB）
  const big = 'x'.repeat( 70000 );
  B.send( { t: 'signal', to: peerA.peerId, data: { sdp: { type: 'offer', sdp: big } } } );
  const bigRelay = await A.next( 8000 );
  check( '跨 64KB 分界的长帧可正常转发', bigRelay.data?.sdp?.sdp?.length === 70000,
    `回传长度 ${ bigRelay.data?.sdp?.sdp?.length }（期望 70000）` );

  B.close();
  const left = await A.next();
  check( '断开后广播 peer-left', left.t === 'peer-left' && left.peerId === peerB.peerId,
    `A 收到 ${ left.t }：${ left.peerId?.slice( 0, 8 ) }` );

  A.close();
}
catch ( err )
{
  check( '测试过程未抛异常', false, err.message );
}
finally
{
  server.close();
}

// ------------------------------------------------------------------ 心跳回归
//
// 2026-10-06 双端联调实测抓到的 P0 bug：pong 没有把连接的 alive 复位，
// 任何连接活不过两个心跳周期（25s×2）就被服务端单方面销毁。
// 症状极具欺骗性：训练数据走 WebRTC DataChannel，所以模型照训；
// 但 ~45s 后节点被静默踢出房间（刷新重连的节点还会发现"房主不在"）。
// 这里用**裸 socket** 客户端验证协议级 ping/pong —— 内置 WebSocket 会自动回
// pong 且把控制帧对 JS 隐藏，看不见这一层，必须自己读写帧。
const BEAT_PORT = Number( process.env.SIGNAL_TEST_PORT || 5199 ) + 1;
const BEAT_MS = 120;

function encodeClientFrame ( payload, opcode = 0x1 )
{
  const mask = crypto.randomBytes( 4 );
  const masked = Buffer.from( payload );
  for ( let i = 0; i < masked.length; i++ ) masked[ i ] ^= mask[ i % 4 ];
  const header = masked.length < 126
    ? ( () => { const h = Buffer.alloc( 2 ); h[ 1 ] = 0x80 | masked.length; return h; } )()
    : ( () => { const h = Buffer.alloc( 4 ); h[ 1 ] = 0x80 | 126; h.writeUInt16BE( masked.length, 2 ); return h; } )();
  header[ 0 ] = 0x80 | opcode;
  return Buffer.concat( [ header, mask, masked ] );
}

/** 裸 WebSocket 客户端：只握手 + join + 读帧；pong 行为由调用方决定。 */
function rawPeer ( label, { autoPong } )
{
  const sock = net.connect( BEAT_PORT, '127.0.0.1' );
  const key = crypto.randomBytes( 16 ).toString( 'base64' );
  const state = { label, pings: 0, texts: [], closed: false, ready: false };
  sock.on( 'connect', () =>
  {
    sock.write(
      'GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n' +
      `Sec-WebSocket-Key: ${ key }\r\nSec-WebSocket-Version: 13\r\n\r\n`,
    );
  } );
  let inbox = Buffer.alloc( 0 );
  let upgraded = false;
  sock.on( 'data', ( chunk ) =>
  {
    inbox = Buffer.concat( [ inbox, chunk ] );
    if ( !upgraded )
    {
      const idx = inbox.indexOf( '\r\n\r\n' );
      if ( idx < 0 ) return;
      if ( !inbox.subarray( 0, idx ).toString().includes( '101' ) ) { sock.destroy(); return; }
      upgraded = true;
      state.ready = true;
      inbox = inbox.subarray( idx + 4 );
      sock.write( encodeClientFrame( JSON.stringify( { t: 'join', roomId: 'beat-room', peer: { peerId: label, name: label, role: 'peer', device: { kind: 'raw' } } } ) ) );
    }
    for ( ;; )
    {
      if ( inbox.length < 2 ) return;
      const opcode = inbox[ 0 ] & 0x0f;
      let len = inbox[ 1 ] & 0x7f;
      let off = 2;
      if ( len === 126 ) { if ( inbox.length < 4 ) return; len = inbox.readUInt16BE( 2 ); off = 4; }
      else if ( len === 127 ) { if ( inbox.length < 10 ) return; len = Number( inbox.readBigUInt64BE( 2 ) ); off = 10; }
      if ( inbox.length < off + len ) return;
      const payload = inbox.subarray( off, off + len );
      inbox = inbox.subarray( off + len );
      if ( opcode === 0x9 )
      {
        state.pings += 1;
        if ( autoPong ) sock.write( encodeClientFrame( payload, 0xa ) );
      }
      else if ( opcode === 0x1 ) state.texts.push( payload.toString( 'utf8' ) );
      else if ( opcode === 0x8 ) { state.closed = true; sock.end(); }
    }
  } );
  sock.on( 'close', () => { state.closed = true; } );
  sock.on( 'error', () => { state.closed = true; } );
  return { sock, state };
}

const health = async () =>
  ( await fetch( `http://127.0.0.1:${ BEAT_PORT }/health` ).then( ( r ) => r.json() ) )
    .rooms.find( ( r ) => r.roomId === 'beat-room' )?.peers.map( ( p ) => p.peerId ) ?? [];

let beatServer = null;
try
{
  beatServer = await startSignalServer( BEAT_PORT, { beatMs: BEAT_MS } );
  const good = rawPeer( 'good-peer', { autoPong: true } );
  const bad = rawPeer( 'bad-peer', { autoPong: false } );
  await new Promise( ( r ) => setTimeout( r, 1200 ) ); // 10 个心跳周期

  const alive = await health();
  const goodSurvives = alive.includes( 'good-peer' );
  const badKicked = !alive.includes( 'bad-peer' );
  check( '心跳：按时回 pong 的连接可长期存活', goodSurvives && good.state.pings >= 5 && !good.state.closed,
    `good-peer 存活=${ goodSurvives }，收到 ${ good.state.pings } 个 ping，socket关闭=${ good.state.closed }（房间内：${ alive.join( ',' ) }）` );
  check( '心跳：不回 pong 的连接被服务端断开', badKicked && bad.state.closed,
    `bad-peer 被踢=${ badKicked }，收到 ${ bad.state.pings } 个 ping 后 socket关闭=${ bad.state.closed }` );

  good.sock.destroy();
  bad.sock.destroy();
}
catch ( err )
{
  check( '心跳回归测试未抛异常', false, err.message );
}
finally
{
  beatServer?.server?.close();
}

let failed = 0;
console.log( '\n公共训练网络 · 信令服务集成自检\n' );
for ( const r of results )
{
  const tag = r.pass ? '\u001b[32mPASS\u001b[0m' : '\u001b[31mFAIL\u001b[0m';
  console.log( `${ tag }  ${ r.name }` );
  console.log( `      ${ r.detail }` );
  if ( !r.pass ) failed += 1;
}
console.log( `\n${ results.length - failed }/${ results.length } 项通过` );
await new Promise( ( r ) => setTimeout( r, 150 ) );
process.exit( failed === 0 ? 0 : 1 );
