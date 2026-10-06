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
