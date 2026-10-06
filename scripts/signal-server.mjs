/**
 * 公共训练网络 · 信令服务器（零依赖）。
 *
 * 只做一件事：帮同一房间里的浏览器交换 SDP / ICE，把 P2P 通道牵起来。
 * 一旦牵线完成，权重、梯度、训练日志全部在浏览器之间直连，不经过本服务。
 *
 * 刻意不引入 `ws` 之类的依赖 —— 契约很小（RFC 6455 握手 + 文本帧），
 * 用 node 内置的 http + crypto 手写反而更可审计、更适合「公共产品」这个定位：
 * 任何人都能读完全部代码，也就更容易自建与被信任。
 *
 * 线路协议（JSON 文本帧）：
 *   C→S  { t:'join',   roomId, peer:{peerId,name,role,device} }
 *   S→C  { t:'joined', peers:[peer,...] }
 *   S→C  { t:'peer-joined', peer }        （向房间内其他人广播）
 *   C→S  { t:'signal', to, data }         data = {sdp}|{ice}
 *   S→C  { t:'signal', from, data }
 *   C→S  { t:'leave' }
 *   S→C  { t:'peer-left', peerId }
 *   S→C  { t:'error', message }
 */
import http from 'node:http';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { networkInterfaces } from 'node:os';

const GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
const DEFAULT_PORT = 5180;
const MAX_FRAME = 1 << 20; // 1 MiB：信令帧远小于此，超限直接断开

// ------------------------------------------------------------------ WebSocket 编解码

function encodeFrame ( str, opcode = 0x1 )
{
  const payload = Buffer.from( str, 'utf8' );
  const len = payload.length;
  let header;
  if ( len < 126 )
  {
    header = Buffer.alloc( 2 );
    header[ 1 ] = len;
  }
  else if ( len < 65536 )
  {
    header = Buffer.alloc( 4 );
    header[ 1 ] = 126;
    header.writeUInt16BE( len, 2 );
  }
  else
  {
    header = Buffer.alloc( 10 );
    header[ 1 ] = 127;
    header.writeBigUInt64BE( BigInt( len ), 2 );
  }
  header[ 0 ] = 0x80 | opcode;
  return Buffer.concat( [ header, payload ] );
}

/** 从 socket 缓冲区里尽量多解出完整帧；返回剩余缓冲。 */
function decodeFrames ( buf, onText, onClose, onPing )
{
  let rest = buf;
  for ( ;; )
  {
    if ( rest.length < 2 ) return rest;
    const fin = ( rest[ 0 ] & 0x80 ) !== 0;
    const opcode = rest[ 0 ] & 0x0f;
    const masked = ( rest[ 1 ] & 0x80 ) !== 0;
    let len = rest[ 1 ] & 0x7f;
    let off = 2;
    if ( len === 126 )
    {
      if ( rest.length < 4 ) return rest;
      len = rest.readUInt16BE( 2 );
      off = 4;
    }
    else if ( len === 127 )
    {
      if ( rest.length < 10 ) return rest;
      len = Number( rest.readBigUInt64BE( 2 ) );
      off = 10;
    }
    if ( len > MAX_FRAME ) throw new Error( `帧过大：${ len }` );
    let mask = null;
    if ( masked )
    {
      if ( rest.length < off + 4 ) return rest;
      mask = rest.subarray( off, off + 4 );
      off += 4;
    }
    if ( rest.length < off + len ) return rest;
    const payload = Buffer.from( rest.subarray( off, off + len ) );
    if ( mask ) for ( let i = 0; i < payload.length; i++ ) payload[ i ] ^= mask[ i % 4 ];
    rest = rest.subarray( off + len );

    if ( opcode === 0x8 ) { onClose(); return Buffer.alloc( 0 ); }
    if ( opcode === 0x9 ) { onPing( payload ); continue; }
    if ( opcode === 0xa ) continue; // pong
    if ( opcode === 0x1 || opcode === 0x0 )
    {
      onText( payload.toString( 'utf8' ), fin );
    }
  }
}

// ------------------------------------------------------------------ 房间状态

/** roomId → Map(peerId → { peer, sock, alive, frag }) */
const rooms = new Map();

function roomOf ( roomId, create = false )
{
  let r = rooms.get( roomId );
  if ( !r && create )
  {
    r = new Map();
    rooms.set( roomId, r );
  }
  return r;
}

function peersOf ( roomId, exceptPeerId )
{
  const r = rooms.get( roomId );
  if ( !r ) return [];
  const out = [];
  for ( const [ id, entry ] of r ) if ( id !== exceptPeerId ) out.push( entry.peer );
  return out;
}

function send ( sock, obj )
{
  if ( sock.destroyed ) return;
  try { sock.write( encodeFrame( JSON.stringify( obj ) ) ); } catch { /* 忽略 */ }
}

function broadcast ( roomId, obj, exceptPeerId )
{
  const r = rooms.get( roomId );
  if ( !r ) return;
  for ( const [ id, entry ] of r ) if ( id !== exceptPeerId ) send( entry.sock, obj );
}

function leaveRoom ( roomId, peerId, quiet = false )
{
  const r = rooms.get( roomId );
  if ( !r ) return;
  const entry = r.get( peerId );
  if ( !entry ) return;
  r.delete( peerId );
  if ( r.size === 0 ) rooms.delete( roomId );
  if ( !quiet )
  {
    console.log( `  ← ${ peerId.slice( 0, 8 ) } 离开 ${ roomId }（房间余 ${ roomOf( roomId )?.size ?? 0 } 人）` );
    broadcast( roomId, { t: 'peer-left', peerId } );
  }
}

// ------------------------------------------------------------------ 成员表（只读观测）

function stats ()
{
  const list = [];
  for ( const [ roomId, m ] of rooms )
  {
    list.push( { roomId, peers: [ ...m.values() ].map( ( e ) => ( { peerId: e.peer.peerId, name: e.peer.name, role: e.peer.role, kind: e.peer.device?.kind } ) ) } );
  }
  return list;
}

// ------------------------------------------------------------------ 服务

export function startSignalServer ( port = DEFAULT_PORT )
{
  const server = http.createServer( ( req, res ) =>
  {
    if ( req.url === '/health' )
    {
      res.writeHead( 200, { 'content-type': 'application/json; charset=utf-8' } );
      res.end( JSON.stringify( { ok: true, rooms: stats() } ) );
      return;
    }
    res.writeHead( 200, { 'content-type': 'text/plain; charset=utf-8' } );
    res.end( '公共训练网络 · 信令服务器在运行。用 WebSocket 连我：ws://<host>:' + port + '\n' );
  } );

  server.on( 'upgrade', ( req, sock ) =>
  {
    const key = req.headers[ 'sec-websocket-key' ];
    if ( !key )
    {
      sock.destroy();
      return;
    }
    const accept = crypto.createHash( 'sha1' ).update( key + GUID ).digest( 'base64' );
    sock.write(
      'HTTP/1.1 101 Switching Protocols\r\n' +
      'Upgrade: websocket\r\n' +
      'Connection: Upgrade\r\n' +
      `Sec-WebSocket-Accept: ${ accept }\r\n\r\n`,
    );
    sock.setNoDelay( true );

    let roomId = null;
    let peerId = null;
    let frag = '';
    /** 跨 TCP 分包的粘包缓冲 */
    let inbox = Buffer.alloc( 0 );

    const cleanup = () =>
    {
      if ( roomId && peerId ) leaveRoom( roomId, peerId );
      try { sock.destroy(); } catch { /* 忽略 */ }
    };

    sock.on( 'data', ( chunk ) =>
    {
      inbox = Buffer.concat( [ inbox, chunk ] );
      try
      {
        inbox = decodeFrames(
          inbox,
          ( text, fin ) =>
          {
            frag += text;
            if ( !fin ) return;
            const raw = frag;
            frag = '';
            let msg;
            try { msg = JSON.parse( raw ); }
            catch { send( sock, { t: 'error', message: '信令帧不是合法 JSON' } ); return; }

            if ( msg.t === 'join' )
            {
              roomId = String( msg.roomId );
              peerId = String( msg.peer.peerId );
              const r = roomOf( roomId, true );
              r.set( peerId, { peer: msg.peer, sock, alive: true } );
              const existing = peersOf( roomId, peerId );
              send( sock, { t: 'joined', peers: existing } );
              broadcast( roomId, { t: 'peer-joined', peer: msg.peer }, peerId );
              console.log( `  → ${ peerId.slice( 0, 8 ) }（${ msg.peer.role }，${ msg.peer.device?.kind ?? '?' }）加入 ${ roomId }，房间现有 ${ r.size } 人` );
              return;
            }

            if ( msg.t === 'signal' )
            {
              const r = rooms.get( roomId );
              const target = r?.get( String( msg.to ) );
              if ( !target ) { send( sock, { t: 'error', message: `目标节点不在房间：${ msg.to }` } ); return; }
              send( target.sock, { t: 'signal', from: peerId, data: msg.data } );
              return;
            }

            if ( msg.t === 'leave' ) { cleanup(); }
          },
          () => cleanup(),
          ( payload ) => { try { sock.write( encodeFrame( payload.toString( 'utf8' ), 0xa ) ); } catch { /* 忽略 */ } },
        );
      }
      catch ( err )
      {
        console.error( '  信令解析异常：', err.message );
        cleanup();
      }
    } );

    sock.on( 'close', cleanup );
    sock.on( 'error', cleanup );
  } );

  // 心跳：25s 未收到 pong 就断开，避免死连接占位
  const beat = setInterval( () =>
  {
    for ( const m of rooms.values() )
    {
      for ( const entry of m.values() )
      {
        if ( !entry.alive )
        {
          try { entry.sock.destroy(); } catch { /* 忽略 */ }
          continue;
        }
        entry.alive = false;
        try { entry.sock.write( encodeFrame( 'ping', 0x9 ) ); } catch { /* 忽略 */ }
      }
    }
  }, 25000 );
  beat.unref?.();

  return new Promise( ( resolve ) =>
  {
    server.listen( port, '0.0.0.0', () => resolve( { server, port } ) );
  } );
}

/** 列出本机所有非回环 IPv4 地址，便于手机用同一个 WiFi 访问。 */
export function lanAddresses ()
{
  const out = [];
  for ( const list of Object.values( networkInterfaces() ) )
  {
    for ( const i of list ?? [] )
    {
      if ( i.family === 'IPv4' && !i.internal ) out.push( i.address );
    }
  }
  return out;
}

const invokedDirectly = !!process.argv[ 1 ] &&
  path.resolve( process.argv[ 1 ] ) === fileURLToPath( import.meta.url );
if ( invokedDirectly )
{
  const port = Number( process.env.SIGNAL_PORT || DEFAULT_PORT );
  await startSignalServer( port );
  console.log( `公共训练网络 · 信令服务器已启动  ws://0.0.0.0:${ port }` );
  console.log( `健康检查： http://127.0.0.1:${ port }/health` );
  for ( const ip of lanAddresses() ) console.log( `局域网：   ws://${ ip }:${ port }` );
}
