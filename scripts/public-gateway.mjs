/**
 * 公网发布网关：把「静态站点」和「信令服务器」并到一个 HTTP 端口上。
 *
 * 为什么需要它：托管平台（如 WorkBuddy「发布为应用」）只暴露一个 HTTP 端口，
 * 而联邦 demo 天然要两个 —— vite 静态产物（页面/模型/语料）+ 信令 WebSocket（5180）。
 * 网关把信令挂到 `/signal` 路径前缀下（HTTP 与 WS 都反代），静态部分直接读 dist/。
 *
 *   PORT=3000 node scripts/public-gateway.mjs
 *
 * 然后页面上信令地址填：`wss://<对外域名>/signal`
 * （也可用联邦页的 `?signal=wss://<域名>/signal` 查询参数直接带好。）
 *
 * 本机模式（同浏览器多标签页）不经过信令，任何部署形态下都可用。
 */
import fs from 'node:fs';
import http from 'node:http';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { startSignalServer } from './signal-server.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const DIST = path.join( root, 'dist' );
const PORT = Number( process.env.PORT || 3000 );
const SIGNAL_PORT = Number( process.env.SIGNAL_PORT || 5180 );
const PREFIX = '/signal';

/** 探测本机信令端口是否已在监听（避免重复拉起）。 */
function signalAlive ()
{
  return new Promise( ( resolve ) =>
  {
    const probe = net.connect( SIGNAL_PORT, '127.0.0.1' );
    probe.once( 'connect', () => { probe.destroy(); resolve( true ); } );
    probe.once( 'error', () => resolve( false ) );
  } );
}

// 信令没起就随网关一起拉起（托管平台只运行网关这一个入口）。
if ( !( await signalAlive() ) )
{
  await startSignalServer( SIGNAL_PORT );
  for ( let i = 0; i < 40 && !( await signalAlive() ); i++ )
    await new Promise( ( r ) => setTimeout( r, 250 ) );
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.crt': 'application/x-x509-ca-cert',
};

function proxyHttp ( req, res, target )
{
  const preq = http.request(
    {
      host: '127.0.0.1',
      port: SIGNAL_PORT,
      path: target,
      method: req.method,
      headers: { ...req.headers, host: `127.0.0.1:${ SIGNAL_PORT }` },
    },
    ( pres ) =>
    {
      res.writeHead( pres.statusCode ?? 502, pres.headers );
      pres.pipe( res );
    },
  );
  preq.on( 'error', ( err ) =>
  {
    res.writeHead( 502, { 'content-type': 'text/plain; charset=utf-8' } );
    res.end( `信令服务不可达：${ err.message }` );
  } );
  req.pipe( preq );
}

/** WS 反代：改写起始行/Host 后做裸 TCP 双向隧道（信令服务器无 Origin 校验）。 */
function proxyUpgrade ( req, sock, head, target )
{
  const upstream = net.connect( SIGNAL_PORT, '127.0.0.1', () =>
  {
    const lines = [ `${ req.method } ${ target } HTTP/1.1` ];
    for ( const [ k, v ] of Object.entries( req.headers ) )
    {
      if ( v === undefined ) continue;
      const value = Array.isArray( v ) ? v.join( ', ' ) : v;
      if ( k === 'host' ) lines.push( `Host: 127.0.0.1:${ SIGNAL_PORT }` );
      else lines.push( `${ k }: ${ value }` );
    }
    upstream.write( lines.join( '\r\n' ) + '\r\n\r\n' );
    if ( head?.length ) upstream.write( head );
    sock.pipe( upstream );
    upstream.pipe( sock );
  } );
  upstream.on( 'error', () => sock.destroy() );
  sock.on( 'error', () => upstream.destroy() );
}

function serveStatic ( res, pathname )
{
  let fp = path.normalize( path.join( DIST, decodeURIComponent( pathname ) ) );
  if ( !fp.startsWith( DIST ) )
  {
    res.writeHead( 403 );
    res.end( 'forbidden' );
    return;
  }
  fs.stat( fp, ( statErr, st ) =>
  {
    if ( !statErr && st.isDirectory() ) fp = path.join( fp, 'index.html' );
    fs.readFile( fp, ( readErr, buf ) =>
    {
      if ( readErr )
      {
        res.writeHead( 404, { 'content-type': 'text/plain; charset=utf-8' } );
        res.end( 'not found' );
        return;
      }
      res.writeHead( 200, {
        'content-type': MIME[ path.extname( fp ).toLowerCase() ] ?? 'application/octet-stream',
        'cache-control': 'no-cache',
      } );
      res.end( buf );
    } );
  } );
}

const server = http.createServer( ( req, res ) =>
{
  const pathname = new URL( req.url ?? '/', 'http://localhost' ).pathname;

  if ( pathname === '/health' )
  {
    res.writeHead( 200, { 'content-type': 'text/plain; charset=utf-8' } );
    res.end( 'ok' );
    return;
  }
  // 站点地图：/ 直达单机训练台
  if ( pathname === '/' || pathname === '' )
  {
    res.writeHead( 302, { location: '/pages/index.html' } );
    res.end();
    return;
  }
  // 信令反代：/signal/* → 127.0.0.1:SIGNAL_PORT/*
  if ( pathname === PREFIX || pathname.startsWith( PREFIX + '/' ) )
  {
    proxyHttp( req, res, pathname.slice( PREFIX.length ) || '/' );
    return;
  }
  serveStatic( res, pathname );
} );

server.on( 'upgrade', ( req, sock, head ) =>
{
  const pathname = new URL( req.url ?? '/', 'http://localhost' ).pathname;
  if ( pathname !== PREFIX && !pathname.startsWith( PREFIX + '/' ) )
  {
    sock.destroy();
    return;
  }
  proxyUpgrade( req, sock, head, pathname.slice( PREFIX.length ) || '/' );
} );

server.listen( PORT, '0.0.0.0', () =>
{
  console.log( `[public-gateway] http://0.0.0.0:${ PORT }  (静态=${ DIST }，信令反代=${ PREFIX } → :${ SIGNAL_PORT })` );
} );
