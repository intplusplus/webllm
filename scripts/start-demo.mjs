/**
 * 一键起「公共训练网络」demo：
 *   1) 在本进程内起信令服务器（零依赖）
 *   2) 拉起 vite dev server，监听 0.0.0.0（手机可通过局域网 IP 访问）
 *   3) 打印本机 / 局域网访问地址
 *
 * 用法： npm run demo
 */
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startSignalServer, lanAddresses } from './signal-server.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const SIGNAL_PORT = Number( process.env.SIGNAL_PORT || 5180 );
const VITE_PORT = Number( process.env.VITE_PORT || 5173 );

await startSignalServer( SIGNAL_PORT );

const ips = lanAddresses();
const line = '─'.repeat( 62 );
console.log( `\n${ line }` );
console.log( '  公共训练网络 · 联邦联训 Demo' );
console.log( line );
console.log( `  信令服务   ws://127.0.0.1:${ SIGNAL_PORT }   （健康检查 http://127.0.0.1:${ SIGNAL_PORT }/health）` );
console.log( `  本机打开   http://127.0.0.1:${ VITE_PORT }/fed.html` );
if ( ips.length === 0 )
{
  console.log( '  局域网     未检测到非回环 IPv4 地址（检查 WiFi / 网卡）' );
}
else
{
  console.log( '  同一 WiFi 下用手机打开：' );
  for ( const ip of ips ) console.log( `             http://${ ip }:${ VITE_PORT }/fed.html` );
  console.log( `  （信令地址会自动取当前页面的 host，端口 ${ SIGNAL_PORT }）` );
}
console.log( `\n  提示：本 demo 的默认引擎是纯 JS 小模型，不依赖 WebGPU，` );
console.log( '        因此局域网 http 也能跑。要切到 WebGPU 引擎需 HTTPS 或本机访问。' );
console.log( `${ line }\n` );

const viteBin = path.join( root, 'node_modules', 'vite', 'bin', 'vite.js' );
const child = spawn(
  process.execPath,
  [ viteBin, '--host', '0.0.0.0', '--port', String( VITE_PORT ), '--strictPort' ],
  { cwd: root, stdio: 'inherit' },
);
child.on( 'exit', ( code ) => process.exit( code ?? 0 ) );
for ( const sig of [ 'SIGINT', 'SIGTERM' ] )
{
  process.on( sig, () => { try { child.kill(); } catch { /* 忽略 */ } process.exit( 0 ); } );
}
