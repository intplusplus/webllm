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
const line = '─'.repeat( 66 );
console.log( `\n${ line }` );
console.log( '  公共训练网络 · 联邦联训 Demo' );
console.log( line );
console.log( `  信令服务   ws://127.0.0.1:${ SIGNAL_PORT }   （健康检查 http://127.0.0.1:${ SIGNAL_PORT }/health）` );
console.log( `  本机打开   http://127.0.0.1:${ VITE_PORT }/fed.html` );

if ( ips.length === 0 )
{
  console.log( '  局域网     未检测到非回环 IPv4 地址 —— 先确认已连上 WiFi' );
}
else
{
  console.log( `\n  ★ 手机在同一个 WiFi 下打开这个地址（建议直接抄进手机浏览器）：\n` );
  for ( const ip of ips ) console.log( `        http://${ ip }:${ VITE_PORT }/fed.html` );
  console.log( `\n    （页面里的「信令地址」会自动指向当前页面的 host，不用手填）` );
}

console.log( `\n  三步跑起来：` );
console.log( `    1. 电脑上打开上面的「本机打开」地址，房间 ID 保持默认，点「创建训练房间（主机）」` );
console.log( `    2. 手机打开上面的局域网地址，房间 ID 保持默认（两边必须一致），点「加入房间（节点）」` );
console.log( `    3. 回到电脑，看到手机出现在节点表里后，点「开始训练」` );
console.log( `\n    只想在本机试：把「联机方式」切到「本机多标签页」，开两个标签页即可，不需要信令服务。` );

console.log( `\n  手机连不上？九成是 Windows 防火墙拦了入站。管理员 PowerShell 跑一次：` );
console.log( `    netsh advfirewall firewall add rule name="webllm demo" dir=in action=allow protocol=TCP localport=${ VITE_PORT },${ SIGNAL_PORT }` );

console.log( `\n  说明：默认引擎是纯 JS 小模型，不依赖 WebGPU，所以局域网 http 也能训练。` );
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
