/**
 * 一键起「公共训练网络」demo。
 *
 *   node scripts/start-demo.mjs            普通 http（CPU 引擎够用，手机随便连）
 *   node scripts/start-demo.mjs --https    本地自签 HTTPS —— WebGPU 需要安全上下文
 *
 * 做三件事：
 *   1) 在本进程内起信令服务器（零依赖）
 *   2) 起 vite dev server（监听 0.0.0.0，手机可访问）
 *   3) 打印手机该怎么连、以及最常见的两个坑（非安全上下文 / 防火墙）
 *
 * --https 模式下额外做一件：用一个普通 http 小服务把本地 CA 证书发出去，
 * 手机先下这个证书装上，之后访问 https 站点就不会被拦。
 */
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startSignalServer, lanAddresses } from './signal-server.mjs';
import { defaultCertDir, ensureCerts } from './lib/cert.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const SIGNAL_PORT = Number( process.env.SIGNAL_PORT || 5180 );
const VITE_PORT = Number( process.env.VITE_PORT || 5173 );
const CA_PORT = Number( process.env.CA_PORT || 5190 );
const useHttps = process.argv.includes( '--https' );

await startSignalServer( SIGNAL_PORT );

const ips = lanAddresses().map( ( ip ) => ip.replace( /^::ffff:/, '' ) );
const scheme = useHttps ? 'https' : 'http';
const wsScheme = useHttps ? 'wss' : 'ws';
const line = '─'.repeat( 70 );

// ---- 证书（仅 https 模式）----
let caInfo = null;
if ( useHttps )
{
  try
  {
    caInfo = ensureCerts( { dir: defaultCertDir( root ), ips } );
    // 手机得先能下载到这个 CA。用普通 http 起一个小服务专门发它。
    http.createServer( ( req, res ) =>
    {
      if ( req.url === '/ca.crt' || req.url === '/' )
      {
        const buf = fs.readFileSync( caInfo.ca );
        res.writeHead( 200, {
          'content-type': 'application/x-x509-ca-cert',
          'content-disposition': 'attachment; filename="webllm-ca.crt"',
        } );
        res.end( buf );
        return;
      }
      res.writeHead( 404 );
      res.end( 'not found' );
    } ).listen( CA_PORT, '0.0.0.0' );
  }
  catch ( err )
  {
    console.error( `\n生成 HTTPS 证书失败：${ err.message}\n` );
    console.error( '可以改用普通 http 模式，然后用 chrome://flags 把站点加入安全来源。\n' );
    process.exit( 1 );
  }
}

console.log( `\n${ line }` );
console.log( `  公共训练网络 · 联邦联训 Demo   （${ useHttps ? 'HTTPS' : 'HTTP'}）` );
console.log( line );
console.log( `  信令服务   ${ wsScheme }://127.0.0.1:${ SIGNAL_PORT }   （健康检查 http://127.0.0.1:${ SIGNAL_PORT }/health）` );
console.log( `  本机打开   ${ scheme }://127.0.0.1:${ VITE_PORT }/pages/fed.html` );

if ( ips.length === 0 )
{
  console.log( '  局域网     未检测到非回环 IPv4 地址 —— 先确认电脑已连上 WiFi' );
}
else
{
  console.log( `\n  ★ 手机在同一个 WiFi 下打开这个地址（建议直接抄进手机浏览器）：\n` );
  for ( const ip of ips ) console.log( `        ${ scheme }://${ ip }:${ VITE_PORT }/pages/fed.html` );
}

console.log( `\n  三步跑起来：` );
console.log( `    1. 电脑打开上面的「本机打开」地址。先看「设备能力」那块：` );
console.log( `       如果显示「安全上下文 ✓、适配器 ✓」，就能用 WebGPU 引擎。` );
console.log( `    2. 手机打开局域网地址，核对「设备能力」；两边一致后，房间 ID 保持默认。` );
console.log( `    3. 电脑点「创建训练房间（主机）」→ 手机点「加入房间（节点）」→ 电脑点「开始训练」。` );

if ( useHttps )
{
  console.log( `\n  手机第一次访问 https 会被拦，按这个顺序做一次即可：` );
  console.log( `    ① 手机浏览器打开  http://${ ips[ 0 ] ?? '<电脑IP>' }:${ CA_PORT }/ca.crt` );
  console.log( '       下载后：设置 → 安全 → 加密与凭据 → 安装证书 → CA 证书 → 选刚下的文件' );
  console.log( `    ② 然后打开  https://${ ips[ 0 ] ?? '<电脑IP>' }:${ VITE_PORT }/pages/fed.html` );
  console.log( `  （本地 CA 放在 ${ path.relative( root, caInfo.dir ) }，重装系统或换 IP 会自动重签）` );
}
else
{
  console.log( `\n  想用 WebGPU 但手机 VITE 拿不到？http 不是安全上下文，浏览器会禁掉 WebGPU。两条路：` );
  console.log( `    ① 最快：手机 Chrome 打开 chrome://flags/#unsafely-treat-insecure-origin-as-secure` );
  console.log( `       把  http://${ ips[ 0 ] ?? '<电脑IP>' }:${ VITE_PORT }  填进去，选 Enabled，重启浏览器。` );
  console.log( '    ② 更正规：改用  npm run demo:https  起 https（需要手机装一次本地 CA）' );
  console.log( '  不想折腾也行：把「训练引擎」选「只用 CPU」，手机照样能参与联合训练。' );
}

console.log( `\n  手机连不上？九成是 Windows 防火墙拦了入站。管理员 PowerShell 跑一次：` );
console.log( `    netsh advfirewall firewall add rule name="webllm demo" dir=in action=allow protocol=TCP localport=${ VITE_PORT },${ SIGNAL_PORT },${ CA_PORT }` );
console.log( `${ line }\n` );

// ---- 起 vite（用 API 而不是 CLI，这样才能塞自签证书）----
const { createServer } = await import( 'vite' );
const httpsOpts = useHttps
  ? { key: fs.readFileSync( caInfo.key ), cert: fs.readFileSync( caInfo.cert ) }
  : undefined;

const vite = await createServer( {
  root,
  server: {
    host: '0.0.0.0',
    port: VITE_PORT,
    strictPort: true,
    ...( httpsOpts ? { https: httpsOpts } : {} ),
  },
} );
await vite.listen();

for ( const sig of [ 'SIGINT', 'SIGTERM' ] )
{
  process.on( sig, () =>
  {
    void vite.close().finally( () => process.exit( 0 ) );
  } );
}
