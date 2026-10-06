/**
 * 真实浏览器里的界面端到端测试（零依赖，直连 CDP）。
 *
 * 动机：前面所有自检都是无头的 —— 核心逻辑验证得很扎实，但**界面从来没有在
 * 真实浏览器里被点过**。而用户的诉求恰恰是「手机和 PC 能同时训练」，
 * 那件事只有在浏览器里才算数。
 *
 * 本机 agent-browser 的 `open` 会永久挂起，所以这里绕开它：
 * 直接 spawn 本机 Chromium（--headless=new --remote-debugging-port），
 * 用 Node 内置的 WebSocket 说 CDP 协议。不引入 playwright / puppeteer。
 *
 * 两个场景各开两个标签页，跑完整流程：
 *   场景 A：本机多标签页（BroadcastChannel，免服务器）
 *   场景 B：WebRTC + 信令（跨设备走的就是这条路径，用本机回环验证）
 *
 * 用法： npm run verify:ui
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { startSignalServer } from './signal-server.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const SHOT_DIR = path.join( root, 'demo', 'screenshots' );
fs.mkdirSync( SHOT_DIR, { recursive: true } );
const VITE_PORT = 5175;
const SIGNAL_PORT = 5181;
const CDP_PORT = 9333;

const results = [];
const check = ( name, pass, detail ) => results.push( { name, pass, detail } );

function findChrome ()
{
  if ( process.env.CHROME_PATH ) return process.env.CHROME_PATH;
  const base = path.join( process.env.LOCALAPPDATA || '', 'ms-playwright' );
  if ( !fs.existsSync( base ) ) return null;
  const dirs = fs.readdirSync( base )
    .filter( ( d ) => d.startsWith( 'chromium-' ) )
    .sort()
    .reverse(); // 优先新版
  for ( const d of dirs )
  {
    const exe = path.join( base, d, 'chrome-win64', 'chrome.exe' );
    if ( fs.existsSync( exe ) ) return exe;
  }
  return null;
}

// ------------------------------------------------------------------ 极简 CDP 客户端

async function connectCdp ( wsUrl )
{
  const ws = new WebSocket( wsUrl );
  const pending = new Map();
  let nextId = 1;

  ws.addEventListener( 'message', ( ev ) =>
  {
    let msg;
    try { msg = JSON.parse( String( ev.data ) ); } catch { return; }
    const slot = pending.get( msg.id );
    if ( !slot ) return;
    pending.delete( msg.id );
    if ( msg.error ) slot.reject( new Error( msg.error.message ) );
    else slot.resolve( msg.result );
  } );

  await new Promise( ( resolve, reject ) =>
  {
    ws.addEventListener( 'open', () => resolve(), { once: true } );
    ws.addEventListener( 'error', () => reject( new Error( 'CDP 连接失败：' + wsUrl ) ), { once: true } );
  } );

  return {
    send ( method, params = {} )
    {
      const id = nextId++;
      return new Promise( ( resolve, reject ) =>
      {
        const timer = setTimeout( () =>
        {
          if ( pending.delete( id ) ) reject( new Error( `CDP 调用超时：${ method }` ) );
        }, 30000 );
        pending.set( id, {
          resolve: ( r ) => { clearTimeout( timer ); resolve( r ); },
          reject: ( e ) => { clearTimeout( timer ); reject( e ); },
        } );
        ws.send( JSON.stringify( { id, method, params } ) );
      } );
    },
    close () { try { ws.close(); } catch { /* 忽略 */ } },
  };
}

async function newPage ( cdpPort )
{
  const res = await fetch( `http://127.0.0.1:${ cdpPort }/json/new?url=about:blank`, { method: 'PUT' } );
  if ( !res.ok ) throw new Error( `创建标签页失败：HTTP ${ res.status }` );
  const info = await res.json();
  const cdp = await connectCdp( info.webSocketDebuggerUrl );
  await cdp.send( 'Page.enable' );
  await cdp.send( 'Runtime.enable' );
  return cdp;
}

async function evaluate ( cdp, expression, timeoutMs = 30000 )
{
  const r = await cdp.send( 'Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  } );
  if ( r.exceptionDetails )
  {
    const d = r.exceptionDetails.exception?.description ?? r.exceptionDetails.text;
    throw new Error( `页面内抛出异常：${ d }` );
  }
  void timeoutMs;
  return r.result.value;
}

async function waitFor ( cdp, predicateExpr, ms, label )
{
  const deadline = Date.now() + ms;
  let last = '';
  while ( Date.now() < deadline )
  {
    try
    {
      const ok = await evaluate( cdp, predicateExpr );
      if ( ok ) return true;
      last = String( await evaluate( cdp, 'document.body.innerText.slice(0,400)' ) );
    }
    catch ( err )
    {
      last = 'evaluate 失败：' + err.message;
    }
    await delay( 300 );
  }
  throw new Error( `等待超时（${ label }）。当前页面文本：${ last.replace( /\s+/g, ' ' ) }` );
}

async function clickButton ( cdp, text )
{
  const ok = await evaluate( cdp, `(() => {
    const b = [ ...document.querySelectorAll( 'button' ) ].find( ( x ) => x.textContent.includes( ${ JSON.stringify( text ) } ) );
    if ( !b ) return false;
    if ( b.disabled ) return 'disabled';
    b.click();
    return true;
  })()` );
  if ( ok !== true ) throw new Error( `按钮「${ text }」不可点击（${ ok }）` );
}

async function screenshot ( cdp, cdpPort, file )
{
  const t = await fetch( `http://127.0.0.1:${ cdpPort }/json/list` ).then( ( r ) => r.json() );
  void t;
  const r = await cdp.send( 'Page.captureScreenshot', { format: 'png', captureBeyondViewport: true } );
  fs.mkdirSync( path.dirname( file ), { recursive: true } );
  fs.writeFileSync( file, Buffer.from( r.data, 'base64' ) );
}

const delay = ( ms ) => new Promise( ( r ) => setTimeout( r, ms ) );

/**
 * 判断状态栏文案。
 * 注意：不能用 body.innerText 里搜「训练完成」—— 成果卡片有一句静态占位文案
 * 「训练完成后，可以下载模型卡…」，会让等待瞬间“通过”，断言于是在训练还没跑完
 * 时就执行了（这个坑已经踩过一次）。只认 .fed-status 这个真状态元素。
 */
const statusHas = ( text ) =>
  `(() => { const e = document.querySelector( '.fed-status' ); return !!e && e.textContent.includes( ${ JSON.stringify( text ) } ); })()`;

// ------------------------------------------------------------------ 一个场景

async function runScenario ( { name, query, base, cdpPort, shotPrefix } )
{
  const url = `${ base }/fed.html?${ query }`;
  const host = await newPage( cdpPort );
  const peer = await newPage( cdpPort );
  const dump = async ( tag, cdp ) =>
  {
    try
    {
      const text = await evaluate( cdp, 'document.body.innerText' );
      fs.writeFileSync( path.join( SHOT_DIR, `${ shotPrefix }-${ tag }.txt` ), text, 'utf8' );
    }
    catch { /* 忽略 */ }
  };

  try
  {
    await host.send( 'Page.navigate', { url } );
    await peer.send( 'Page.navigate', { url } );

    await waitFor( host, 'document.readyState === "complete"', 20000, `${ name } 主机页面加载` );
    await waitFor( peer, 'document.readyState === "complete"', 20000, `${ name } 节点页面加载` );

    // 语料加载完成后两个按钮才可用
    await waitFor( host, '![...document.querySelectorAll("button")].find(b=>b.textContent.includes("创建训练房间")).disabled', 30000, `${ name } 主机按钮就绪` );
    await waitFor( peer, '![...document.querySelectorAll("button")].find(b=>b.textContent.includes("加入房间")).disabled', 30000, `${ name } 节点按钮就绪` );

    await clickButton( host, '创建训练房间' );
    await waitFor( host, statusHas( '房间已创建' ), 30000, `${ name } 主机建房` );

    await clickButton( peer, '加入房间' );
    // 节点拿到清单的判据：任务卡片里出现了「清单指纹」，且占位文案消失
    await waitFor(
      peer,
      `(() => { const t = document.body.innerText; return t.includes( '清单指纹' ) && !t.includes( '尚未加入任何房间' ); })()`,
      40000,
      `${ name } 节点收到清单`,
    );

    // 主机节点表里应该出现两条「通道就绪」（自己 + 对端）
    const rosterSeen = await waitFor(
      host,
      `((document.body.innerText.match( /通道就绪/g ) || []).length >= 2)`,
      30000,
      `${ name } 主机看到节点`,
    ).then( () => true ).catch( () => false );

    await clickButton( host, '开始训练' );

    await waitFor( host, statusHas( '训练完成' ), 180000, `${ name } 主机训练完成` );
    const peerDone = await waitFor( peer, statusHas( '训练完成' ), 60000, `${ name } 节点收到完成` )
      .then( () => true ).catch( () => false );

    const hostText = await evaluate( host, 'document.body.innerText' );
    const peerText = await evaluate( peer, 'document.body.innerText' );
    const hostStatus = String( await evaluate( host, `(document.querySelector( '.fed-status' ) || {}).textContent || ''` ) );
    const peerStatus = String( await evaluate( peer, `(document.querySelector( '.fed-status' ) || {}).textContent || ''` ) );

    // --- 断言 ---
    check( `[${ name }] 两个标签页都加载出界面`, hostText.includes( '连接房间' ) && peerText.includes( '连接房间' ),
      `主机文本 ${ hostText.length } 字符，节点 ${ peerText.length } 字符` );

    check( `[${ name }] 主机创建房间并看到节点`, rosterSeen,
      `主机节点表可见=${ rosterSeen }` );

    check( `[${ name }] 主机完成训练并生成模型卡`,
      hostStatus.includes( '训练完成' ) && hostText.includes( '传输总量' ) && hostText.includes( '探针 loss' ),
      `状态栏「${ hostStatus.slice( 0, 60 ) }」，含「传输总量」=${ hostText.includes( '传输总量' ) }` );

    check( `[${ name }] 节点侧也收到完成并生成模型卡`,
      peerDone && peerStatus.includes( '训练完成' ) && peerText.includes( '传输总量' ),
      `节点等待到完成=${ peerDone }，状态栏「${ peerStatus.slice( 0, 60 ) }」，含「传输总量」=${ peerText.includes( '传输总量' ) }` );

    const ledgerOk = /通过/.test( hostText ) && /第 \d+ 轮 · 全局探针 loss/.test( hostText );
    check( `[${ name }] 贡献账本渲染出逐轮裁决`, ledgerOk,
      `含「通过」=${ /通过/.test( hostText ) }，含逐轮汇总=${ /第 \d+ 轮 · 全局探针 loss/.test( hostText ) }` );

    const lossDrop = hostText.includes( '最终' ) && hostText.includes( '初始' );
    check( `[${ name }] 模型卡包含 loss 下降记录`, lossDrop, `匹配到「初始/最终」结构=${ lossDrop }` );

    // 续写：验证推理路径在浏览器里可用
    await evaluate( host, `(() => {
      const i = [ ...document.querySelectorAll( 'input' ) ].find( ( x ) => x.placeholder && x.placeholder.includes( '前缀' ) );
      if ( !i ) return false;
      i.value = 'The ';
      return true;
    })()` );
    await clickButton( host, '续写' );
    await delay( 2500 );
    // 取最后一个 pre.fed-log：训练日志也在用这个类，续写结果是后插入的那个
    const genText = await evaluate( host, `(() => {
      const all = [ ...document.querySelectorAll( 'pre.fed-log' ) ];
      return all.length ? all[ all.length - 1 ].textContent : '';
    })()` );
    check( `[${ name }] 训练出的模型能在页面上续写`, typeof genText === 'string' && genText.length > 40 && !genText.includes( '失败' ),
      `续写输出 ${ String( genText ).length } 字符：${ String( genText ).replace( /\s+/g, ' ' ).slice( 0, 70 ) }` );
    fs.writeFileSync( path.join( SHOT_DIR, `${ shotPrefix }-generate.txt` ), String( genText ), 'utf8' );
  }
  finally
  {
    try { await screenshot( host, cdpPort, path.join( SHOT_DIR, `${ shotPrefix }-host.png` ) ); } catch { /* 忽略 */ }
    try { await screenshot( peer, cdpPort, path.join( SHOT_DIR, `${ shotPrefix }-peer.png` ) ); } catch { /* 忽略 */ }
    try { await dump( 'host', host ); } catch { /* 忽略 */ }
    try { await dump( 'peer', peer ); } catch { /* 忽略 */ }
    host.close();
    peer.close();
  }
}

// ------------------------------------------------------------------ 主流程

const chrome = findChrome();
if ( !chrome )
{
  console.log( '未找到 Chromium（ms-playwright 缓存里没有）。跳过界面测试。' );
  process.exit( 2 );
}

console.log( `公共训练网络 · 界面端到端测试（真实 Chromium）\n` );
console.log( `浏览器：${ chrome }\n` );

// Chromium 需要一个独立的用户数据目录。每次用新的临时目录，
// 且**不在脚本里删除它** —— 删一个大目录会撞上宿主的安全删除守卫，
// 交给操作系统回收临时目录即可。
const userDataDir = fs.mkdtempSync( path.join( os.tmpdir(), 'wb-ui-chrome-' ) );
console.log( `浏览器 profile：${ userDataDir }\n` );

const { server: signalServer } = await startSignalServer( SIGNAL_PORT );
console.log( `信令服务器：ws://127.0.0.1:${ SIGNAL_PORT }` );

const { createServer } = await import( 'vite' );
const vite = await createServer( {
  root,
  configFile: false,
  logLevel: 'error',
  server: { host: '127.0.0.1', port: VITE_PORT, strictPort: true },
} );
await vite.listen();
const base = `http://127.0.0.1:${ VITE_PORT }`;
console.log( `开发服务器：${ base }\n` );

const child = spawn( chrome, [
  '--headless=new',
  `--remote-debugging-port=${ CDP_PORT }`,
  `--user-data-dir=${ userDataDir }`,
  '--no-first-run',
  '--no-default-browser-check',
  '--disable-extensions',
  '--disable-background-networking',
  '--window-size=430,1400',
  'about:blank',
], { stdio: 'ignore' } );

let exitCode = 1;
try
{
  // 等 CDP 端口就绪
  const deadline = Date.now() + 30000;
  for ( ;; )
  {
    try
    {
      const v = await fetch( `http://127.0.0.1:${ CDP_PORT }/json/version` ).then( ( r ) => r.json() );
      if ( v.webSocketDebuggerUrl ) break;
    }
    catch { /* 还没起来 */ }
    if ( Date.now() > deadline ) throw new Error( 'Chromium 的 CDP 端口 30s 内没就绪' );
    await delay( 400 );
  }

  await runScenario( {
    name: '本机多标签页',
    query: `quick=1&room=ui-local&mode=local`,
    base, cdpPort: CDP_PORT, shotPrefix: 'local',
  } );

  await runScenario( {
    name: 'WebRTC + 信令',
    query: `quick=1&room=ui-rtc&signal=${ encodeURIComponent( `ws://127.0.0.1:${ SIGNAL_PORT }` ) }`,
    base, cdpPort: CDP_PORT, shotPrefix: 'webrtc',
  } );

  let failed = 0;
  console.log( '结果：\n' );
  for ( const r of results )
  {
    const tag = r.pass ? '\u001b[32mPASS\u001b[0m' : '\u001b[31mFAIL\u001b[0m';
    console.log( `${ tag }  ${ r.name }` );
    console.log( `      ${ r.detail }` );
    if ( !r.pass ) failed += 1;
  }
  console.log( `\n${ results.length - failed }/${ results.length } 项通过` );
  console.log( `截图已保存到 demo/screenshots/` );
  exitCode = failed === 0 ? 0 : 1;
}
catch ( err )
{
  console.log( `\n测试中断：${ err.message}` );
  for ( const r of results ) console.log( `${ r.pass ? 'PASS' : 'FAIL' }  ${ r.name } — ${ r.detail }` );
  console.log( `\n0/${ results.length || 1 } 项通过（中断）` );
}
finally
{
  try { child.kill( 'SIGKILL' ); } catch { /* 忽略 */ }
  await signalServer.close();
  await vite.close();
  await delay( 300 );
  process.exit( exitCode );
}
