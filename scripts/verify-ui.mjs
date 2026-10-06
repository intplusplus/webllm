/**
 * 真实浏览器里的界面端到端测试。
 *
 * 动机：其他自检都是无头的 —— 核心逻辑验证得很扎实，但**界面从来没有在真实浏览器里
 * 被点过**。而用户的诉求恰恰是「手机和 PC 能同时训练」，那件事只有在浏览器里才算数。
 *
 * 本机 agent-browser 的 `open` 会永久挂起，所以这里绕开它：直接 spawn 本机 Chromium，
 * 用 Node 内置 WebSocket 说 CDP（见 scripts/lib/cdp.mjs）。不引入 playwright / puppeteer。
 *
 * 两个场景各开两个标签页，跑完整流程：
 *   场景 A：本机多标签页（BroadcastChannel，免服务器）
 *   场景 B：WebRTC + 信令（跨设备走的就是这条路径，用本机回环验证）
 *
 * 用法： npm run verify:ui
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startSignalServer } from './signal-server.mjs';
import {
  clickButton,
  delay,
  evaluate,
  findChrome,
  launchChrome,
  newPage,
  screenshot,
  statusHas,
  waitFor,
} from './lib/cdp.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const SHOT_DIR = path.join( root, 'demo', 'screenshots' );
fs.mkdirSync( SHOT_DIR, { recursive: true } );

const VITE_PORT = 5175;
const SIGNAL_PORT = 5181;
const CDP_PORT = 9333;

const results = [];
const check = ( name, pass, detail ) => results.push( { name, pass, detail } );

async function runScenario ( { name, query, base, cdpPort, shotPrefix } )
{
  // 房主与节点是**两个页面**（host.html / join.html），职责不同，URL 也不同。
  const hostUrl = `${ base }/host.html?${ query }`;
  const peerUrl = `${ base }/join.html?${ query }`;
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
    await host.send( 'Page.navigate', { url: hostUrl } );
    await peer.send( 'Page.navigate', { url: peerUrl } );

    await waitFor( host, 'document.readyState === "complete"', 20000, `${ name } 主机页面加载` );
    await waitFor( peer, 'document.readyState === "complete"', 20000, `${ name } 节点页面加载` );

    // 语料加载 + 能力探测完成后按钮才可用
    await waitFor( host, '![...document.querySelectorAll("button")].find(b=>b.textContent.includes("创建训练房间")).disabled', 40000, `${ name } 主机按钮就绪` );
    await waitFor( peer, '![...document.querySelectorAll("button")].find(b=>b.textContent.includes("加入房间")).disabled', 40000, `${ name } 节点按钮就绪` );

    await clickButton( host, '创建训练房间' );
    await waitFor( host, statusHas( '房间已创建' ), 30000, `${ name } 主机建房` );

    await clickButton( peer, '加入房间' );
    // 注意：清单**不再**在建房时下发 —— 这是本轮的核心改动。
    // 房主要等点了「开始训练」、按全网能力协商出引擎之后才统一下发，
    // 所以节点这里只能等「已加入、在等房主」。等「清单指纹」会永远等不到。
    await waitFor(
      peer,
      statusHas( '等待房主下发任务' ),
      40000,
      `${ name } 节点已加入`,
    );

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
    check( `[${ name }] 两个标签页都加载出各自的角色页`, hostText.includes( '创建房间' ) && peerText.includes( '选择房间' ),
      `主机含「创建房间」=${ hostText.includes( '创建房间' ) }，节点含「选择房间」=${ peerText.includes( '选择房间' ) }` );

    check( `[${ name }] 能力面板渲染出来`, hostText.includes( '安全上下文' ) && hostText.includes( '适配器' ),
      `含「安全上下文」=${ hostText.includes( '安全上下文' ) }，含「适配器」=${ hostText.includes( '适配器' ) }` );

    check( `[${ name }] 主机创建房间并看到节点`, rosterSeen, `主机节点表可见=${ rosterSeen }` );

    check( `[${ name }] 房主按全网能力协商引擎`,
      /引擎协商结果/.test( hostText ),
      `日志含「引擎协商结果」=${ /引擎协商结果/.test( hostText ) }` );

    check( `[${ name }] 节点在开训时才收到房主下发的清单`,
      peerText.includes( '清单指纹' ) && !peerText.includes( '尚未加入任何房间' ),
      `节点含「清单指纹」=${ peerText.includes( '清单指纹' ) }，仍显示未加入=${ peerText.includes( '尚未加入任何房间' ) }` );

    check( `[${ name }] 主机完成训练并生成模型卡`,
      hostStatus.includes( '训练完成' ) && hostText.includes( '传输总量' ) && hostText.includes( '探针 loss' ),
      `状态栏「${ hostStatus.slice( 0, 60 ) }」，含「传输总量」=${ hostText.includes( '传输总量' ) }` );

    check( `[${ name }] 节点侧也收到完成并生成模型卡`,
      peerDone && peerStatus.includes( '训练完成' ) && peerText.includes( '传输总量' ),
      `节点等待到完成=${ peerDone }，状态栏「${ peerStatus.slice( 0, 60 ) }」` );

    const ledgerOk = /通过/.test( hostText ) && /第 \d+ 轮 · 全局探针 loss/.test( hostText );
    check( `[${ name }] 贡献账本渲染出逐轮裁决`, ledgerOk,
      `含「通过」=${ /通过/.test( hostText ) }，含逐轮汇总=${ /第 \d+ 轮 · 全局探针 loss/.test( hostText ) }` );

    check( `[${ name }] 模型卡包含 loss 下降记录`, hostText.includes( '最终' ) && hostText.includes( '初始' ),
      `含「初始/最终」结构=${ hostText.includes( '初始' ) && hostText.includes( '最终' ) }` );

    // 续写：验证推理路径在浏览器里可用
    await evaluate( host, `(() => {
      const i = [ ...document.querySelectorAll( 'input' ) ].find( ( x ) => x.placeholder && x.placeholder.includes( '前缀' ) );
      if ( !i ) return false;
      i.value = 'The ';
      return true;
    })()` );
    await clickButton( host, '续写' );
    await delay( 2500 );
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
    try { await screenshot( host, path.join( SHOT_DIR, `${ shotPrefix }-host.png` ) ); } catch { /* 忽略 */ }
    try { await screenshot( peer, path.join( SHOT_DIR, `${ shotPrefix }-peer.png` ) ); } catch { /* 忽略 */ }
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

console.log( '公共训练网络 · 界面端到端测试（真实 Chromium）\n' );
console.log( `浏览器：${ chrome }\n` );

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

let browser = null;
let exitCode = 1;
try
{
  browser = await launchChrome( { chrome, port: CDP_PORT } );
  console.log( `浏览器 profile：${ browser.userDataDir }\n` );

  await runScenario( {
    name: '本机多标签页',
    query: 'quick=1&room=ui-local&mode=local',
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
  console.log( '截图与页面快照已保存到 demo/screenshots/' );
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
  try { browser?.close(); } catch { /* 忽略 */ }
  await signalServer.close();
  await vite.close();
  await delay( 300 );
  process.exit( exitCode );
}
