/**
 * 双端联调：PC（无头 Chromium 当房主）× 手机 Chrome（真机节点）。
 *
 * 与 verify-ui.mjs 的区别：那边两个标签页都在本机，这边**节点侧是用户的手机**，
 * 通过 `adb forward tcp:9222 localabstract:chrome_devtools_remote` 把手机 Chrome 的
 * CDP 端口转到本机，再用同一套零依赖 CDP 客户端（scripts/lib/cdp.mjs）驱动。
 * 走的是真实链路：手机 ↔ PC 的 WebRTC DataChannel + 信令服务器。
 *
 * 前置条件：
 *   1. adb 已无线连上手机（adb devices 能看到设备）
 *   2. adb forward tcp:9222 localabstract:chrome_devtools_remote
 *   3. adb reverse tcp:5173 tcp:5173  +  adb reverse tcp:5180 tcp:5180
 *      （reverse 让手机访问 http://localhost:5173 —— localhost 是安全上下文，
 *        WebGPU 直接可用，不用装证书也不用 chrome://flags）
 *   4. demo 服务在跑：npm run demo（或 start-demo.cmd），5173 + 5180
 *
 * 场景：
 *   node scripts/verify-phone.mjs gpu      大模型 GPU 基线：两端 WebGPU，preset=large
 *   node scripts/verify-phone.mjs refresh  训练进行中刷新手机页面 → 重连对齐
 *   node scripts/verify-phone.mjs kill     训练进行中杀掉手机浏览器 → 重启重连对齐
 *   node scripts/verify-phone.mjs          三个全跑
 *
 * 为什么用 am force-stop 而不是断 WiFi 测断开：**手机一断 WiFi，系统会顺手关掉无线调试**，
 * ADB 一起没，脚本无法自救（2026-10-06 实测，用户确认）。断网场景请手动开飞行模式后
 * 观察两端日志；协议的「掉线止损 + 重连对齐」由 refresh/kill 覆盖。
 *
 * 环境变量： PHONE_TAB  指定手机标签页 id（默认自动挑一个 join.html 页面）
 *           KEEP=1     跑完不关闭 PC 无头浏览器（便于人工查看）
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import
{
    clickButton,
    connectCdp,
    delay,
    evaluate,
    findChrome,
    launchChrome,
    screenshot,
    statusHas,
    waitFor,
} from './lib/cdp.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const SHOT_DIR = path.join( root, 'demo', 'screenshots' );
fs.mkdirSync( SHOT_DIR, { recursive: true } );

const PC_BASE = 'http://127.0.0.1:5173';
// 手机走 localhost —— 见头部说明：安全上下文，WebGPU 可用。
const PHONE_BASE = 'http://localhost:5173';
// quick 小步数 + large 档 GPU 模型（4 层 4 头 128 维）+ 2 分片（PC 一片、手机一片）
const PARAMS = 'quick=1&preset=large&shards=2&rounds=6&steps=8&batch=8&lr=0.003';

const results = [];
const check = ( name, pass, detail ) => results.push( { name, pass, detail: String( detail ).slice( 0, 200 ) } );

// 无线设备的 adb 序列号（ip:port）。ensureTunnels() 里赋值。
let SERIAL = null;

const alphabet = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';
const newRoom = () => [ ...crypto.getRandomValues( new Uint8Array( 8 ) ) ]
    .map( ( b ) => alphabet[ b % alphabet.length ] ).join( '' );

// ------------------------------------------------------------------ 基础设施

async function ensureDemoUp ()
{
    for ( const port of [ 5173, 5180 ] )
    {
        try
        {
            await fetch( `http://127.0.0.1:${ port }`, { method: 'GET' } );
        }
        catch
        {
            throw new Error( `demo 服务没起（${ port } 无响应）。请先双击 start-demo.cmd 或 npm run demo` );
        }
    }
}

/** 发现无线设备的序列号。注意：无线调试的端口会变（用户重开后会拿到新端口，
 *  同一台手机还可能同时出现 mDNS 条目），所以只认 `ip:port + device` 这一行。 */
function phoneSerial ()
{
    const out = execFileSync( 'adb', [ 'devices' ], { encoding: 'utf8', timeout: 15000 } );
    const line = out.split( '\n' ).map( ( s ) => s.trim() ).find( ( s ) => /^\d+\.\d+\.\d+\.\d+:\d+\s+device\b/.test( s ) );
    if ( !line )
    {
        throw new Error( 'adb 没看到无线设备。请确认手机「开发者选项 → 无线调试」已打开，并 adb connect <ip>:<port>' );
    }
    return line.split( /\s+/ )[ 0 ];
}

/** 确保 CDP 转发与 localhost 反向隧道都在（无线调试重连后 forward/reverse 规则会丢）。 */
function ensureTunnels ()
{
    const serial = phoneSerial();
    SERIAL = serial;
    const adb = ( ...args ) => execFileSync( 'adb', [ '-s', serial, ...args ], { encoding: 'utf8', timeout: 15000 } );
    const fwd = adb( 'forward', '--list' );
    if ( !/tcp:9222/.test( fwd ) ) adb( 'forward', 'tcp:9222', 'localabstract:chrome_devtools_remote' );
    const rev = adb( 'reverse', '--list' );
    if ( !/tcp:5173/.test( rev ) ) adb( 'reverse', 'tcp:5173', 'tcp:5173' );
    if ( !/tcp:5180/.test( rev ) ) adb( 'reverse', 'tcp:5180', 'tcp:5180' );
}

/** 确保手机上有浏览器在跑。这台机器的浏览器是 **Edge**（UA: EdgA，包名 com.microsoft.emmx）；
 *  被系统杀掉后 9222 没有 page target，用 adb 把它拉起来（顺手开个 join 页恢复现场）。 */
async function ensurePhoneBrowser ()
{
    for ( let attempt = 0; attempt < 3; attempt++ )
    {
        try
        {
            const list = await fetch( 'http://localhost:9222/json' ).then( ( r ) => r.json() );
            if ( list.some( ( t ) => t.type === 'page' ) ) return;
        }
        catch { /* devtools 还没起来 */ }
        try
        {
            execFileSync( 'adb', [ '-s', SERIAL, 'shell', 'am', 'start', '-n', 'com.microsoft.emmx/com.microsoft.ruby.Main', '-a', 'android.intent.action.VIEW', '-d', `${ PHONE_BASE }/join.html` ], { timeout: 15000 } );
        }
        catch { /* 忽略，下面重试 */ }
        await delay( 3500 );
    }
    throw new Error( '手机浏览器（Edge）拉不起来：9222 没有 page target。请手动打开 Edge 再跑' );
}

/** 唤醒手机屏幕并延长灭屏时间。
 *  2026-10-06 联调实测：手机默认 ~30s 灭屏后 Chrome 会冻结页面，训练节点静默掉队。
 *  fed 页面现在会自动持锁，这里再加一道壳（脚本跑多久都亮屏）。 */
function wakePhone ()
{
    try
    {
        execFileSync( 'adb', [ '-s', SERIAL, 'shell', 'input keyevent KEYCODE_WAKEUP' ], { timeout: 10000 } );
        execFileSync( 'adb', [ '-s', SERIAL, 'shell', 'wm dismiss-keyguard' ], { timeout: 10000 } );
        execFileSync( 'adb', [ '-s', SERIAL, 'shell', 'settings put system screen_off_timeout 1800000' ], { timeout: 10000 } );
    }
    catch { /* 唤醒失败不致命：页面 Wake Lock 仍会尽力保活 */ }
}

/** 给页面装一个 console/错误收集器（CDP 事件版客户端不缓存事件，用注入脚本兜底）。 */
async function instrument ( cdp )
{
    await cdp.send( 'Page.enable' );
    await cdp.send( 'Runtime.enable' );
    await cdp.send( 'Page.addScriptToEvaluateOnNewDocument', {
        source: `(() => {
      window.__logs = [];
      const push = ( s ) => { try { window.__logs.push( String( s ) ); } catch {}
        if ( window.__logs.length > 400 ) window.__logs.shift(); };
      const orig = console.log.bind( console );
      console.log = ( ...a ) => { push( a.map( ( x ) => { try { return typeof x === 'object' ? JSON.stringify( x ) : String( x ); } catch { return '?'; } } ).join( ' ' ) ); orig( ...a ); };
      const origWarn = console.warn.bind( console );
      console.warn = ( ...a ) => { push( 'WARN ' + a.map( String ).join( ' ' ) ); origWarn( ...a ); };
      const origErr = console.error.bind( console );
      console.error = ( ...a ) => { push( 'ERROR ' + a.map( String ).join( ' ' ) ); origErr( ...a ); };
      window.addEventListener( 'error', ( e ) => push( 'window.onerror: ' + e.message + ' @' + ( e.filename || '' ) + ':' + e.lineno ) );
      window.addEventListener( 'unhandledrejection', ( e ) => push( 'unhandledrejection: ' + ( e.reason && e.reason.message || e.reason ) ) );
    })()`,
    } );
}

const readLogs = ( cdp ) => evaluate( cdp, '(window.__logs || []).join("\\n")' );

/** fed 应用的日志写在 DOM 的 pre.fed-log 里（不写 console），要读训练日志得从这里取。 */
const readDomLog = ( cdp ) => evaluate( cdp, `(() => { const p=[...document.querySelectorAll('pre.fed-log')]; return p.length?p[0].textContent:''; })()` );

/** 找一个**渲染进程活着**的标签页（urlPattern 过滤）。
 *  Edge 被 force-stop 后会恢复所有旧标签页，但恢复出来的都是僵尸：
 *  CDP 连上去 Runtime.enable 都超时。只有 am start 拉起的**前台**页是活的。
 *  另外 am start 的 -d URL 会在第一个 & 处被截断，所以不带参数开页，
 *  再用 CDP Page.navigate 导航到完整 URL。 */
async function findLivePage ( urlPattern = /join\.html/ )
{
    for ( let round = 0; round < 12; round++ )
    {
        let list = [];
        try
        {
            list = await fetch( 'http://localhost:9222/json' ).then( ( r ) => r.json() );
        }
        catch { await delay( 2500 ); continue; }
        const candidates = list.filter( ( t ) => t.type === 'page' && urlPattern.test( t.url || '' ) )
            // am start 新开的页 URL 最短（不带房间参数），排前面先探，少等僵尸页的超时
            .sort( ( a, b ) => a.url.length - b.url.length );
        for ( const tab of candidates )
        {
            try
            {
                const cdp = await connectCdp( tab.webSocketDebuggerUrl );
                const rs = await Promise.race( [
                    cdp.send( 'Runtime.enable' ).then( () => evaluate( cdp, 'document.readyState' ) ),
                    delay( 4000 ).then( () => { throw new Error( 'timeout' ); } ),
                ] );
                if ( rs ) return tab;
                cdp.close();
            }
            catch { /* 僵尸页，试下一个 */ }
        }
        await delay( 2500 );
    }
    return null;
}

/** 连上手机 Edge 的 CDP（adb forward 之后，9222 就是手机上的 devtools remote）。
 *  优先挑**活着的** join 页 —— Edge 恢复出来的僵尸页接上去会超时。 */
async function connectPhone ()
{
    const list = await fetch( 'http://localhost:9222/json' ).then( ( r ) => r.json() );
    let tab = null;
    if ( process.env.PHONE_TAB )
    {
        tab = list.find( ( t ) => t.id === process.env.PHONE_TAB ) ?? null;
        if ( !tab ) throw new Error( `PHONE_TAB=${ process.env.PHONE_TAB } 不存在` );
    }
    else
    {
        // 先找活着的 join 页（僵尸页会在 4s 探针里被跳过）
        tab = await findLivePage( /join\.html/ );
        // 没有活的 join 页：找任意活页，之后 openBoth 会把它导航到 join URL
        if ( !tab ) tab = await findLivePage( /.*/ );
    }
    if ( !tab )
    {
        throw new Error( '手机 Edge 上没有渲染进程活着的标签页。请手动打开 Edge 再跑' );
    }
    const cdp = await connectCdp( tab.webSocketDebuggerUrl );
    await instrument( cdp );
    // 关键：Chrome/Edge Android 会冻结/回收**后台**标签页（关 WebSocket、断 RTC、最终重载页面）。
    // 早前两轮联调故障的根因都是被测页在后台。先切前台再开训。
    await cdp.send( 'Page.bringToFront' ).catch( () => { } );
    return { cdp, tab };
}

async function openHostPage ( browser )
{
    const res = await fetch( `http://127.0.0.1:${ browser.cdpPort }/json/new?url=about:blank`, { method: 'PUT' } );
    const info = await res.json();
    const cdp = await connectCdp( info.webSocketDebuggerUrl );
    await instrument( cdp );
    await cdp.send( 'Page.bringToFront' ).catch( () => { } );
    return cdp;
}

/** 两端开页 + 等按钮就绪。extra 里的同名参数会**覆盖** PARAMS（如 rounds=20）。 */
async function openBoth ( hostCdp, peer, room, extra = '' )
{
    const p = new URLSearchParams( PARAMS );
    for ( const [ k, v ] of new URLSearchParams( extra ) ) p.set( k, v );
    const query = `${ p.toString() }&room=${ room }`;
    await peer.send( 'Page.navigate', { url: `${ PHONE_BASE }/join.html?${ query }` } );
    await hostCdp.send( 'Page.navigate', { url: `${ PC_BASE }/host.html?${ query }` } );
    await waitFor( hostCdp, 'document.readyState === "complete"', 20000, 'PC 房主页加载' );
    await waitFor( peer, 'document.readyState === "complete"', 30000, '手机节点页加载' );
    await waitFor( hostCdp, '![...document.querySelectorAll("button")].find(b=>b.textContent.includes("创建训练房间")).disabled', 60000, 'PC 建房按钮就绪' );
    await waitFor( peer, '![...document.querySelectorAll("button")].find(b=>b.textContent.includes("加入房间")).disabled', 60000, '手机加入按钮就绪' );
}

/** 手机加入房间：优先点列表项，等不到就直填房间号。 */
async function phoneJoin ( peer, room )
{
    let joined = false;
    const deadline = Date.now() + 12000;
    while ( Date.now() < deadline && !joined )
    {
        const hit = await evaluate( peer, `(() => {
      const b = [ ...document.querySelectorAll( '.fed-roomitem' ) ].find( ( x ) => x.textContent.includes( ${ JSON.stringify( room ) } ) );
      if ( !b ) return 'missing';
      b.click();
      return 'clicked';
    })()` ).catch( () => 'err' );
        if ( hit === 'clicked' ) joined = true;
        else await delay( 1000 );
    }
    if ( !joined )
    {
        await evaluate( peer, `(() => {
      const room = [ ...document.querySelectorAll( 'input' ) ].find( ( x ) => x.className.includes( 'fed-input' ) );
      if ( room ) room.value = ${ JSON.stringify( room ) };
      return !!room;
    })()` );
    }
    await clickButton( peer, '加入房间' );
    // 点击后的状态可能走两条路：
    //   建房后加入   → 「已加入，等待房主下发任务…」（assign 要等房主点开训）
    //   训练中重连   → assign 与 sync 秒到，状态可能直接从空闲跳到「已对齐到第 N 轮」
    // 后者整个窗口可能 <300ms（轮询间隔），所以等「任一有效状态」，别等某个瞬时空窗。
    const joinedOk = await waitFor(
        peer,
        `(() => { const s = (document.querySelector('.fed-status')||{}).textContent||'';
      return /等待房主下发任务|已对齐到第|第 \\d+\\/|训练完成|训练已结束/.test(s); })()`,
        40000, '手机已加入',
    ).then( () => true ).catch( () => false );
    if ( !joinedOk )
    {
        const s = String( await evaluate( peer, `(document.querySelector('.fed-status')||{}).textContent||''` ) );
        throw new Error( `点击加入后 40s 内未进入任何房间状态（当前状态：「${ s }」）` );
    }
}

/** 提取手机设备能力快照（引擎协商的第一现场输入）。 */
async function phoneCapability ( peer )
{
    const cap = await evaluate( peer, `(() => {
    const lines = document.body.innerText.split( '\\n' ).filter( ( l ) => /安全上下文|适配器/.test( l ) );
    return lines.slice( 0, 2 ).join( ' | ' );
  })()` );
    return String( cap ).replace( /\s+/g, ' ' ).slice( 0, 160 );
}

/** 等主机账本出现第 N 轮（用于卡时间点做中断）。 */
async function waitHostRound ( hostCdp, round, ms = 120000 )
{
    return waitFor(
        hostCdp,
        `(document.body.innerText.match( /第 ${ round } 轮[^\\n]*全局探针 loss/g ) || []).length > 0`,
        ms, `主机账本到第 ${ round } 轮`,
    ).then( () => true ).catch( () => false );
}

/** 等待某端到达指定状态；超时不抛异常，返回 { ok, status } 供判定与取证。 */
async function waitStatus ( cdp, text, ms, label )
{
    const ok = await waitFor( cdp, statusHas( text ), ms, label ).then( () => true ).catch( () => false );
    const status = String( await evaluate( cdp, `(document.querySelector( '.fed-status' ) || {}).textContent || ''` ).catch( () => '(读取失败)' ) );
    return { ok, status };
}

const roundLines = ( text ) => ( text.match( /第 \d+ 轮 · 全局探针 loss[^\n]*/g ) || [] );

/** 两端状态栏文本。 */
const bothStatus = async ( hostCdp, peer ) => ( {
    host: String( await evaluate( hostCdp, `(document.querySelector( '.fed-status' ) || {}).textContent || ''` ) ),
    peer: String( await evaluate( peer, `(document.querySelector( '.fed-status' ) || {}).textContent || ''` ) ),
} );

async function dump ( tag, hostCdp, peer, hostLogs, peerLogs )
{
    const hostText = String( await evaluate( hostCdp, 'document.body.innerText' ).catch( () => '' ) );
    const peerText = String( await evaluate( peer, 'document.body.innerText' ).catch( () => '' ) );
    fs.writeFileSync( path.join( SHOT_DIR, `phone-${ tag }-host.txt` ), hostText, 'utf8' );
    fs.writeFileSync( path.join( SHOT_DIR, `phone-${ tag }-peer.txt` ), peerText, 'utf8' );
    fs.writeFileSync( path.join( SHOT_DIR, `phone-${ tag }-host.log` ), hostLogs || '', 'utf8' );
    fs.writeFileSync( path.join( SHOT_DIR, `phone-${ tag }-peer.log` ), peerLogs || '', 'utf8' );
    await screenshot( hostCdp, path.join( SHOT_DIR, `phone-${ tag }-host.png` ) ).catch( () => { } );
    await screenshot( peer, path.join( SHOT_DIR, `phone-${ tag }-peer.png` ) ).catch( () => { } );
    return { hostText, peerText };
}

/** 共同判定（两端完成 / 引擎 / 清单 / 账本 / 摘要一致 / 手机无异常）。 */
async function finalChecks ( tag, hostCdp, peer, { expectGpu } )
{
    const st = await bothStatus( hostCdp, peer );
    const hostText = String( await evaluate( hostCdp, 'document.body.innerText' ).catch( () => '' ) );
    const peerText = String( await evaluate( peer, 'document.body.innerText' ).catch( () => '' ) );
    const peerLogs = await readLogs( peer ).catch( () => '' );

    // 晚到重连的终态是「训练已结束…你加入晚了」，与「训练完成」同等有效
    check( `[${ tag }] 两端都到达训练终态`, st.host.includes( '训练完成' ) && ( st.peer.includes( '训练完成' ) || st.peer.includes( '训练已结束' ) ),
        `PC「${ st.host.slice( 0, 40 ) }」 手机「${ st.peer.slice( 0, 40 ) }」` );
    check( `[${ tag }] 引擎协商为 ${ expectGpu ? 'gpu-tinygpt（两端 WebGPU）' : 'mlp' }`,
        /引擎协商结果/.test( hostText ) && ( expectGpu ? /gpu-tinygpt/.test( hostText ) : /mlp/.test( hostText ) ),
        ( hostText.match( /引擎协商结果[^\n]*/ ) || [ '未见该日志' ] )[ 0 ].slice( 0, 140 ) );
    check( `[${ tag }] 手机收到房主下发的清单`, peerText.includes( '清单指纹' ),
        `含「清单指纹」=${ peerText.includes( '清单指纹' ) }` );
    // 账本表格只渲染「最近一轮」明细；逐轮汇总不重复出现 —— 判据用模型卡与贡献者表
    const card = /([\d.]+)（初始） → ([\d.]+)（最终）/.exec( hostText );
    check( `[${ tag }] 模型卡 loss 明显下降`, !!card && Number( card[ 2 ] ) < Number( card[ 1 ] ) - 0.3,
        card ? `初始 ${ card[ 1 ] } → 最终 ${ card[ 2 ] }` : '未见模型卡' );
    // 注意：refresh/drop 场景里手机是中途加入的，物理上只能参加剩余轮次；
    // 浏览器重启会让手机变成**另一个 peerId**，贡献者表里会有多行「我的手机」，要累加。
    const phoneRows = [ ...hostText.matchAll( /我的手机\s+Android\s+(\d+)\s+([\d,]+)\s+(\d+)\s*\/\s*(\d+)/g ) ];
    const totalRounds = phoneRows.reduce( ( a, m ) => a + Number( m[ 1 ] ), 0 );
    const totalOk = phoneRows.reduce( ( a, m ) => a + Number( m[ 3 ] ), 0 );
    const totalBad = phoneRows.reduce( ( a, m ) => a + Number( m[ 4 ] ), 0 );
    check( `[${ tag }] 手机参与了聚合且无异常`, totalOk >= 1 && totalBad === 0,
        phoneRows.length ? `共 ${ phoneRows.length } 个实例：参与 ${ totalRounds } 轮 · 通过/异常 ${ totalOk }/${ totalBad }` : '贡献者表未见手机' );
    const hostDigest = ( hostText.match( /全局权重摘要\s*([0-9a-f]{6,})/ ) || [] )[ 1 ];
    const peerDigest = ( peerText.match( /全局权重摘要\s*([0-9a-f]{6,})/ ) || [] )[ 1 ];
    check( `[${ tag }] 两端全局权重摘要一致`, !!hostDigest && hostDigest === peerDigest,
        `PC=${ hostDigest } 手机=${ peerDigest }` );
    check( `[${ tag }] 手机侧无未捕获异常`, !/unhandledrejection|window\.onerror/.test( peerLogs ),
        peerLogs.split( '\n' ).filter( ( l ) => /unhandledrejection|window\.onerror/.test( l ) ).slice( 0, 2 ).join( ' ; ' ) || '无' );
}

// ------------------------------------------------------------------ 场景

async function scenarioGpu ( hostCdp, phone )
{
    const room = newRoom();
    const peer = phone.cdp;
    console.log( `\n=== 场景 gpu：大模型 GPU 基线（preset=large，房间 ${ room }）===` );
    await openBoth( hostCdp, peer, room );
    console.log( `手机能力：${ await phoneCapability( peer ) }` );

    await clickButton( hostCdp, '创建训练房间' );
    await waitFor( hostCdp, statusHas( '房间已创建' ), 30000, 'PC 建房' );
    await phoneJoin( peer, room );
    console.log( `[${ new Date().toLocaleTimeString() }] 手机已加入` );

    const rosterOk = await waitFor( hostCdp, `((document.body.innerText.match( /通道就绪/g ) || []).length >= 2)`, 30000, 'PC 看到手机' )
        .then( () => true ).catch( () => false );
    check( '[gpu] PC 节点表出现手机（通道就绪）', rosterOk, `rosterOk=${ rosterOk }` );

    await clickButton( hostCdp, '开始训练' );
    await waitFor( hostCdp, statusHas( '训练完成' ), 300000, 'PC 训练完成' );
    console.log( `[${ new Date().toLocaleTimeString() }] PC 完成，等手机…` );
    const peerDone = await waitStatus( peer, '训练完成', 200000, '手机训练完成' );
    console.log( `[${ new Date().toLocaleTimeString() }] 手机侧：${ peerDone.ok ? '完成' : '未完成' }（${ peerDone.status }）` );
    if ( !peerDone.ok ) check( '[gpu] 手机侧训练完成', false, `状态停在「${ peerDone.status }」` );

    const hostText = String( await evaluate( hostCdp, 'document.body.innerText' ) );
    const rounds = roundLines( hostText );
    console.log( '逐轮记录：' );
    for ( const r of rounds ) console.log( `  ${ r }` );

    await dump( 'gpu', hostCdp, peer, await readLogs( hostCdp ).catch( () => '' ), await readLogs( peer ).catch( () => '' ) );
    await finalChecks( 'gpu', hostCdp, peer, { expectGpu: true } );

    // 手机参与度：账本里 Android 行的通过次数
    const peerText = String( await evaluate( peer, 'document.body.innerText' ) );
    const android = ( peerText.match( /我的手机[\s\S]{0,80}/ ) || [ '' ] )[ 0 ];
    check( '[gpu] 手机全程参与聚合', /Android/.test( android ) && !/0 \/ 0/.test( android ),
        android.replace( /\s+/g, ' ' ).slice( 0, 120 ) );
}

async function scenarioRefresh ( hostCdp, phone )
{
    const room = newRoom();
    const peer = phone.cdp;
    console.log( `\n=== 场景 refresh：训练中刷新手机页面重连（房间 ${ room }）===` );
    await openBoth( hostCdp, peer, room );
    await clickButton( hostCdp, '创建训练房间' );
    await waitFor( hostCdp, statusHas( '房间已创建' ), 30000, 'PC 建房' );
    await phoneJoin( peer, room );
    await clickButton( hostCdp, '开始训练' );
    const reached = await waitHostRound( hostCdp, 2, 150000 );
    console.log( `[${ new Date().toLocaleTimeString() }] 主机已到第 2 轮（${ reached }），刷新手机页面模拟重连` );

    const reloadOk = await peer.send( 'Page.navigate', { url: `${ PHONE_BASE }/join.html?${ PARAMS }&room=${ room }&rejoin=${ Date.now() }` } )
        .then( () => true ).catch( () => false );
    await waitFor( peer, `![...document.querySelectorAll("button")].find(b=>b.textContent.includes("加入房间")).disabled`, 60000, '手机页面重载' );
    await phoneJoin( peer, room );
    // 注意：节点收到清单后状态栏不会变（要等 sync 帧或第一轮）——
    // 这里等的是「清单指纹出现」或「已对齐」或「训练状态」，不是某个具体文案。
    const synced = await waitFor(
        peer,
        `(() => { const t = document.body.innerText; const s = (document.querySelector('.fed-status')||{}).textContent||'';
      return t.includes('清单指纹') || /已对齐到第|第 \\d+\\/|训练完成/.test(s); })()`,
        90000, '手机收到清单或对齐',
    ).then( () => true ).catch( () => false );
    const st = await bothStatus( hostCdp, peer );
    const aligned = /已对齐到第 (\d+) 轮/.exec( st.peer );
    console.log( `[${ new Date().toLocaleTimeString() }] 手机状态：${ st.peer }` );
    check( '[refresh] 手机重连后拿到清单并对齐到当前轮（≥2）', synced && !!aligned && Number( aligned[ 1 ] ) >= 2,
        `状态「${ st.peer.slice( 0, 60 ) }」 reloadOk=${ reloadOk } 清单=${ synced }` );

    await waitFor( hostCdp, statusHas( '训练完成' ), 300000, 'PC 训练完成' );
    const peerDone = await waitStatus( peer, '训练完成', 200000, '手机训练完成' );
    console.log( `[${ new Date().toLocaleTimeString() }] 手机侧：${ peerDone.ok ? '完成' : '未完成' }（${ peerDone.status }）` );
    if ( !peerDone.ok ) check( '[refresh] 手机侧训练完成', false, `状态停在「${ peerDone.status }」` );
    const hostText = String( await evaluate( hostCdp, 'document.body.innerText' ) );
    for ( const r of roundLines( hostText ) ) console.log( `  ${ r }` );

    await dump( 'refresh', hostCdp, peer, await readLogs( hostCdp ).catch( () => '' ), await readLogs( peer ).catch( () => '' ) );
    await finalChecks( 'refresh', hostCdp, peer, { expectGpu: true } );
}

async function scenarioKill ( hostCdp, phone )
{
    const room = newRoom();
    const peer = phone.cdp;
    // rounds=40：手机重启约需 10~40s（含上面的观察窗），主机单机一轮不到 1 秒。
    // 40 轮主机单跑也要 30 秒上下，保证手机是在**训练进行中**重连，验「sync 对齐」路径。
    console.log( `\n=== 场景 kill：训练中杀掉手机浏览器（force-stop，不碰 WiFi/ADB）→ 重启重连（房间 ${ room }，40 轮）===` );
    await openBoth( hostCdp, peer, room, '&rounds=40' );
    await clickButton( hostCdp, '创建训练房间' );
    await waitFor( hostCdp, statusHas( '房间已创建' ), 30000, 'PC 建房' );
    await phoneJoin( peer, room );
    await clickButton( hostCdp, '开始训练' );
    const reached = await waitHostRound( hostCdp, 2, 150000 );
    console.log( `[${ new Date().toLocaleTimeString() }] 主机已到第 2 轮（${ reached }），杀掉手机上的浏览器` );

    // am force-stop：等价于用户从后台划掉整个浏览器 / 系统回收（比崩一个页面更狠）。
    // **不断 WiFi** —— 手机一断 WiFi，系统会顺手关掉无线调试（用户 2026-10-06 确认），
    // ADB 一起没，脚本无法自救。
    execFileSync( 'adb', [ '-s', SERIAL, 'shell', 'am', 'force-stop', 'com.microsoft.emmx' ], { timeout: 15000 } );

    // 杀掉后的窗口期：实时打印主机状态（在等？超时止损？还是循环中断？——都要看见）。
    // 只观察 10 秒：手机重启总共也就 10 秒上下，窗口太长主机会先跑完。
    for ( let i = 0; i < 2; i++ )
    {
        await delay( 5000 );
        const s = String( await evaluate( hostCdp, `(document.querySelector('.fed-status')||{}).textContent||''` ).catch( () => '(读取失败)' ) );
        console.log( `  +${ ( i + 1 ) * 5 }s 主机状态：${ s.slice( 0, 90 ) }` );
    }
    // 主机推进判定看**状态栏**（账本只显示最近一轮，等不到「第 4 轮」行）
    const hostAdvanced = await waitFor(
        hostCdp,
        `(() => { const s=(document.querySelector('.fed-status')||{}).textContent||''; return /第 [4-9]\\/|第 \\d{2}\\/|训练完成/.test(s); })()`,
        150000, '主机继续推进（≥第4轮或完成）',
    ).then( () => true ).catch( () => false );
    check( '[kill] 手机被杀期间主机训练不阻塞（继续推进）', hostAdvanced,
        `继续推进=${ hostAdvanced }` );

    // 重启 Edge 并开一个**无参** join 页（am start 的 -d URL 会在第一个 & 处被截断，
    // 带不了房间号，所以只负责把浏览器拉起来，完整 URL 交给 CDP Page.navigate）。
    execFileSync( 'adb', [ '-s', SERIAL, 'shell', 'am', 'start', '-n', 'com.microsoft.emmx/com.microsoft.ruby.Main', '-a', 'android.intent.action.VIEW', '-d', `${ PHONE_BASE }/join.html` ], { timeout: 15000 } );

    // 找活着的页（恢复出来的僵尸页 CDP 超时，必须跳过）
    const live = await findLivePage( /join\.html/ );
    if ( !live ) throw new Error( 'Edge 重启后找不到渲染进程活着的 join 标签页' );
    phone.cdp.close();
    phone.cdp = await connectCdp( live.webSocketDebuggerUrl );
    await instrument( phone.cdp );
    // 导航到带房间号的完整 URL（新开的一页，注入的收集器这次能生效）
    await phone.cdp.send( 'Page.navigate', { url: `${ PHONE_BASE }/join.html?${ new URLSearchParams( PARAMS ).toString() }&rounds=40&room=${ room }&rejoin=${ Date.now() }` } );
    await phone.cdp.send( 'Page.bringToFront' ).catch( () => { } );
    console.log( `[${ new Date().toLocaleTimeString() }] 手机重启完成，活标签页 ${ live.id }` );

    // 冷启动 + vite 首编译 + 语料加载，最多给 120s
    await waitFor( phone.cdp, `![...document.querySelectorAll("button")].find(b=>b.textContent.includes("加入房间")).disabled`, 120000, '新标签页就绪' );

    await phoneJoin( phone.cdp, room );
    // 同 refresh：等「清单/对齐/训练态」任一出现，再读状态做对齐判定
    const synced = await waitFor(
        phone.cdp,
        `(() => { const t = document.body.innerText; const s = (document.querySelector('.fed-status')||{}).textContent||'';
      return t.includes('清单指纹') || /已对齐到第|第 \\d+\\/|训练完成/.test(s); })()`,
        90000, '手机收到清单或对齐',
    ).then( () => true ).catch( () => false );
    const st = await bothStatus( hostCdp, phone.cdp ).catch( () => ( { host: '', peer: '' } ) );
    const aligned = /已对齐到第 (\d+) 轮/.exec( st.peer );
    const late = /训练已结束/.test( st.peer );
    console.log( `[${ new Date().toLocaleTimeString() }] 手机重连后状态：${ st.peer }` );
    check( '[kill] 手机重启后拿到清单并进入正确状态（对齐当前轮 or 明确的收尾提示）',
        synced && ( ( !!aligned && Number( aligned[ 1 ] ) >= 3 ) || late ),
        `状态「${ st.peer.slice( 0, 60 ) }」 清单=${ synced } 对齐=${ aligned ? aligned[ 1 ] : '-' } 晚到=${ late }` );

    // 等主机收尾，同时监测「主机页面被重载回初始状态」（渲染进程崩溃的早期信号）。
    // 2026-10-06 全套联调时主机页面在 kill 场景剩余轮次中崩溃重载过一次，
    // 表现为盲等 5 分钟超时；加了这场比赛就能立即失败并保留现场。
    const pcOutcome = await Promise.race( [
        waitFor( hostCdp, statusHas( '训练完成' ), 300000, 'PC 训练完成' ).then( () => 'done' ),
        waitFor(
            hostCdp,
            `(() => { const s=(document.querySelector('.fed-status')||{}).textContent||'';
               return /空闲/.test(s) && [...document.querySelectorAll('button')].some((b)=>b.textContent.includes('创建训练房间')); })()`,
            300000, '主机页面疑似重载',
        ).then( () => 'reloaded' ),
    ] );
    if ( pcOutcome === 'reloaded' )
    {
        throw new Error( 'PC 房主页面中途重载回了初始状态（渲染进程崩溃？）—— 取证见 demo/screenshots/phone-interrupt-host.*' );
    }
    // 晚到路径的终态文案是「训练已结束…你加入晚了」，与「训练完成」同等有效；
    // 而且它**立即**就是终态，不需要等。晚到就不用等，省得白等 200s 超时。
    const peerDone = late
        ? { ok: false, status: st.peer }
        : await waitStatus( phone.cdp, '训练完成', 200000, '手机训练完成' );
    const peerTerminal = peerDone.ok || /训练已结束|训练完成/.test( peerDone.status );
    console.log( `[${ new Date().toLocaleTimeString() }] 手机侧：${ peerTerminal ? '已收尾' : '未收尾' }（${ peerDone.status }）` );
    if ( !peerTerminal ) check( '[kill] 手机侧到达训练终态', false, `状态停在「${ peerDone.status }」` );
    const hostText = String( await evaluate( hostCdp, 'document.body.innerText' ) );
    for ( const r of roundLines( hostText ) ) console.log( `  ${ r }` );

    await dump( 'kill', hostCdp, phone.cdp, await readLogs( hostCdp ).catch( () => '' ), await readLogs( phone.cdp ).catch( () => '' ) );
    await finalChecks( 'kill', hostCdp, phone.cdp, { expectGpu: true } );

    // 主机日志（DOM）里应出现「掉线止损 → 重新加入」的痕迹
    const hostDomLog = String( await readDomLog( hostCdp ).catch( () => '' ) );
    check( '[kill] 主机日志记录了掉线与重启重连',
        /已断开/.test( hostDomLog ) && /(在训练进行中加入|已就绪)/.test( hostDomLog ),
        hostDomLog.split( '\n' ).filter( ( l ) => /断开|加入|就绪|恢复|等待|中断/.test( l ) ).slice( -5 ).join( ' ; ' ).slice( 0, 200 ) || '无' );
}

// ------------------------------------------------------------------ 主流程

const scenarios = { gpu: scenarioGpu, refresh: scenarioRefresh, kill: scenarioKill };
const wanted = process.argv.slice( 2 ).filter( ( a ) => a in scenarios );
const runList = wanted.length ? wanted : [ 'gpu', 'refresh', 'kill' ];

const chrome = findChrome();
if ( !chrome )
{
    console.log( '未找到 Chromium（ms-playwright 缓存）。中止。' );
    process.exit( 2 );
}

await ensureDemoUp();
ensureTunnels();
wakePhone();
await ensurePhoneBrowser();
console.log( `双端联调 · 场景：${ runList.join( ', ' ) }` );
console.log( `PC 房主 ${ PC_BASE } · 手机节点 ${ PHONE_BASE }（adb reverse → 安全上下文）\n` );

let browser = null;
let phone = null;
let hostCdp = null;
try
{
    browser = await launchChrome( { chrome, port: 9335, windowSize: '1000,1600' } );
    phone = await connectPhone();
    console.log( `手机标签页：${ phone.tab.url }（id ${ phone.tab.id }）` );
    hostCdp = await openHostPage( browser );

    for ( const name of runList ) await scenarios[ name ]( hostCdp, phone );

    let failed = 0;
    console.log( '\n联调结果：\n' );
    for ( const r of results )
    {
        console.log( `${ r.pass ? 'PASS' : 'FAIL' }  ${ r.name }\n      ${ r.detail }` );
        if ( !r.pass ) failed += 1;
    }
    console.log( `\n${ results.length - failed }/${ results.length } 项通过` );
    console.log( '截图与日志：demo/screenshots/phone-*.png|txt|log' );
    process.exitCode = failed === 0 ? 0 : 1;
}
catch ( err )
{
    console.log( `\n联调中断：${ err.message }` );
    for ( const r of results ) console.log( `${ r.pass ? 'PASS' : 'FAIL' }  ${ r.name } — ${ r.detail }` );
    // 取证：中断时把主机页的注入日志（console/未捕获异常）和页面文本落盘。
    // 2026-10-06 联调中主机页面在 kill 场景中途崩溃重载过，下一次再犯要有现场。
    try
    {
        const logs = String( await readLogs( hostCdp ).catch( () => '' ) );
        const text = String( await evaluate( hostCdp, 'document.body.innerText' ).catch( () => '' ) );
        fs.writeFileSync( path.join( SHOT_DIR, 'phone-interrupt-host.log' ), logs, 'utf8' );
        fs.writeFileSync( path.join( SHOT_DIR, 'phone-interrupt-host.txt' ), text, 'utf8' );
        console.log( `主机取证已写入 demo/screenshots/phone-interrupt-host.*（日志 ${ logs.length } 字符）` );
    }
    catch { /* 浏览器可能已死，忽略 */ }
    process.exitCode = 1;
}
finally
{
    if ( process.env.KEEP !== '1' )
    {
        try { browser?.close(); } catch { /* 忽略 */ }
        phone?.cdp?.close();
    }
}
