/**
 * 极简 CDP 工具：直接驱动本机 Chromium，零依赖（不装 playwright / puppeteer）。
 *
 * 为什么不直接用 agent-browser：本机它的 `open` 会永久挂起（CLI/daemon 的 IPC 有问题），
 * 但 Chromium 本身是好的（ms-playwright 缓存里有 chrome.exe）。
 * 于是自己 spawn + 说 CDP 协议 —— 反而更好控制，也能拿到完整的能力信息。
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';

export const delay = ( ms ) => new Promise( ( r ) => setTimeout( r, ms ) );

/** 在 ms-playwright 缓存里找最新的 chromium。 */
export function findChrome ()
{
  if ( process.env.CHROME_PATH ) return process.env.CHROME_PATH;
  const base = path.join( process.env.LOCALAPPDATA || '', 'ms-playwright' );
  if ( !fs.existsSync( base ) ) return null;
  const dirs = fs.readdirSync( base )
    .filter( ( d ) => d.startsWith( 'chromium-' ) )
    .sort()
    .reverse();
  for ( const d of dirs )
  {
    const exe = path.join( base, d, 'chrome-win64', 'chrome.exe' );
    if ( fs.existsSync( exe ) ) return exe;
  }
  return null;
}

export async function connectCdp ( wsUrl )
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
        }, 60000 );
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

/**
 * 启动一个 headless Chromium 并等 CDP 就绪。
 * 返回 { child, cdpPort, userDataDir, close() }。
 */
export async function launchChrome ( { chrome, extraArgs = [], port, windowSize = '430,1400' } = {} )
{
  const cdpPort = port ?? 9333;
  const userDataDir = fs.mkdtempSync( path.join( os.tmpdir(), 'wb-ui-chrome-' ) );
  const child = spawn( chrome, [
    '--headless=new',
    `--remote-debugging-port=${ cdpPort }`,
    `--user-data-dir=${ userDataDir }`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    '--disable-background-networking',
    `--window-size=${ windowSize }`,
    ...extraArgs,
    'about:blank',
  ], { stdio: 'ignore' } );

  const deadline = Date.now() + 40000;
  for ( ;; )
  {
    try
    {
      const v = await fetch( `http://127.0.0.1:${ cdpPort }/json/version` ).then( ( r ) => r.json() );
      if ( v.webSocketDebuggerUrl ) return { child, cdpPort, userDataDir, close: () => { try { child.kill( 'SIGKILL' ); } catch { /* 忽略 */ } } };
    }
    catch { /* 还没起来 */ }
    if ( Date.now() > deadline )
    {
      try { child.kill( 'SIGKILL' ); } catch { /* 忽略 */ }
      throw new Error( 'Chromium 的 CDP 端口 40s 内没就绪' );
    }
    await delay( 400 );
  }
}

export async function newPage ( cdpPort )
{
  const res = await fetch( `http://127.0.0.1:${ cdpPort }/json/new?url=about:blank`, { method: 'PUT' } );
  if ( !res.ok ) throw new Error( `创建标签页失败：HTTP ${ res.status }` );
  const info = await res.json();
  const cdp = await connectCdp( info.webSocketDebuggerUrl );
  await cdp.send( 'Page.enable' );
  await cdp.send( 'Runtime.enable' );
  return cdp;
}

export async function evaluate ( cdp, expression )
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
  return r.result.value;
}

export async function waitFor ( cdp, predicateExpr, ms, label )
{
  const deadline = Date.now() + ms;
  let last = '';
  while ( Date.now() < deadline )
  {
    try
    {
      if ( await evaluate( cdp, predicateExpr ) ) return true;
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

export async function clickButton ( cdp, text )
{
  const ok = await evaluate( cdp, `(() => {
    const b = [ ...document.querySelectorAll( 'button' ) ].find( ( x ) => x.textContent.includes( ${ JSON.stringify( text ) } ) );
    if ( !b ) return 'missing';
    if ( b.disabled ) return 'disabled';
    b.click();
    return true;
  })()` );
  if ( ok !== true ) throw new Error( `按钮「${ text }」不可点击（${ ok }）` );
}

export async function screenshot ( cdp, file )
{
  const r = await cdp.send( 'Page.captureScreenshot', { format: 'png', captureBeyondViewport: true } );
  fs.mkdirSync( path.dirname( file ), { recursive: true } );
  fs.writeFileSync( file, Buffer.from( r.data, 'base64' ) );
}

/** 状态栏文案。用它判断进度，不要用 body.innerText 搜关键词（会被静态占位文案骗到）。 */
export const statusHas = ( text ) =>
  `(() => { const e = document.querySelector( '.fed-status' ); return !!e && e.textContent.includes( ${ JSON.stringify( text ) } ); })()`;
