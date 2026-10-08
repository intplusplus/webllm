/**
 * GPU 后端验收：用 CDP 驱动 headless Chromium 跑 `pages/ir-gpu.html`，
 * 调 `window.__IR_GPU__.run()`（IR 在 GPU 上跑一次前向，三方对拍）。
 *
 * 前置：本地服务必须在跑（`npm run dev`，默认 5173）。
 * 为什么要真浏览器：WebGPU 只在浏览器里存在 —— 这条验收**没法在 Node 里做**。
 *
 * 用法：
 *   node scripts/verify-ir-gpu.mjs                       # 默认 http://localhost:5173
 *   node scripts/verify-ir-gpu.mjs --base http://localhost:5173 --log demo/ir-gpu.log
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { findChrome, launchChrome, newPage, evaluate, delay } from './lib/cdp.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );

function arg ( name, fallback )
{
  const i = process.argv.indexOf( `--${ name }` );
  return i >= 0 ? process.argv[ i + 1 ] : fallback;
}

const base = ( arg( 'base', 'http://localhost:5173' ) ).replace( /\/$/, '' );
const logFile = arg( 'log', null );
const plain = [];
const say = ( s ) =>
{
  console.log( s );
  plain.push( s + '\n' );
};
const finish = ( code ) =>
{
  if ( logFile ) fs.writeFileSync( logFile, plain.join( '' ), 'utf8' );
  process.exit( code );
};

say( `GPU 后端验收 · ${ base }` );

// ---- 0. 先确认本地服务在跑，并预检模块能否编译（比等页面超时快得多） ----
try
{
  await fetch( `${ base }/pages/ir-gpu.html` );
}
catch ( err )
{
  say( `\n✗ 连不上本地服务（${ err.message }）。\n  先起服务：npm run dev` );
  finish( 1 );
}

for ( const mod of [ '/src/app/ir-gpu-test.ts', '/src/ir/gpu-impls.ts', '/src/ir/gpu-exec.ts' ] )
{
  const res = await fetch( `${ base }${ mod }` );
  const body = await res.text();
  const bad = /Transform failed|Internal server error|Failed to resolve|error TS\d+/i.test( body );
  if ( res.status !== 200 || bad )
  {
    say( `\n✗ 模块编译失败：${ mod }（HTTP ${ res.status }）` );
    say( body.slice( 0, 900 ) );
    finish( 1 );
  }
}
say( '  模块预检通过（入口 + gpu-impls + gpu-exec 都能编译）' );

// ---- 1. 起 headless Chromium（WebGPU 需要显式开权限） ----
const chrome = findChrome();
if ( !chrome ) { say( '\n✗ 找不到 Chromium（ms-playwright 缓存里没有）' ); finish( 1 ); }

let launched = null;
try
{
  launched = await launchChrome( {
    chrome,
    port: 9431,
    extraArgs: [ '--enable-unsafe-webgpu', '--disable-gpu-sandbox' ],
  } );
  const cdp = await newPage( launched.cdpPort );
  await cdp.send( 'Page.navigate', { url: `${ base }/pages/ir-gpu.html` } );

  // ---- 2. 等页面把 API 挂上去 ----
  const deadline = Date.now() + 60000;
  let ready = false;
  while ( Date.now() < deadline )
  {
    try
    {
      ready = await evaluate( cdp, 'typeof window.__IR_GPU__ === "object" && typeof window.__IR_GPU__.run === "function"' );
    }
    catch { /* 页面还没就绪 */ }
    if ( ready ) break;
    await delay( 400 );
  }
  if ( !ready )
  {
    const txt = await evaluate( cdp, 'document.body.innerText.slice(0,600)' ).catch( () => '(取不到)' );
    say( `\n✗ 60s 内页面没挂上 __IR_GPU__。页面文本：\n${ txt }` );
    finish( 1 );
  }
  say( '  headless Chromium 就绪' );

  // ---- 3. 跑对拍 ----
  const t0 = Date.now();
  const r = await evaluate( cdp, 'window.__IR_GPU__.run()' );
  const ms = Date.now() - t0;

  if ( !r || r.ok !== true )
  {
    say( `\n✗ FAIL（${ ms } ms）` );
    say( r?.error ? `  错误：${ r.error }` : `  返回：${ JSON.stringify( r ) }` );
    finish( 1 );
  }

  const pct = ( x ) => x.toExponential( 2 );
  say( `\nPASS  GPU 后端 · IR 编译产物在真实显卡上跑通（${ ms } ms）` );
  say( `      适配器：${ r.adapter.vendor } / ${ r.adapter.architecture }，maxBufferSize=${ ( r.adapter.maxBufferSize / 1024 / 1024 ).toFixed( 0 ) } MiB，f16=${ r.adapter.hasF16 }` );
  say( `      模型：${ r.nodeCount } 节点（B=${ r.B }, T=${ r.T }）；artifact 计划 ${ r.plannedPasses } 个 pass` );
  say( `      实际：${ r.stats.dispatches } 次 dispatch，${ r.stats.passBreaks } 次 pass 边界；透传节点 ${ r.stats.passthrough.length } 个` );
  say( `      GPU ↔ CPU（同一 artifact，ENG-V7）：maxAbs=${ pct( r.gpuVsCpuIr.maxAbs ) } maxRel=${ pct( r.gpuVsCpuIr.maxRel ) }（${ r.gpuVsCpuIr.elements } 个 logits）` );
  say( `      GPU ↔ 手写参考（IR 语义没走样）：maxAbs=${ pct( r.gpuVsRef.maxAbs ) } maxRel=${ pct( r.gpuVsRef.maxRel ) }` );
  say( `      CPU-IR ↔ 手写参考（对照基线）：maxAbs=${ pct( r.cpuIrVsRef.maxAbs ) } maxRel=${ pct( r.cpuIrVsRef.maxRel ) }` );
  say( `      前 5 个 logits：${ r.sample.map( ( v ) => v.toFixed( 4 ) ).join( ', ' ) }` );
  say( '\n  ENG-V7 · 多后端跑同一模型，数值对拍容差内一致 ✓' );
  finish( 0 );
}
catch ( err )
{
  say( `\n✗ 验收脚本异常：${ err.message }` );
  finish( 1 );
}
finally
{
  try { launched?.close(); } catch { /* 忽略 */ }
}
