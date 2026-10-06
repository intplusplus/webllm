/**
 * WebGPU 能力探针：启动本机 Chromium，如实报告这个环境到底能不能跑 WebGPU 训练。
 *
 * 用途：
 *   1. 开发时确认无头 Chromium 有没有可用的 WebGPU 适配器（决定自动化测试能覆盖到哪）
 *   2. 手机排查时对照：同一段探测逻辑也跑在 fed.html 的「能力诊断」里
 *
 * 用法：
 *   node scripts/probe-webgpu.mjs                # 默认参数
 *   node scripts/probe-webgpu.mjs --swiftshader  # 强制软件适配器
 */
import http from 'node:http';
import { findChrome, launchChrome, newPage, evaluate, delay } from './lib/cdp.mjs';

const chrome = findChrome();
if ( !chrome )
{
  console.log( '未找到 Chromium（ms-playwright 缓存为空）。' );
  process.exit( 2 );
}

const useSwift = process.argv.includes( '--swiftshader' );
const extraArgs = [
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--use-angle=swiftshader',
  ...( useSwift ? [ '--use-webgpu-adapter=swiftshader' ] : [] ),
];

console.log( `浏览器：${ chrome }` );
console.log( `参数：${ extraArgs.join( ' ' ) }\n` );

// WebGPU 只在安全上下文可用。localhost / 127.0.0.1 算安全上下文，
// 所以必须在一个真实的 http 源上探测 —— 在 about:blank 上探会被 isSecureContext=false 骗到。
const PAGE_PORT = 5399;
const pageServer = http.createServer( ( _req, res ) =>
{
  res.writeHead( 200, { 'content-type': 'text/html; charset=utf-8' } );
  res.end( '<!DOCTYPE html><title>webgpu probe</title><body>probe</body>' );
} );
await new Promise( ( r ) => pageServer.listen( PAGE_PORT, '127.0.0.1', r ) );

const { cdpPort, close } = await launchChrome( { chrome, extraArgs } );
let cdp;
try
{
  cdp = await newPage( cdpPort );
  await cdp.send( 'Page.navigate', { url: `http://127.0.0.1:${ PAGE_PORT }/` } );
  await delay( 800 );

  const report = await evaluate( cdp, `(async () => {
    const out = { secure: window.isSecureContext, hasGpuApi: typeof navigator.gpu !== 'undefined', adapters: [] };
    if ( !out.hasGpuApi ) return out;

    const tryAdapter = async ( label, opts ) => {
      try {
        const a = await navigator.gpu.requestAdapter( opts );
        if ( !a ) { out.adapters.push( { label, ok: false, reason: 'requestAdapter 返回 null' } ); return; }
        const info = a.info || {};
        let device = null;
        try { device = await a.requestDevice(); } catch ( e ) { /* 记录下面 */ }
        out.adapters.push( {
          label, ok: true,
          vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
          features: [ ...a.features ].slice( 0, 12 ),
          maxBufferSize: a.limits.maxBufferSize,
          deviceOk: !!device,
          deviceError: device ? undefined : 'requestDevice 失败',
        } );
      } catch ( e ) {
        out.adapters.push( { label, ok: false, reason: String( e && e.message || e ) } );
      }
    };

    await tryAdapter( 'default', {} );
    await tryAdapter( 'fallback(软件)', { forceFallbackAdapter: true } );
    return out;
  })()` );

  console.log( '探测结果：' );
  console.log( `  安全上下文 isSecureContext = ${ report.secure }` );
  console.log( `  navigator.gpu 存在         = ${ report.hasGpuApi }` );
  for ( const a of report.adapters )
  {
    if ( a.ok )
    {
      console.log( `  [${ a.label }] 可用：${ a.vendor ?? '?' } / ${ a.architecture ?? '?' } / ${ a.device ?? a.description ?? '?' }` );
      console.log( `                 features: ${ ( a.features || [] ).join( ', ' ) || '(无)' }` );
      console.log( `                 maxBufferSize: ${ a.maxBufferSize }，requestDevice: ${ a.deviceOk ? 'OK' : '失败' }` );
    }
    else
    {
      console.log( `  [${ a.label }] 不可用：${ a.reason }` );
    }
  }
}
finally
{
  try { cdp?.close(); } catch { /* 忽略 */ }
  close();
  pageServer.close();
  await delay( 200 );
  process.exit( 0 );
}
