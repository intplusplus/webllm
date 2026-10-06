/**
 * 打包「自包含单文件 demo」： demo/fed-standalone.html
 *
 * 目标是把整个联邦联训 demo 压成一个 HTML 文件：双击即可打开，不需要 Vite、
 * 不需要信令服务器、不需要联网 —— 关掉网络也能跑。
 * 配合界面里的「本机多标签页」模式，同一个浏览器开两个标签页、填同一个房间 ID，
 * 就能看到两个节点真的在同一份数据上协同训练。
 *
 * 三件事要内联：
 *   1. JS（esbuild 打成 IIFE，file:// 下 inline module 会被 CORS 拦，所以不用 ESM）
 *   2. CSS（fed.css 被 fed-app.ts import，esbuild 会单独产出 .css）
 *   3. 语料（1.1MB 文本，file:// 下 fetch 拿不到，走 corpus.ts 的内联钩子）
 *
 * 用法： npm run build:standalone
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import esbuild from 'esbuild';
import { rawPlugin } from './lib/bundle.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const tmp = path.join( root, 'node_modules', '.cache', 'wb-standalone' );
fs.rmSync( tmp, { recursive: true, force: true } );
fs.mkdirSync( tmp, { recursive: true } );

await esbuild.build( {
  entryPoints: [ path.join( root, 'src', 'app', 'fed-main.ts' ) ],
  bundle: true,
  format: 'iife',
  platform: 'browser',
  target: 'es2022',
  outdir: tmp,
  entryNames: 'app',
  minify: true,
  legalComments: 'none',
  logLevel: 'warning',
  define: { 'process.env.NODE_ENV': '"production"' },
  // 工程里的 WGSL 着色器用 Vite 的 `?raw` 导入，esbuild 不认，需要这个插件
  plugins: [ rawPlugin ],
} );

const js = fs.readFileSync( path.join( tmp, 'app.js' ), 'utf8' );
const cssPath = path.join( tmp, 'app.css' );
const css = fs.existsSync( cssPath ) ? fs.readFileSync( cssPath, 'utf8' ) : '';
const corpus = fs.readFileSync( path.join( root, 'public', 'data', 'tinyshakespeare.txt' ), 'utf8' );

if ( /(^|[;\s])import[\s(]/.test( js ) )
{
  throw new Error( '打包结果里仍有 import 语句，file:// 下会失败' );
}
if ( css.length < 500 ) throw new Error( `CSS 未被打包进来（${ css.length } 字节）` );
if ( corpus.length < 1e5 ) throw new Error( '语料过短，可能读错文件' );

// 语料以 JSON 字符串内联；把 < 转义掉，避免正文里出现 </script> 提前闭合标签
const corpusLiteral = JSON.stringify( corpus ).replace( /</g, '\\u003c' );

const html = `<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
    <meta name="theme-color" content="#0b0f16" />
    <title>公共训练网络 · 联邦联训 Demo（单文件版）</title>
    <style>
${css}
    </style>
  </head>
  <body>
    <nav class="site-nav">
      <span style="color:#e6edf3;font-weight:600">公共训练网络 · 联邦联训</span>
      <span class="sep">/</span>
      <span style="color:#8b98a9">单文件版 · 零服务器 · 离线可用</span>
    </nav>
    <div id="app"></div>
    <script>window.__WEBLLM_FED_VIEW__ = 'host';</script>
    <script>window.__WEBLLM_INLINE_CORPUS__ = ${corpusLiteral};</script>
    <script>
${js}
    </script>
    <p style="max-width:880px;margin:6px auto 40px;padding:0 14px;color:#8b98a9;font-size:12px;line-height:1.7">
      说明：这个单文件版把 JS / CSS / 语料全部内联，双击即可离线运行，可以直接创建房间、
      单机跑完一次联邦训练（含 loss 曲线、贡献账本、模型卡下载与续写）。
      多标签页联训请改用 <code style="color:#e6edf3">npm run demo</code> 启动的本地服务 ——
      部分浏览器对 <code style="color:#e6edf3">file://</code> 页面的 BroadcastChannel
      有安全限制，双标签页在该协议下可能无法互通。
    </p>
  </body>
</html>
`;

const outDir = path.join( root, 'demo' );
fs.mkdirSync( outDir, { recursive: true } );
const outFile = path.join( outDir, 'fed-standalone.html' );
fs.writeFileSync( outFile, html, 'utf8' );

const kb = ( n ) => ( n / 1024 ).toFixed( 0 ) + ' KB';
console.log( `已生成 ${ path.relative( root, outFile ) }` );
console.log( `  HTML 合计 ${ kb( Buffer.byteLength( html ) ) }（JS ${ kb( js.length ) } · CSS ${ kb( css.length ) } · 语料 ${ kb( corpus.length ) }）` );
console.log( '  双击即可打开；选「本机多标签页」模式，开两个标签页填同一房间 ID 就能联训。' );
