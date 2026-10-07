/**
 * 打包「模型编写台」的自包含单文件版： demo/ir-studio.html
 *
 * 与 fed-standalone 同一套路：JS（IIFE）/ CSS / 语料三样全内联，
 * 双击即可离线打开，不需要 Vite、不需要服务器、不需要联网。
 *
 * 用法： npm run build:standalone:ir
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import esbuild from 'esbuild';
import { rawPlugin } from './lib/bundle.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const tmp = path.join( root, 'node_modules', '.cache', 'wb-standalone-ir' );
fs.rmSync( tmp, { recursive: true, force: true } );
fs.mkdirSync( tmp, { recursive: true } );

await esbuild.build( {
  entryPoints: [ path.join( root, 'src', 'app', 'ir-main.ts' ) ],
  bundle: true,
  // file:// 下 inline module 会被 CORS 拦，所以打成 IIFE 而不是 ESM。
  format: 'iife',
  platform: 'browser',
  target: 'es2022',
  outdir: tmp,
  entryNames: 'app',
  minify: true,
  legalComments: 'none',
  logLevel: 'warning',
  define: { 'process.env.NODE_ENV': '"production"' },
  // 工程里的 WGSL 用 Vite 的 `?raw` 导入，esbuild 不认，需要这个插件。
  plugins: [ rawPlugin ],
} );

const js = fs.readFileSync( path.join( tmp, 'app.js' ), 'utf8' );
const cssPath = path.join( tmp, 'app.css' );
const css = fs.existsSync( cssPath ) ? fs.readFileSync( cssPath, 'utf8' ) : '';
const corpus = fs.readFileSync( path.join( root, 'public', 'data', 'tinyshakespeare.txt' ), 'utf8' );

if ( /(^|[;\s])import[\s(]/.test( js ) )
  throw new Error( '打包结果里仍有 import 语句，file:// 下会失败' );
if ( css.length < 500 ) throw new Error( `CSS 未被打包进来（${ css.length } 字节）` );
if ( corpus.length < 1e5 ) throw new Error( '语料过短，可能读错文件' );
if ( !js.includes( 'specHash' ) && !js.includes( 'spec' ) )
  throw new Error( '打包结果里看不到 IR 相关代码，入口可能接错了' );

// 语料以 JSON 字符串内联；把 < 转义掉，避免正文里的 </script> 提前闭合标签。
const corpusLiteral = JSON.stringify( corpus ).replace( /</g, '\\u003c' );

const html = `<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
    <meta name="theme-color" content="#0b0f16" />
    <title>模型编写台 · Spec IR（单文件版）</title>
    <style>
${css}
    </style>
  </head>
  <body>
    <nav class="site-nav">
      <span style="color:#e6edf3;font-weight:600">webllm · 模型编写台</span>
      <span class="sep">/</span>
      <span style="color:#8b98a9">单文件版 · 零服务器 · 离线可用</span>
    </nav>
    <div id="app"></div>
    <script>window.__WEBLLM_INLINE_CORPUS__ = ${corpusLiteral};</script>
    <script>
${js}
    </script>
  </body>
</html>
`;

const outDir = path.join( root, 'demo' );
fs.mkdirSync( outDir, { recursive: true } );
const outFile = path.join( outDir, 'ir-studio.html' );
fs.writeFileSync( outFile, html, 'utf8' );

const kb = ( n ) => ( n / 1024 ).toFixed( 0 ) + ' KB';
console.log( `已生成 ${ path.relative( root, outFile ) }` );
console.log( `  HTML 合计 ${ kb( Buffer.byteLength( html ) ) }（JS ${ kb( js.length ) } · CSS ${ kb( css.length ) } · 语料 ${ kb( corpus.length ) }）` );
console.log( '  双击即可打开：改模板看诊断、点修复看它真的消失、点训练看 loss 真的下降。' );
