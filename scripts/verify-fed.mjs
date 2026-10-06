/**
 * 用 esbuild 把 src/fed/selfcheck.ts 打成 Node 可执行模块并运行。
 * 这样自检脚本与浏览器里跑的是同一份源码，不存在「测的是另一套实现」。
 *
 * 用法： npm run verify:fed
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import esbuild from 'esbuild';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const cacheDir = path.join( root, 'node_modules', '.cache', 'wb-verify' );
fs.mkdirSync( cacheDir, { recursive: true } );
const outfile = path.join( cacheDir, 'selfcheck.mjs' );

await esbuild.build( {
  entryPoints: [ path.join( root, 'src', 'fed', 'selfcheck.ts' ) ],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'es2022',
  outfile,
  logLevel: 'warning',
} );

const { runFedSelfCheck } = await import( pathToFileURL( outfile ).href );

const textPath = path.join( root, 'public', 'data', 'tinyshakespeare.txt' );
const text = fs.readFileSync( textPath, 'utf8' );

console.log( `公共训练网络 · 联邦训练核心自检（语料 ${ text.length.toLocaleString() } 字符）\n` );

const results = await runFedSelfCheck( text );
let failed = 0;
for ( const r of results )
{
  const tag = r.pass ? '\u001b[32mPASS\u001b[0m' : '\u001b[31mFAIL\u001b[0m';
  console.log( `${ tag }  ${ r.name }` );
  console.log( `      ${ r.detail }` );
  if ( !r.pass ) failed += 1;
}
console.log( `\n${ results.length - failed }/${ results.length } 项通过` );
process.exit( failed === 0 ? 0 : 1 );
