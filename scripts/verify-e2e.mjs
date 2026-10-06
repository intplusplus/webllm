/**
 * 端到端联邦训练测试的运行器（用 esbuild 打包 src/fed/e2e.ts 后在 Node 里跑）。
 *
 * 用法： npm run verify:e2e
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import esbuild from 'esbuild';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const cacheDir = path.join( root, 'node_modules', '.cache', 'wb-verify' );
fs.mkdirSync( cacheDir, { recursive: true } );
const outfile = path.join( cacheDir, 'e2e.mjs' );

await esbuild.build( {
  entryPoints: [ path.join( root, 'src', 'fed', 'e2e.ts' ) ],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'es2022',
  outfile,
  logLevel: 'warning',
} );

const { runFedE2E } = await import( pathToFileURL( outfile ).href );
const text = fs.readFileSync( path.join( root, 'public', 'data', 'tinyshakespeare.txt' ), 'utf8' );

console.log( `公共训练网络 · 联邦训练端到端测试（3 节点房间，其中 1 个谎报 probeLoss）\n` );

const { results, card } = await runFedE2E( text );
let failed = 0;
for ( const r of results )
{
  const tag = r.pass ? '\u001b[32mPASS\u001b[0m' : '\u001b[31mFAIL\u001b[0m';
  console.log( `${ tag }  ${ r.name }` );
  console.log( `      ${ r.detail }` );
  if ( !r.pass ) failed += 1;
}
if ( card )
{
  console.log( '\n主机账本（逐轮裁决）：' );
  for ( const e of card.ledger )
  {
    const verdict = e.verdict === 'ok' ? '通过' : e.verdict === 'suspect' ? '异常·剔除' : '超时·剔除';
    console.log( `  r${ e.round }  ${ e.name.padEnd( 10 ) }  自报 ${ e.probeLoss.toFixed( 4 ) }  份额 ${ ( e.share * 100 ).toFixed( 0 ).padStart( 3 ) }%  ${ verdict }` );
  }
}
console.log( `\n${ results.length - failed }/${ results.length } 项通过` );
process.exit( failed === 0 ? 0 : 1 );
