/**
 * 联邦训练核心的自检运行器。
 * 用法： npm run verify:fed
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { bundleAndImport } from './lib/bundle.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const { runFedSelfCheck } = await bundleAndImport( root, 'src/fed/selfcheck.ts', 'selfcheck' );

const text = fs.readFileSync( path.join( root, 'public', 'data', 'tinyshakespeare.txt' ), 'utf8' );

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
