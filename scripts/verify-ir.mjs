/**
 * Spec IR（P0/P1/P2）的验收自检运行器。
 * 用法： npm run verify:ir
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { bundleAndImport } from './lib/bundle.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const { runIrSelfCheck } = await bundleAndImport( root, 'src/tests/ir/selfcheck.ts', 'ir-selfcheck' );

// `--log <file>`：把结果按 UTF-8 落盘（Windows 控制台编码会吃掉中文，CI/IDE 里靠它取回）。
const logIdx = process.argv.indexOf( '--log' );
const logFile = logIdx >= 0 ? process.argv[ logIdx + 1 ] : null;
const plain = [];

const header = 'Spec IR · 静态分析与编译层自检（design 03/05/06）\n';
console.log( header );
plain.push( header );

const results = await runIrSelfCheck();
let failed = 0;
for ( const r of results )
{
  const tag = r.pass ? '\u001b[32mPASS\u001b[0m' : '\u001b[31mFAIL\u001b[0m';
  const line = `${ r.pass ? 'PASS' : 'FAIL' }  ${ r.name }\n      ${ r.detail }\n`;
  console.log( `${ tag }  ${ r.name }` );
  console.log( `      ${ r.detail }` );
  plain.push( line );
  if ( !r.pass ) failed += 1;
}
const footer = `\n${ results.length - failed }/${ results.length } 项通过\n`;
console.log( footer );
plain.push( footer );

if ( logFile ) fs.writeFileSync( logFile, plain.join( '' ), 'utf8' );
process.exit( failed === 0 ? 0 : 1 );
