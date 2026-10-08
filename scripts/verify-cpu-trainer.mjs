/**
 * CpuTrainer（CPU 兜底训练台）无头冒烟运行器。
 * 用法： npm run verify:cpu
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { bundleAndImport } from './lib/bundle.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const { runCpuTrainerSmoke } = await bundleAndImport( root, 'src/tests/train/cpu-trainer-smoke.ts', 'cpu-trainer-smoke' );

const results = await runCpuTrainerSmoke();

let failed = 0;
console.log( 'CpuTrainer · 无头冒烟（CPU 兜底训练台全链路）\n' );
for ( const r of results )
{
  const tag = r.pass ? '\u001b[32mPASS\u001b[0m' : '\u001b[31mFAIL\u001b[0m';
  console.log( `${ tag }  ${ r.name }` );
  console.log( `      ${ r.detail }` );
  if ( !r.pass ) failed += 1;
}
console.log( `\n${ results.length - failed }/${ results.length } 项通过\n` );
process.exit( failed === 0 ? 0 : 1 );
