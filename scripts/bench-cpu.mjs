/**
 * CPU 引擎耗时基准的驱动脚本：打包 bench-entry.ts 后在 Node 里跑。
 * 用法： node scripts/bench-cpu.mjs [small|medium|large] [steps] [batch]
 */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import esbuild from 'esbuild';
import { rawPlugin } from './lib/bundle.mjs';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const preset = process.argv[ 2 ] ?? 'medium';
const steps = Number( process.argv[ 3 ] ?? 20 );
const batch = Number( process.argv[ 4 ] ?? 32 );

const text = fs.readFileSync( path.join( root, 'public', 'data', 'tinyshakespeare.txt' ), 'utf8' );
const tmp = fs.mkdtempSync( path.join( os.tmpdir(), 'wb-bench-' ) );

await esbuild.build( {
  entryPoints: [ path.join( root, 'scripts', 'bench-entry.ts' ) ],
  bundle: true,
  format: 'cjs',
  platform: 'node',
  target: 'node20',
  outfile: path.join( tmp, 'bench.cjs' ),
  logLevel: 'warning',
  plugins: [ rawPlugin ],
} );

const { runBench } = await import( 'file://' + path.join( tmp, 'bench.cjs' ).replace( /\\/g, '/' ) );
const r = await runBench( text, preset, { steps, batch } );

console.log( `preset=${ preset }  steps=${ steps }  batch=${ batch }` );
console.log( `  参数量          ${ r.params.toLocaleString() }` );
console.log( `  训练池          ${ r.ids.toLocaleString() } 字符（shard #0）` );
console.log( `  单步            ${ r.oneStepMs.toFixed( 0 ) } ms` );
console.log( `  一轮（${ steps } 步） ${ r.roundMs.toFixed( 0 ) } ms` );
console.log( `  探针评估        ${ r.evalMs.toFixed( 0 ) } ms` );
console.log( `  主线程单次阻塞  ≈ ${ r.perRoundBudgetMs.toFixed( 0 ) } ms（每步一次）` );
