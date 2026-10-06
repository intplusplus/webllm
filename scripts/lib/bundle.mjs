/**
 * 把浏览器端 TS 打成 Node 可跑的 ESM（自检脚本用）。
 *
 * 关键点：工程里的 WGSL 着色器是用 Vite 的 `?raw` 后缀导入的，
 * esbuild 不认识这个后缀，会直接报错。这里补一个插件把 `xxx?raw`
 * 变成「读文件内容、以字符串导出」，语义与 Vite 一致。
 *
 * 注意：因为 CPU 引擎的工厂里也会引用 GPU 引擎（房间引擎全网必须一致，
 * 两种引擎都要能构造），所以即使只测 CPU 路径，打包也会把
 * src/train/trainer.ts 和 21 个 .wgsl 一并带进来。这是有意为之：
 * 测的就是真实依赖图，不是裁剪过的子集。
 */
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import esbuild from 'esbuild';

/** 让 esbuild 支持 Vite 风格的 `?raw` 导入。 */
export const rawPlugin = {
  name: 'vite-raw',
  setup ( build )
  {
    build.onResolve( { filter: /\?raw$/ }, ( args ) => ( {
      path: path.resolve( args.resolveDir, args.path.replace( /\?raw$/, '' ) ),
      namespace: 'vite-raw',
    } ) );
    build.onLoad( { filter: /.*/, namespace: 'vite-raw' }, async ( args ) => ( {
      contents: await fs.promises.readFile( args.path, 'utf8' ),
      loader: 'text',
    } ) );
  },
};

/**
 * 打包入口并动态 import 它。
 * @param {string} root 项目根
 * @param {string} entryRel 相对项目根的入口（如 'src/fed/selfcheck.ts'）
 * @param {string} name 缓存文件名（如 'selfcheck'）
 */
export async function bundleAndImport ( root, entryRel, name )
{
  const cacheDir = path.join( root, 'node_modules', '.cache', 'wb-verify' );
  fs.mkdirSync( cacheDir, { recursive: true } );
  const outfile = path.join( cacheDir, `${ name }.mjs` );

  await esbuild.build( {
    entryPoints: [ path.join( root, entryRel ) ],
    bundle: true,
    format: 'esm',
    platform: 'node',
    target: 'es2022',
    outfile,
    logLevel: 'warning',
    plugins: [ rawPlugin ],
  } );

  return import( pathToFileURL( outfile ).href );
}
