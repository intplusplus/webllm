/**
 * 下载 Qwen2.5-0.5B-Instruct-GPTQ-Int4 权重到 public/models/。
 *
 * 为什么需要这个脚本：model.safetensors 有 438MB，被 .gitignore 排除，
 * 所以克隆仓库后需要跑一次 `npm run fetch:model` 拉回来。
 *
 * 用法：
 *   node scripts/fetch-model.mjs                 # 下载缺失的文件
 *   node scripts/fetch-model.mjs --force         # 强制重新下载
 *   node scripts/fetch-model.mjs --only tokenizer # 只下指定文件
 */
import { createWriteStream, existsSync, mkdirSync, statSync, renameSync, unlinkSync } from 'node:fs';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = 'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4';
const BASE = `https://huggingface.co/${ REPO }/resolve/main`;
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DEST = join(ROOT, 'public', 'models', 'qwen2.5-0.5b-int4');

/** 顺序即下载顺序：小的先下，尽早能用。 */
const FILES = [
  { name: 'config.json', size: 1264 },
  { name: 'tokenizer_config.json', size: 7310 },
  { name: 'vocab.json', size: 2780000 },
  { name: 'merges.txt', size: 1670000 },
  { name: 'tokenizer.json', size: 7030000 },
  { name: 'model.safetensors', size: 459383592 },
];

const args = process.argv.slice(2);
const force = args.includes('--force');
const onlyIdx = args.indexOf('--only');
const only = onlyIdx >= 0 ? args[onlyIdx + 1] : null;

function human(bytes) {
  if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(2) + ' GiB';
  if (bytes >= 1024 ** 2) return (bytes / 1024 ** 2).toFixed(2) + ' MiB';
  return (bytes / 1024).toFixed(1) + ' KiB';
}

/**
 * 流式下载到 .part 临时文件，成功后原子改名。
 * 直接写目标文件的话，中断会留下一个「看起来存在但损坏」的文件。
 */
async function download(file) {
  const target = join(DEST, file.name);
  const part = target + '.part';

  if (!force && existsSync(target) && statSync(target).size === file.size) {
    console.log(`  ✓ ${ file.name } 已存在（${ human(file.size) }），跳过`);
    return;
  }

  const res = await fetch(`${ BASE }/${ file.name }`, { redirect: 'follow' });
  if (!res.ok) throw new Error(`下载 ${ file.name } 失败：HTTP ${ res.status }`);

  const total = Number(res.headers.get('content-length') ?? file.size);
  let received = 0;
  let lastPrint = 0;

  const progress = new TransformStream({
    transform(chunk, controller) {
      received += chunk.byteLength;
      const now = Date.now();
      if (now - lastPrint > 300) {
        lastPrint = now;
        const pct = ((received / total) * 100).toFixed(1);
        process.stdout.write(`\r  ↓ ${ file.name } ${ pct }% (${ human(received) }/${ human(total) })   `);
      }
      controller.enqueue(chunk);
    },
  });

  await pipeline(
    Readable.fromWeb(res.body.pipeThrough(progress)),
    createWriteStream(part),
  );
  renameSync(part, target);
  process.stdout.write('\r');
  console.log(`  ✓ ${ file.name } (${ human(statSync(target).size) })`);
}

const targets = only ? FILES.filter((f) => f.name === only) : FILES;
if (targets.length === 0) {
  console.error(`--only "${ only }" 未匹配到任何文件。可选：${ FILES.map((f) => f.name).join(', ')}`);
  process.exit(1);
}

mkdirSync(DEST, { recursive: true });
console.log(`模型目录：${ DEST }\n共 ${ targets.length } 个文件\n`);

let failed = false;
for (const file of targets) {
  try {
    await download(file);
  } catch (err) {
    failed = true;
    console.error(`  ✗ ${ file.name }: ${ err.message }`);
    const part = join(DEST, file.name + '.part');
    if (existsSync(part)) unlinkSync(part);
  }
}

console.log();
if (failed) {
  console.error('部分文件下载失败，可重跑本脚本（已完成的文件会自动跳过）。');
  process.exit(1);
}
console.log('全部就绪，运行 `npm run dev` 后打开 http://127.0.0.1:5173');