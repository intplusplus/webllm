#!/usr/bin/env node
/**
 * sync-remotes.mjs —— 本地主仓库 ↔ Gitee / GitHub 双远端同步（本地优先）
 *
 * 用法：
 *   node scripts/sync-remotes.mjs [--dry-run] [--remotes=origin,github]
 *
 * 策略（本地是唯一事实源）：
 *   0. 前置检查：工作区必须干净、当前在 main 分支，否则退出码 3
 *   1. git fetch 每个远端（失败自动重试一次）
 *   2. 远端领先且本地无待推提交 → ff-only 快进拉取
 *   3. 双方各有提交（分叉）→ 不动仓库，打印处理指引，退出码 2
 *   4. 本地领先 → git push 到每个远端
 *
 * 退出码：0 = 正常；1 = 网络/未预期异常（必落日志）；2 = 分叉待人工；3 = 前置检查未过
 * 凭证：不保存任何令牌；推送走 git config 里已配置的远端（SSH）。
 * 日志：.sync/sync.log（已被 .gitignore 忽略），成功与失败都写。
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const BRANCH = 'main';
const DEFAULTS = ['origin', 'github'];
const DRY = process.argv.includes('--dry-run');
const remoteArg = process.argv.find((a) => a.startsWith('--remotes='));
const remotes = (remoteArg ? remoteArg.slice('--remotes='.length).split(',') : DEFAULTS)
  .map((s) => s.trim())
  .filter(Boolean);

const stamp = () => new Date().toISOString().replace('T', ' ').slice(0, 19);
const say = (msg) => console.log(`[${stamp()}] ${msg}`);

function git(...args) {
  return execFileSync('git', args, { cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
}

/** 执行会写仓库的命令：输出实时打印，失败时抛出带完整上下文的错误 */
function sh(cmd, args, label) {
  say(`\\$ ${label}`);
  let out;
  try {
    out = execFileSync(cmd, args, { cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  } catch (e) {
    const stderr = (e.stderr || '').toString().trim();
    const stdout = (e.stdout || '').toString().trim();
    throw new Error(`\`\`\`\`\`\`\`
命令: ${label}
退出码: ${e.status ?? 'unknown'}
stdout: ${stdout || '(空)'}
stderr: ${stderr || '(空)'}
`);
  }
  const text = (out || '').trim();
  if (text) console.log(text);
}

const logFile = path.join(repoRoot, '.sync', 'sync.log');
function persist(lines) {
  fs.mkdirSync(path.dirname(logFile), { recursive: true });
  fs.appendFileSync(logFile, lines.map((l) => `[${stamp()}] ${l}`).join('\n') + '\n');
}

function fetchRemote(r) {
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      sh('git', ['fetch', r, BRANCH], `git fetch ${r} ${BRANCH}`);
      return;
    } catch (e) {
      if (attempt === 2) throw e;
      say(`fetch ${r} 失败，1 秒后重试：${e.message.split('\n')[1] || e.message}`);
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 1000);
    }
  }
}

async function main() {
  // ---- 0. 前置检查 -------------------------------------------------------
  const dirty = git('status', '--porcelain');
  if (dirty) {
    console.error('工作区不干净，请先 commit 或 stash：\n' + dirty);
    persist(['FAIL: 工作区不干净，同步中止']);
    return 3;
  }
  if (git('rev-parse', '--abbrev-ref', 'HEAD') !== BRANCH) {
    console.error(`当前不在 ${BRANCH} 分支，同步中止`);
    persist([`FAIL: 不在 ${BRANCH} 分支，同步中止`]);
    return 3;
  }

  const known = git('remote').split('\n').filter(Boolean);
  for (const r of remotes) {
    if (!known.includes(r)) {
      console.error(`远端 ${r} 未配置（已知：${known.join(', ')}）`);
      persist([`FAIL: 远端 ${r} 不存在`]);
      return 3;
    }
  }

  const summary = [];
  say(`同步开始：分支 ${BRANCH}，远端 [${remotes.join(', ')}]${DRY ? '（dry-run）' : ''}`);

  // ---- 1. fetch -----------------------------------------------------------
  if (!DRY) {
    for (const r of remotes) fetchRemote(r);
  }

  // ---- 2/4. 判定 ----------------------------------------------------------
  const aheadBehind = {};
  for (const r of remotes) {
    const counts = git('rev-list', '--left-right', '--count', `${BRANCH}...${r}/${BRANCH}`);
    const [ahead, behind] = counts.split(/\s+/).map(Number);
    aheadBehind[r] = { ahead, behind };
    summary.push(`  ${r.padEnd(8)} 本地领先 ${ahead} / 落后 ${behind}`);
  }
  summary.forEach((l) => say(l));

  const clash = Object.entries(aheadBehind).filter(([, v]) => v.ahead > 0 && v.behind > 0).map(([k]) => k);
  if (clash.length > 0) {
    console.error([
      '',
      '检测到分叉（本地与远端各有提交）：' + clash.join(', '),
      '自动停止，不做任何合并。请人工处理：',
      `  1. git log --oneline ${BRANCH}..${clash[0]}/${BRANCH}   # 看远端多了什么`,
      `  2. 决定取舍：git merge ${clash[0]}/${BRANCH} 或 git rebase`,
      '  3. 解决后重跑本脚本',
    ].join('\n'));
    persist(['FAIL: 分叉 ' + clash.join(','), ...summary]);
    return 2;
  }

  // ---- 2. 远端领先 → ff 快进 ----------------------------------------------
  if (!DRY) {
    for (const [r, v] of Object.entries(aheadBehind)) {
      if (v.behind > 0) sh('git', ['merge', '--ff-only', `${r}/${BRANCH}`], `git merge --ff-only ${r}/${BRANCH}`);
    }
  }

  // ---- 4. 本地领先 → push --------------------------------------------------
  if (!DRY) {
    for (const [r, v] of Object.entries(aheadBehind)) {
      if (v.ahead > 0 || v.behind > 0) sh('git', ['push', r, BRANCH], `git push ${r} ${BRANCH}`);
    }
  }

  const head = git('rev-parse', '--short', BRANCH);
  say(`同步完成：${BRANCH} = ${git('log', '-1', '--format=%h %s', BRANCH)} -> [${remotes.join(', ')}] HEAD=${head}`);
  persist([`OK: ${BRANCH} HEAD=${head}, 同步 [${remotes.join(',')}]`, ...summary]);
  return 0;
}

try {
  process.exit(await main());
} catch (e) {
  const msg = String(e.message || e).split('\n').slice(0, 6).join(' | ');
  console.error('同步异常中止：\n' + (e.stack || e.message || e));
  persist([`FAIL: 异常 ${msg}`]);
  process.exit(1);
}
