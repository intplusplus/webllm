#!/usr/bin/env node
/**
 * sync-remotes.mjs —— 本地主仓库 ↔ Gitee / GitHub 双远端同步（本地优先）
 *
 * 用法：
 *   node scripts/sync-remotes.mjs [--dry-run] [--remotes=origin,github]
 *
 * 策略（本地是唯一事实源）：
 *   0. 前置检查：工作区必须干净、当前在 main 分支，否则退出码 3
 *   1. git fetch 每个远端
 *   2. 远端领先且本地无待推提交 → ff-only 快进拉取
 *   3. 双方各有提交（分叉）→ 不动仓库，打印处理指引，退出码 2
 *   4. 本地领先 → git push 到每个远端
 *
 * 凭证：不保存任何令牌；推送走 git config 里已配置的远端（SSH）。
 * 日志：.sync/sync.log（已被 .gitignore 忽略）。
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

function sh(args, label) {
  say(`\\$ ${label}`);
  const out = execFileSync(args.cmd, args.args, { cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'inherit', 'inherit'] });
  return out;
}

const logFile = path.join(repoRoot, '.sync', 'sync.log');
function persist(lines) {
  fs.mkdirSync(path.dirname(logFile), { recursive: true });
  fs.appendFileSync(logFile, lines.map((l) => `[${stamp()}] ${l}`).join('\n') + '\n');
}

function short(sha) {
  return git('rev-parse', '--short', sha);
}

// ---- 0. 前置检查 -------------------------------------------------------
const dirty = git('status', '--porcelain');
if (dirty) {
  console.error('工作区不干净，请先 commit 或 stash：\n' + dirty);
  persist(['FAIL: 工作区不干净，同步中止']);
  process.exit(3);
}
if (git('rev-parse', '--abbrev-ref', 'HEAD') !== BRANCH) {
  console.error(`当前不在 ${BRANCH} 分支，同步中止`);
  persist([`FAIL: 不在 ${BRANCH} 分支，同步中止`]);
  process.exit(3);
}

// 校验远端存在
const known = git('remote').split('\n').filter(Boolean);
for (const r of remotes) {
  if (!known.includes(r)) {
    console.error(`远端 ${r} 未配置（已知：${known.join(', ')}）`);
    persist([`FAIL: 远端 ${r} 不存在`]);
    process.exit(3);
  }
}

const summary = [];
const aheadBehind = {}; // r -> {ahead, behind}

say(`同步开始：分支 ${BRANCH}，远端 [${remotes.join(', ')}]${DRY ? '（dry-run）' : ''}`);

// ---- 1. fetch + 2/4. 判定 ---------------------------------------------
if (!DRY) {
  for (const r of remotes) {
    sh({ cmd: 'git', args: ['fetch', r, BRANCH] }, `git fetch ${r} ${BRANCH}`);
  }
}

let diverged = false;
for (const r of remotes) {
  const counts = git('rev-list', '--left-right', '--count', `${BRANCH}...${r}/${BRANCH}`);
  const [ahead, behind] = counts.split(/\s+/).map(Number);
  aheadBehind[r] = { ahead, behind };
  summary.push(`  ${r.padEnd(8)} 本地领先 ${ahead} / 落后 ${behind}`);
  if (ahead > 0 && behind > 0) diverged = true;
}

summary.forEach((l) => say(l));

if (diverged) {
  const clash = Object.entries(aheadBehind).filter(([, v]) => v.ahead > 0 && v.behind > 0).map(([k]) => k);
  console.error([
    '',
    '检测到分叉（本地与远端各有提交）：' + clash.join(', '),
    '自动停止，不做任何合并。请人工处理：',
    `  1. git log --oneline ${BRANCH}..<remote>/${BRANCH}   # 看远端多了什么`,
    `  2. 决定取舍：git merge <remote>/${BRANCH} 或 git rebase`,
    '  3. 解决后重跑本脚本',
  ].join('\n'));
  persist(['FAIL: 分叉 ' + clash.join(','), ...summary]);
  process.exit(2);
}

// ---- 2. 远端领先 → ff 快进 ---------------------------------------------
if (!DRY) {
  const behindRemote = Object.entries(aheadBehind).filter(([, v]) => v.behind > 0).map(([k]) => k);
  for (const r of behindRemote) {
    sh({ cmd: 'git', args: ['merge', '--ff-only', `${r}/${BRANCH}`] }, `git merge --ff-only ${r}/${BRANCH}`);
  }
}

// ---- 4. 本地领先 → push -------------------------------------------------
if (!DRY) {
  for (const r of remotes) {
    sh({ cmd: 'git', args: ['push', r, BRANCH] }, `git push ${r} ${BRANCH}`);
  }
}

const head = short(BRANCH);
say(`同步完成：${BRANCH} = ${git('log', '-1', '--format=%h %s', BRANCH)} -> [${remotes.join(', ')}] HEAD=${head}`);
persist([`OK: ${BRANCH} HEAD=${head}, 推送 [${remotes.join(',')}]`, ...summary]);
process.exit(0);
