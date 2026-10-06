---
name: webllm-dev-loop
description: Runs the webllm repo's verification matrix and drives real-device PC-phone federated training debugging over ADB and CDP. Use when the user asks to verify changes, run tests, debug federated training, or connect a phone for joint debugging. Do not use for inference-only or Qwen-loading tasks that never touch src/fed.
---

# webllm 开发验证环与真机联调

本 skill 只收录**这个仓库特有的操作流程与踩过的坑**；通用知识（CDP/WebRTC 是什么）不重复。
细节文档在仓库内（随代码走，永远最新）：`docs/架构总览.md`、`docs/真机联调指南.md`、`docs/bug记录.md`。

## 1. 先判断该跑什么

| 改动范围 | 必跑 |
|---|---|
| 任何 `src/**` | `npm run typecheck` + `npm run build` |
| `src/fed/**` 或 `src/tests/fed/**` | 再加 `verify:fed`、`verify:e2e`、`verify:ui` |
| `scripts/signal-server.mjs` 或传输层 | 再加 `verify:signal`（含裸 socket 心跳回归） |
| 涉及手机端行为 / 用户要求联调 | 再加 `npm run verify:phone`（真机，约 3-6 分钟/场景） |
| 移动/重命名 src 文件 | 同步 4 类引用：html 入口、scripts 入口、`?raw` kernel 导入（trainer.ts / model/*.ts / tests/kernels.ts）、verify-*.mjs 的 bundle 入口 |

**全绿才算完**；任何一项 FAIL 时先读产物（`demo/screenshots/phone-*`、控制台输出），
不要重复跑同一个命令期待不同结果。

## 2. 验证矩阵（按序）

```bash
npm run typecheck          # tsc --noEmit，0 错误
npm run build              # tsc + vite build 四页
npm run verify:fed         # 19 联邦核心（引擎收敛/权重帧/篡改拦截/聚合等价）
npm run verify:e2e         # 20 三节点端到端（含故意作弊节点被抓）
npm run verify:signal      # 9  信令协议（join/relay/peer-left/长帧/心跳）
npm run verify:ui          # 20 真实 Chromium（本机多标签 + WebRTC 两场景）
npm run verify:phone       # 真机：gpu / refresh / kill 三场景（需手机在线）
```

## 3. 真机联调 SOP

前置（自愈已内建于 `scripts/verify-phone.mjs`，但要知道原理）：

1. 手机「开发者选项 → 无线调试」打开；`adb connect <ip>:<port>`（端口会变）
2. `adb forward tcp:9222 localabstract:chrome_devtools_remote` —— 手机 Edge 的 CDP
3. `adb reverse tcp:5173 tcp:5173` + `adb reverse tcp:5180 tcp:5180` —— **手机 WebGPU 的正解**：
   手机开 `http://localhost:5173`（localhost 是安全上下文，免证书免 flags）
4. demo 服务由**用户**起（`start-demo.cmd` 或 `npm run demo`，5173+5180）；
   联调脚本只读它、不杀它。服务没起会明确报错，不要代替用户常驻。

运行：`node scripts/verify-phone.mjs [gpu|refresh|kill]`，不带参数跑全部。
判定看输出最后的结果表；产物在 `demo/screenshots/phone-*.txt|png`。

场景语义：gpu=大模型基线；refresh=训练中刷新手机页面（软重连）；
kill=训练中 `am force-stop` 杀浏览器后重启（硬重连）。两者都验证
「主机不阻塞 + 手机对齐到当前轮 + 两端权重摘要一致」。

## 4. Android / Edge 坑（全部实测复现，违反必浪费时间）

- **断 WiFi 会连带关掉无线调试**（系统行为）→ 断开测试只用 refresh/kill 场景，
  永远不要 `svc wifi disable` 或开飞行模式来做自动化断开测试
- Edge 包名 `com.microsoft.emmx`，启动 Activity 是 `com.microsoft.ruby.Main`
- `am start -d "<url>"` 的 URL 在**第一个 & 处被截断** —— 只开无参页，
  完整 URL 用 CDP `Page.navigate`
- force-stop 后 Edge 恢复的旧标签页是**僵尸**（CDP 连上就超时）；只有 am start
  拉起的**前台**页是活的。找活页要对每个候选做 ~3s 超时探针，URL 短的排前面
- 被测标签页必须 `Page.bringToFront`：后台页会被冻结/回收（关 WS、断 RTC、重载页面）
- `Page.crash` 与 `/json/new` 在这台 Edge 上不支持（返回非 JSON/无响应）
- adb 命令一律带 `-s <serial>`（无线设备 + mDNS 条目并存会报 more than one device）
- 转发僵死（规则在但 fetch 失败）：`forward --remove tcp:9222` 再 add
- 手机侧 fed 页训练期间会自动持 Wake Lock；脚本侧也设 30 分钟灭屏

## 5. 本项目不变量（改动前对照，完整版见 docs/架构总览.md §5）

1. 同 pass 内 dispatch 间不得有对同一 buffer 的 RAW/WAR/WAW（未定义行为，BUG-001）
2. RTCDataChannel 大帧必须背压发送；`dc.send` 队列满**同步抛异常**（BUG-008）
3. 信令服务器收到 pong 必须复位 alive（BUG-007）
4. 训练全程 fp32；无 fp32 atomicAdd → 权重梯度用「每输出元素唯一线程 + 规约」
5. 单 kernel 最多 4 bind group → arena + offset，不按张量绑
6. 联邦：引擎全网一致（manifest 定死）、字符表全网一致、中途加入必须 sync 对齐
7. 跨设备权重比较用容差（1e-3），不哈希对拍

## 6. 已证实无效的手段（别再试）

- `agent-browser open <url>`：本机永久挂起（CLI/daemon IPC 问题）→ 用
  spawn chromium + 零依赖 CDP（scripts/lib/cdp.mjs，verify-ui/verify-phone 都在用）
- swiftshader / `--use-angle` 等 flags：会把 headless Chromium 的 WebGPU 弄坏；
  只留 `--headless=new` 就能拿到真实适配器
- 在 about:blank 上探测 WebGPU：`isSecureContext=false`，结论必然错误
- `svc wifi disable` 做断开测试：见第 4 节

## 7. 取证与排障入口

| 症状 | 先看哪里 |
|---|---|
| 真机联调中断 | `demo/screenshots/phone-interrupt-host.txt|log`（脚本自动落盘） |
| 主机「房主循环中断」 | 同上；历史根因是 DC send queue is full（BUG-008，已修） |
| 手机页面静默停更 | 灭屏冻结 / 后台标签页 / 僵尸页（第 4 节） |
| 房间列表看不到房主 | 信令存活：`curl http://127.0.0.1:5180/rooms`；心跳 bug 见 BUG-007 |
| 验证脚本改源码不生效 | vite HMR 边界；必要时重启 dev server |
