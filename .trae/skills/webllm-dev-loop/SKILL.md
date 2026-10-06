---
name: webllm-dev-loop
description: Runs the webllm repo's verification matrix and drives real-device PC-phone federated training debugging over ADB and CDP. Use when the user asks to verify changes, run tests, debug federated training, or connect a phone for joint debugging. Do not use for inference-only or Qwen-loading tasks that never touch src/fed.
---

# webllm 开发验证环与真机联调

本 skill 是 Trae 路由入口；**权威内容以仓库根 `AGENTS.md` 为准**（多 harness 通用版，
Cursor/Copilot/Cline/Windsurf/Claude Code 的适配层也读它）。执行前先读 AGENTS.md，
深度文档在 `docs/`（先读 `docs/架构总览.md` 与 `docs/真机联调指南.md`）。

速记（完整版见 AGENTS.md）：

- 验证矩阵：`npm run typecheck` → `npm run build` → `verify:fed` → `verify:e2e` →
  `verify:signal` → `verify:ui` →（涉手机）`verify:phone`，全绿才算完
- 真机联调：`adb forward tcp:9222 localabstract:chrome_devtools_remote` +
  `adb reverse tcp:5173/5180`，手机开 `http://localhost:5173/pages/join.html`，
  再 `node scripts/verify-phone.mjs [gpu|refresh|kill]`
- 断开测试不能断 WiFi（系统会连带关掉无线调试），用 refresh/kill 场景
- 不变量：同 pass 禁 RAW/WAR/WAW；DC 大帧必须背压；信令 pong 必须复位 alive；
  训练 fp32；arena+offset 绑定；跨设备权重比较用容差
- 黑名单：agent-browser open（永久挂起）、swiftshader flags（弄坏 headless WebGPU）、
  about:blank 探测 WebGPU、`Page.crash`/`/json/new`（本机 Edge 不支持）
