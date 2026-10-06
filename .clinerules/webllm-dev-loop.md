# webllm 项目指南（Cline）

权威操作手册在仓库根 `AGENTS.md`——执行任务前先读它。本文件是薄入口。

速记：
- 改 `src/**` 后按序跑 `npm run typecheck` → `npm run build` → 相应 `npm run verify:*`（fed/e2e/signal/ui/phone），全绿才算完
- 真机联调：`adb forward tcp:9222` + `adb reverse tcp:5173` 后手机开 `http://localhost:5173/pages/join.html`（localhost=安全上下文，WebGPU 免证书），再 `node scripts/verify-phone.mjs`
- 断开测试不能断 WiFi（系统会连带关掉无线调试），用 refresh/kill 场景
- RTCDataChannel 大帧必须背压发送（`dc.send` 队列满会同步抛异常）；信令服务器收到 pong 必须复位 alive
- 深度文档在 `docs/`，先读 `docs/架构总览.md`；黑名单（agent-browser、swiftshader flags 等）见 AGENTS.md
