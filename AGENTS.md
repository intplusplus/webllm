# AGENTS.md

> 本文件是给**任何 AI 编码 harness**（Codex、Claude Code、Cursor、Copilot、Cline、
> Windsurf、Trae、OpenHands…）的项目操作手册。Trae 用户另有
> `.trae/skills/webllm-dev-loop/`（内容以本文件为权威源）。
> 深度文档在 `docs/`（见文末索引），**细节以docs为准，本文件保证可快速执行**。

## 项目是什么

webllm：浏览器内自研 LLM 引擎（WebGPU + WGSL，TypeScript + Vite），从零实现
tiny-GPT 训练闭环与 Qwen2.5-0.5B int4 推理；第二条线是**公共训练网络**：
PC 与手机同一 WiFi 下打开网页即可联邦训练同一个模型（P2P、数据不出本地）。

> 目录职责：`PLAN.md` 是**实施入口**（接下来做什么、怎么验收）；
> `design/` 是**权威设计稿**（v3.1 目标态，先读 `design/README.md`）；
> `docs/` 存**现状与记录**（架构总览/实测/bug/联调/设计评审记录）。
> 二者冲突时 design 定义目标，docs 描述现状。

## 仓库地图

```
pages/           页面入口：index(单机训练台) / fed(着陆) / host(房主) / join(节点) / webgpu-check(探测)
design/          权威设计稿 v3.1（目标态）：01 愿景 / 02 架构总纲 / 03 IR / 04 语言AI面 /
                 05 算子内核 / 06 执行引擎 / 07 学习策略 + 07b 算法目录 / 08 家族压测 /
                 09 联邦协议 / 10 去中心安全 / 11 数据与模型公地 / 12 路线图 / 13 风险开放问题 /
                 14 训练工程现实回应 / 15 产品形态与协作模型 / 16 协议与规范 / 17 Agent任务模型
                 （先读 design/README.md；文档与目录规范见其末节）
src/app/         入口装配：main / fed-main / selftest
src/gpu/         WebGPU 引擎：device/arena/buffer/pipeline + kernels/{gemm,norm,attention,rope,activation,embedding,misc}/（35 个 WGSL）
src/model/       模型图：tiny-gpt / qwen（前反向 + 绑定）
src/train/       trainer（循环+AdamW）/ data / sft
src/infer/       生成循环 / sampler
src/weights/     safetensors / GPTQ int4 反量化 / qwen-loader
src/reference/   CPU 参考实现（GPU 对拍基准）
src/fed/         公共训练网络：protocol / engine(+gpu-engine) / node(编排) / transport / bus / corpus / capability
src/tests/       页面自检注册表（kernels+model = 44 项）+ {ops,train,infer,qwen,perf}/ 分组 + fed/(无头与端到端)
scripts/         自检与工具：verify-*.mjs / signal-server / start-demo / build-standalone / lib/{cdp,bundle,cert}
docs/            权威文档（先读 docs/架构总览.md）
```

## 常用命令

```bash
npm run dev / demo / demo:https     # dev server / 一键联邦 demo（信令+dev server）/ 自签 https
npm run typecheck / build           # 静态检查 / 四页构建
npm run verify:fed                  # 联邦核心自检（Node 无头，19 项）
npm run verify:e2e                  # 三节点端到端（20 项，含抓作弊节点）
npm run verify:signal               # 信令协议（9 项，含裸 socket 心跳回归）
npm run verify:ui                   # 真实 Chromium 界面测试（20 项）
npm run verify:phone                # 真机双端联调（需手机 ADB 在线）
node scripts/verify-phone.mjs gpu|refresh|kill   # 单场景
```

## 验证矩阵（改完代码按序跑，全绿才算完）

| 改动范围 | 必跑 |
|---|---|
| 任何 `src/**` | `npm run typecheck` + `npm run build` |
| `src/fed/**` | 再加 `verify:fed`、`verify:e2e`、`verify:ui` |
| 信令/传输层 | 再加 `verify:signal` |
| 手机端行为 / 用户要求联调 | 再加 `npm run verify:phone` |
| 移动/重命名 src 文件 | 同步：html 入口、scripts 入口、`?raw` kernel 导入（`src/train/trainer.ts`、`src/model/*.ts`、`src/tests/kernels.ts`）、verify-*.mjs 的 bundle 入口 |

## 真机联调 SOP（速查，完整版 docs/真机联调指南.md）

1. 手机「开发者选项 → 无线调试」打开；`adb connect <ip>:<port>`（端口会变，重开后要重连）
2. `adb forward tcp:9222 localabstract:chrome_devtools_remote` + `adb reverse tcp:5173 tcp:5173` + `adb reverse tcp:5180 tcp:5180`
3. **手机 WebGPU 正解**：手机开 `http://localhost:5173/pages/join.html`（localhost = 安全上下文，免证书免 flags）
4. demo 服务由**用户**起（`start-demo.cmd` / `npm run demo`）；联调脚本只读不杀
5. `node scripts/verify-phone.mjs [gpu|refresh|kill]`；产物落 `demo/screenshots/phone-*`

**断开测试绝不能断 WiFi**（系统会连带关掉无线调试，ADB 全没）——用 refresh（刷新页面）
/kill（`am force-stop` 杀浏览器）场景覆盖。断线恢复可能需要用户手动重开无线调试。

## 工程不变量（违反必炸）

1. 同 compute pass 内 dispatch 间不得有对同一 buffer 的 RAW/WAR/WAW（未定义行为，BUG-001）
2. RTCDataChannel 大帧必须背压发送；`dc.send` 队列满时**同步抛异常**（BUG-008）
3. 信令服务器收到 pong 必须把连接 `alive` 复位（BUG-007）
4. 训练全程 fp32；无 fp32 atomicAdd → 权重梯度用「每输出元素唯一线程 + 规约」
5. 单 kernel 最多 4 bind group → arena + offset，不按张量绑
6. 联邦：引擎全网一致（manifest 定死）、字符表全网一致、中途加入必须 sync 对齐到当前全局权重
7. 跨设备权重比较用容差（1e-3），不做哈希对拍

## 已证实无效的手段（别再试）

- `agent-browser open`：本机永久挂起 → 用 spawn chromium + 零依赖 CDP（`scripts/lib/cdp.mjs`）
- swiftshader / `--use-angle` flags：会弄坏 headless WebGPU；只留 `--headless=new`
- 在 about:blank 探测 WebGPU：非安全上下文，结论必然错误
- `svc wifi disable` / 飞行模式做自动化断开测试：见上
- `Page.crash`、`/json/new`：本机 Edge Android 不支持；杀浏览器用 `am force-stop` + `am start`

## 文档索引（先读哪个）

| 想… | 读 |
|---|---|
| 快速建立全局理解 | docs/架构总览.md（分层/调用链/不变量/测试体系/债务清单） |
| 目标态设计（v3.1 要变成什么） | design/README.md（权威设计稿索引） |
| 设计为什么这么定 / 历史评审推导 | docs/设计评审记录-2026-10-07.md、docs/设计评审记录-2026-10-07-第二轮.md |
| 真机联调 / 手机 WebGPU | docs/真机联调指南.md |
| 查历史 bug 与教训 | docs/bug记录.md |
| 多设备共同训练的选型 | docs/共同训练方式选型.md |
| 产品规划与路线图 | docs/公共AI网络-规划.md |
