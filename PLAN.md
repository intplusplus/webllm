# PLAN.md · 实施计划（从 demo 到真实可用）

> **本文件是实施入口。** 从哪开始、按什么顺序、每一步怎么验收，都写在这里。
>
> 三份文档分工：
> - `PLAN.md`（本文件）= **实施**：接下来做什么、怎么算做完
> - `design/` = **目标态**：要做成什么样、为什么（先读 `design/README.md`）
> - `AGENTS.md` = **操作手册**：命令、验证矩阵、工程不变量
> - `docs/` = **现状与历史**：架构总览、bug、评审记录

---

## 0. 状态（2026-10-08）

- **已删除** `prototype/`（三轮交互原型，已废弃。需要回看用 `git log -- prototype`）。
- **已验证** `tsc --noEmit` 通过、0 错误；仓库可直接继续开发。
- **已有骨架（真代码，能跑）**：
  `src/gpu/*`（WebGPU + 35 个 WGSL kernel）、`src/model/*`（tiny-gpt / qwen 前后向）、
  `src/train/*`（trainer + AdamW）、`src/infer/*`（生成）、`src/fed/*`（联邦网络，端到端可跑）、
  `pages/*`（4 页）、`scripts/verify-*.mjs`（5 套自检）。
- **还没有的**：契约层与统计层——交错分片、FedNova 步数归一化、双份额、可推导份额、
  可验账本（Merkle）、压缩 + 误差反馈、quorum+grace、降级阶梯、统计铁律。

## 1. 目标：先让「一个人 + 一台设备」真实可用

不做协作、不做公地、不做激励。**只做一件事**：一个人打开网页，能定义模型、加数据、
训练、拿到一个属于自己的模型，并且能导出。

### M1 验收（做完就是"真实可用"）

| 编号 | 用户能做的事 | 判定 |
|---|---|---|
| M1-a | 我用**一句话**或 **JSX** 定义一个模型 | AI 生成的是合法 IR；JSX ↔ IR 双向能推回 |
| M1-b | 写错时**当场告诉我错在哪、怎么修** | `infer()` 报诊断码 + 可机读 `fix`，点了 fix 报错消失 |
| M1-c | 把数据放进去 | 显示词表 / 数据指纹 / 探针窗口 |
| M1-d | **真的训练**，看到 loss 下降 | 训练 loss 与探针 loss **都**下降；曲线是真数据 |
| M1-e | 让它写一段字 | 采样自真实权重，风格像训练语料 |
| M1-f | 导出我的模型 | 架构（KB）与权重（MB）**分开**导出，权重带 `specHash` |

## 2. 任务清单（M1）

> 原则：**每条都有可跑的验收命令**。先加验收，再写实现（对拍驱动）。

| # | 任务 | 落点 | 验收 |
|---|---|---|---|
| **T1** | Spec IR 最小实现：`Model/ModelNode/TrainerSpec/Meta` + canonical 序列化 + `specHash` | `src/ir/` | 新 `verify:ir`：同 IR 必同哈希；`device/shard` 变更哈希**不变**（IR-V4） |
| **T2** | `infer()` 静态检查 + `AutoFix`（先做 10 个高频诊断码） | `src/ir/infer.ts` | 构造 10 个错误模型，全部报对 `code` + `fix`（IR-V2） |
| **T3** | `emit()`：把 IR 落到**现有 kernel 调度**上（复用 `src/gpu/*`） | `src/ir/emit.ts` | 与手写 `trainer.ts` **数值对拍**一致（容差 1e-5） |
| **T4** | 组件注册表（op 契约：props/io/caps/cost/wires） | `src/registry/` | `register.query(task, budget)` 返回裁剪后的 op 列表（LANG-V1） |
| **T5** | 数据登记：词表 / `datasetHash` / **探针先切（held-out）** | 复用并改 `src/fed/corpus.ts` | 断言：探针区间**不出现**在训练集中（INV-17） |
| **T6** | 单机训练台页面：IR 编辑 → 数据 → 训练 → 模型卡 | `pages/studio.html` + `src/ui/studio-*.ts` | `npm run verify:ui` 覆盖主路径；真机跑完一次训练 |
| **T7** | 模型卡 + 分离导出（架构 KB / 权重 MB + `specHash` 绑定） | `src/ir/modelcard.ts` | 导出→重新导入，哈希一致（PROD-V13） |
| **T8** | 本地持久化：架构库 / 权重库**分成两个库** | IndexedDB 封装 | 刷新后资产还在；**删架构 ⇒ 依赖它的权重显示不可用**（PROD-V13） |

> T1/T2 是地基，**不可跳过**（IR 一旦返工，上层全部重写，见 `design/12 §6`）。
> T3 是"能不能真训"的分水岭；T6 是"能不能用"的分水岭。

## 3. 目录规范

### 现状（保持不变，够用）

```
pages/      页面入口（html）
src/app/    入口装配
src/gpu/    WebGPU：device/arena/buffer/pipeline + kernels/（35 个 WGSL）
src/model/  模型图（tiny-gpt / qwen）
src/train/  trainer / data / sft
src/infer/  生成 / sampler
src/weights/safetensors / GPTQ / qwen-loader
src/fed/    联邦网络（protocol / engine / node / transport / bus / corpus / capability）
src/tests/  自检注册表 + 分组
scripts/    verify-*.mjs / 工具
design/     目标态设计稿（NN-主题.md，编号只增不重排）
docs/       现状与记录
```

### 新增代码的落位（M1 用得到）

| 新东西 | 放哪 | 说明 |
|---|---|---|
| Spec IR 数据模型 / 序列化 / 哈希 | `src/ir/` | 纯数据，无 GPU 依赖 |
| `infer()` / `plan()` | `src/ir/` | 静态分析，**不跑 GPU** |
| `emit()` | `src/ir/emit.ts` | 唯一允许 import `src/gpu/` 的位置 |
| 组件注册表 | `src/registry/` | op 契约 + 查询 |
| 单机训练台 UI | `src/ui/studio-*.ts` | 页面逻辑（html 放 `pages/`） |
| 本地持久化 | `src/store/` | IndexedDB 封装 |
| 自检 | `src/tests/ir/`、`src/tests/registry/` | 一文件一组，注册到自检注册表 |

★待定：是否要**移动**现有 `src/fed/`、`src/model/`（让分层更整齐）？
**建议先不动**——现有结构能跑，M1 不动它就是降低风险。要整理放到 M2 之后。

## 4. 开发约定（每次都遵守）

1. **验证矩阵**（`AGENTS.md`）：改 `src/**` 必跑 `typecheck` + `build`；改了自检要跑对应 `verify:*`。
2. **一次提交一类事**：`feat:` / `fix:` / `design:` / `docs:` / `chore:`；设计与代码分提交。
3. **先加验收，再写实现**：每个 T 先落一条 `verify:*` 用例，再写代码。
4. **不变量优先**：`design/02 §4` 的 INV-1..26 是硬约束，违反必炸。
5. **文档同步**：新增设计稿要更新 `design/README.md`（清单 + 依赖图 + 术语表）。

## 5. 怎么把这件事交给 AI 继续做

把这三句话给它，就够它自己开工：

> 1. 先读 `AGENTS.md`（操作手册）与 `design/README.md`（目标态索引）。
> 2. 按 `PLAN.md §2` 的任务清单从 **T1** 开始，**每条先写验收用例再写实现**。
> 3. 每完成一条，跑对应 `verify:*`，然后单独提交（`feat:` 前缀）。
