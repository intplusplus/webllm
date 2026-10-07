# 17 · 通用 Agent 任务模型

> 前置：15 §10.2（Agent 是一等资产）、16（协议与规范）、11 §2.4（Probe）、09 §6（验证义务化）。
> 定位：设计一个**不限于特定任务**的任务模型——它是本系统「agent 层」的核心规范（对应 SPEC-5）。
> 状态：v0.1 讨论稿（2026-10-08），含**调研 + 学术推理 + 未来推演**。
> **凡标 ★待定 或 ⚠推演 的，是要一起推的地方。**

---

## 0. 一句话

**Task 是内容寻址的一等对象**：目标 + 输入 + 产出 + **验收** + 约束 + 权责 + 血统。
**人和 agent 是同构的执行者，验收即探针（可复算）。**
于是"任务"成为 AI 网络里**可组合、可复用、可验证**的原语。

## 0.1 用户原话（2026-10-08）

> "agent 任务这块也是一样，**不限于特定任务**，你帮我想想，我也没想好，
> 做一下调研，学术推理，创新思维，未来推理。"

所以本文不做"写代码的 agent"或"做研究的 agent"——
**做的是"任务本身"如何被表示、被委派、被验收、被组合的规范。**

---

## 1. 为什么先做"任务"而不是先做"agent"

| 如果先做 agent | 如果先做任务模型 |
|---|---|
| 每个场景一个 agent，互不通用 | 任务一次定义，任意 agent 可执行 |
| 能力写死在实现里 | 能力是可声明、可验证的凭证 |
| 无法比较、无法组合 | 任务有 inputs/outputs ⇒ 天然可组合 |
| 换模型要重写 agent | 换执行者不改任务定义 |

**核心判断：agent 是易变的（模型半年一换），任务是稳定的（"把 X 变成 Y 并证明做到了"千年不变）。
把稳定的那层做成规范，易变的那层做成实现。**

---

## 2. 调研：2026 的任务与委派现状

### 2.1 通信层已有"任务"语义，但很薄（A2A）

A2A v1.0 定义了标准任务生命周期状态：

```
submitted → working → (input-required) → completed
                     ↘ failed | canceled | rejected
```

**这是"传输级"状态机**：它回答"这个任务现在到哪一步了"，
**不回答**"凭什么说它完成了"、"谁有权验收"、"证据是什么"。这是关键缺口。

### 2.2 实践中被总结出的"委派合同"（ITECS 2026，11 条款）

业界（面向企业合同的实践）已经积累出一套清单，值得直接吸收：

| # | 条款 | 对应本文 |
|---|---|---|
| 1 | 任务身份与**有界目标**（含显式排除项） | `goal` + `constraints.scope` |
| 2 | 输入与上下文（源版本、假设、缺失输入行为） | `inputs` |
| 3 | 被委派方与**能力**（模型/prompt/工具/评测版本） | `executor` + `capability` |
| 4 | **权限边界**（可提议/可执行/可批准/永不） | `authority` |
| 5 | 数据限制（字段/系统/地域/留存/禁二次使用） | `constraints.dataPolicy` |
| 6 | **预算与资源上限**（token/工具/算力/重试） | `constraints.budget` |
| 7 | 截止与检查点（心跳、停滞阈值） | `constraints.deadline` |
| 8 | **验收与验证者**（可观测通过标准 + 授权验收者） | **`acceptance`** ★核心 |
| 9 | **交接包**（产物 + 血统 + 假设 + 未决风险） | `evidence` |
| 10 | **再委派与问责**（是否允许、最大深度、权限衰减） | `delegation` |
| 11 | 重试/重分配/升级/停止 | `state` 转移表 |

### 2.3 "验收合同先行 + 独立验证者"（Factory Missions）

Factory 的 Missions 把 **validation contract 写在实现之前**，且**由独立于 worker 的 validator 检查**。
⇒ 与本项目的 **Probe 闸门**（11 §2.4）**同构**。这验证了我们的方向：
**验收不该是执行者自证，而是先写好的、独立的、可执行的检查。**

### 2.4 资格闸门（cellcog 2026，8 项硬前置）

identity / capability / data access / tool authority / sensitivity / capacity / policy / evaluation。
**不达即 `no-route`，而不是"给最不差的 agent"**——这条对公共网络极其重要。

### 2.5 数据侧：机器可判的许可（Croissant 1.1 / PROV-O / ODRL / DUO）

Croissant 1.1（2026-02）已把 **PROV-O（血统）+ ODRL/DUO（用法许可）** 变成机器可判的元数据，
且明确面向"agent 自主使用数据"的场景。
⇒ 任务的 `inputs` 引用数据集时，**许可能否满足可以机器判定**，不需要人工读条款。

### 2.6 缺口结论（本文要填的）

```
已有的：传输级状态机（A2A） + 企业清单（ITECS/cellcog） + 数据侧元数据（Croissant）
缺的：  把"任务"做成【内容寻址 · 可验证 · 可组合 · 跨组织】的一等开放对象，
        并把【验收】绑定到【可复算的探针】，而不是绑定到"一份合同"或"一个人的签字"
```

**本系统填的就是这个缺口。** 差异化收敛成一句：
**别人的验收是声明（declaration），我们的验收是可复算的探针（probe）。**

---

## 3. 设计：`Task` 数据结构

```jsonc
Task := {
  taskId:      contentHash,                 // 内容寻址：任务定义本身可 fork/可引用
  goal: {
    statement:  string,                     // 有界、可观测（禁止"妥善处理"式目标）
    scope:      { in: [...], out: [...] },  // ★ 显式排除项（条款 1）
    successCriteria: [{
      id, trigger, expected, tolerance     // ★ 可被独立测试（条款 8）
    }]
  },
  inputs:  [{ role, ref: contentHash, kind: 'dataset'|'model'|'adapter'|'artifact'|'value',
              license, dataPolicy }],       // 条款 2/5；许可可机器判定（§2.5）
  outputs: [{ role, kind, schema, required }],
  acceptance: Acceptance,                   // ★★ 见 §4 —— 本设计的核心
  constraints: {
    capability:  { deviceClass, requires }, // 执行者需具备的能力
    budget:      { flops, bytes, seconds, credits, retries },
    deadline:    { dueAt, heartbeat, staleAfter, checkpoints },
    sensitivity: 'public'|'internal'|'confidential',
    regions:     [...]
  },
  authority: {                              // ★ 条款 4：agent 安全的产品化落点（15 P⑩）
    mayPropose: [...], mayExecute: [...], mayApprove: [...], mustNever: [...]
  },
  delegation: { allowed, maxDepth, maxChildren, attenuate: true },   // 条款 10
  executor:  did | 'open' | [did],          // 人 / agent / 公开招募（同构）
  verifier:  did | 'probe-only' | [did],    // ★ 条款 8：必须独立于 executor
  evidence:  [contentHash],                 // 条款 9：交接包（产物 + 血统 + 未决风险）
  provenance: { parentTask?, author, license, createdAt },           // PROV-O 剖面
  state:     'draft'|'submitted'|'accepted'|'working'|'input-required'
           | 'completed'|'failed'|'canceled'|'rejected'|'expired'    // 对齐 A2A（§2.1）
}
```

**四个关键设计点**：

1. **一切都是引用**：`inputs`/`outputs`/`evidence` 全指向 `contentHash`，任务本身只有几 KB。
2. **`goal` 必须可独立验收**：写不出 `trigger + expected` 的目标，就是**没分解够**。
3. **`authority` 显式化**（可提议 / 可执行 / 可批准 / **永不**）——这是 agent 权限的规范落点。
4. **`verifier` 必须独立**：`executor == verifier` 只在显式声明 `L0 自用` 时允许。

---

## 4. 核心创新：验收即探针（Acceptance = Probe）

**这是本文与所有现有任务协议的分水岭。**

```jsonc
Acceptance := {
  kind: 'probe' | 'deterministic' | 'human' | 'composite',
  probeRef?:  probeHash,          // 复用 11 §2.4 的 Probe（能力原子 + CI + CRN + MDE）
  thresholds: { [atomId]: { min, ci } },
  verifierCount: int,             // k-of-n 独立复算（09 §6.2 验证义务化）
  dispute: 'fork-only' | 'appeal',
  honestNote?: string             // ★ 不可自动验收的部分必须显式声明，不得假装
}
```

### 4.1 为什么这一条能统一四种任务

| 任务类型 | 验收 = 什么探针 |
|---|---|
| **训练任务** | 模型能力原子提升且**统计显著**（配对 CRN + MDE，INV-21） |
| **数据任务** | 质量 / 去重 / 去污染 / 分布配比（11 §2.3 的管线算子 + Probe） |
| **评测任务** | 本身就是一个 Probe 的执行 |
| **agent 子任务** | 确定性检查（可判定）或小探针（不可判定时降级） |

**一个机制覆盖全部**——这是"不限于特定任务"的实现方式：
**不定义"任务类型"，只定义"怎么验收"，剩下全靠组合。**

### 4.2 与"企业 SOW 清单"的本质区别

```
企业 SOW：  验收 = 一段自然语言的通过标准 + 一个授权签字的人
本设计：    验收 = 一个内容寻址的探针 + 一个可复算的判定 + 一个统计上诚实的结论
```
前者**不可复算、不可审计、不可组合**；后者**可以**，因此可以进账本、可以被第三方重放。

### 4.3 诚实边界（必须写进规范）

- **"品味类"目标无法自动验收**（"写得好看"、"有洞察力"）⇒ 必须落到 `kind:'human'`，
  且**必须在 `honestNote` 里声明"此任务不可自动验收"**——不许包装成 probe。
- 公开探针会被针对性优化（11 §2.5 的 BDC 问题）⇒ 双层评测（公开 advisory / 私有 authoritative）。

---

## 5. 任务即 DAG：分解、组合、复用

### 5.1 分解规则（借 ITECS 的风险切分）

一个有界的子任务必须：**可独立分配 · 输入明确 · 产出明确 · 无需重建隐藏推理即可验收**。
按**风险**而非流程切分：研究 ≠ 推荐 ≠ 批准 ≠ 执行 ≠ 对外沟通（后果不同就分开）。

```
大任务
 ├─ 子任务 A（研究：产出证据）        verifier = probe-only
 ├─ 子任务 B（推荐：基于 A）          verifier = 人
 ├─ 子任务 C（批准：不可委派的权责）   executor = 人（MUST）
 └─ 子任务 D（执行：基于批准的方案）   verifier = 确定性 + 探针
```

### 5.2 组合性：任务 = 函数

因为任务有明确的 `inputs`/`outputs`/`failure` 语义，**它可以被当成函数**：

```
Task<A→B> ∘ Task<B→C>  =  Task<A→C>          // 只要 B 的 schema 匹配
```

⇒ 于是"任务"可以像库函数一样被**引用、复用、组合**，而且每个组合都有**可验证的语义**。
**这是"任务市场即标准库"（§8）的技术基础。**

### 5.3 复用：任务定义进公地

`taskId = contentHash(定义)` ⇒ 任务定义本身可 fork、可署名、可进公地（15 §2.1）。
引用一个公开任务 ⇒ 自动继承它的验收标准与权责结构。

---

## 6. 四权分离：定义 / 执行 / 验收 / 负责

对应 09 §6（验证义务化）与 ITECS 的问责链（条款 10）：

| 角色 | 谁 | 可否委派 |
|---|---|---|
| **定义** | 任务的作者 | — |
| **执行** | executor（人 / agent / 开放招募） | 可（受 `delegation` 约束） |
| **验收** | verifier（**必须独立**，或 probe 自动） | **不可由执行者担任** |
| **负责** | parent 任务的所有者 | **不可委派**（"委派不转移责任"） |

**三条硬规则**：
1. `executor == verifier` 仅当显式声明 `L0 自用`（15 §3.1 的自家设备档）。
2. 递归深度 MUST 有界（`maxDepth`），否则任务树爆炸（§8 风险）。
3. 权限**逐层衰减**（`attenuate: true`）——子任务权限 ⊆ 父任务权限。

---

## 7. 能力与委派：匹配靠凭证，不靠自我介绍

**核心原则（借 cellcog 的资格闸门）**：**资格是硬前置，排在排序之前。**

```
资格闸门（不达即 no-route，绝不"退而求其次"）
  identity · capability · data-access · tool-authority · sensitivity · capacity · policy · evaluation
        ↓ 全部通过才进入排序
排序（可插拔策略，规范不规定具体权重）
  capability fit · evidence quality · reliability · context fit · capacity · cost
```

**能力 = 可验证凭证（VC）**：
```
Capability := {
  subject: did,                      // 谁
  claims: [{ taskClass, evidence: [probeHash], issuedAt, expiresAt }],  // 能做什么，凭什么
  issuer:  did,                      // 谁背书
  ceiling: { sensitivity, toolAuthority, budget }
}
```
- **能力会过期**：模型 / prompt / 工具 / 权限任一变更 ⇒ 证据失效 ⇒ 需重跑探针。
  （这直接修掉"上个月测过所以这个月也行"的常见错误。）
- **与 07 降级阶梯对接**：能力不足不是被拒绝，而是**换角色**（反传 → 前向 → 局部 → 零阶 → 仅评估 → 聚合）。

---

## 8. 未来推演（★为推演，非承诺）

> 方法说明：下面按"可预期（1–2 年）/ 中期（3–5 年）/ 远期（约 10 年）"分层，
> **越往后越确定是不确定的**，全部标注，不作为承诺（守 01 §5 非目标）。

### 8.1 可预期（近）

- **Recipe 与 Task 合并**：训练任务就是 `Task{outputs: Model, acceptance: 训练探针}`（15 §2.1 + 本文）。
  ⇒ 一条规范同时描述"训练"和"干活"。
- **任务浏览器**：像 npm 一样的任务库，但每个包**自带验收**（可复算），而非只有 README。
- **人机混编队列**：一个任务队列里，人和 agent 是同一种 worker，靠能力凭证被分派。

### 8.2 中期（3–5 年）

- **任务找人**：从"我找工具"变成"任务广播 → 合格执行者接单"（无平台 P2P 任务市场）。
- **任务即资产**：一个"写好的任务 + 验收 + 技能包"可被无限复用组合 ⇒ 出现**任务的标准库**。
- **自组织分解**：大任务被 agent 自主拆成子任务树并竞标执行（受 §6 四权分离 + 深度上限约束）。

### 8.3 远期（约 10 年，⚠推演）

- **命题**：AI 网络的原语**不是"模型"，而是"可验证的任务"**。
  模型只是任务的产物之一；网络的价值度量从"我有多少参数"变成"我能完成哪些可验证任务"。
- **推论 1**：评价体系从"基准分数"转向"任务完成记录"（可复算、防刷、跨组织可验证）。
- **推论 2**：经济从"卖算力 / 卖 token"转向"**卖可验证的结果**"（但本设计**不发币**，见 11 §4）。
- **推论 3**：能力地图——一张"可完成任务"的公共图谱，取代"模型排行榜"。

### 8.4 推演带来的风险（必须提前写下来）

| 风险 | 说明 | 已有对策 / 缺口 |
|---|---|---|
| **任务递归爆炸** | 任务可无限分解 ⇒ 组合爆炸 | `maxDepth`/`maxChildren` 硬上限（§3） |
| **验证者困境** | 没人愿意免费验收 | 验证义务化 + 旗标（09 §6.2，R-50） |
| **任务洗白** | 把有害总目标拆成看似无害的子任务 | ⚠**缺口**：跨任务的目标级检测无解，需治理层（11 §6） |
| **探针过拟合** | 针对验收探针优化而非真做事 | 双层评测 + 探针 fork 演化（11 §2.5） |
| **权责真空** | agent 自主决策造成损害时无人负责 | 四权分离 + 责任不可委派（§6），但**法律层未解决** |

> 结论：**任务模型的设计要比 agent 的设计更保守**——因为任务是可组合的，
> 一个漏洞会被组合放大。所以本文处处设"上界"（深度、预算、权限、验收独立性）。

---

## 9. 与现有协议/标准的关系

| 层 | 关系 |
|---|---|
| **A2A** | 本规范的 `taskId` 可被 A2A 的 task 引用；`state` 对齐其生命周期（§2.1） |
| **MCP** | 任务的执行过程用 MCP 调工具；任务本身不被 MCP 表达（它没有这一步） |
| **ANP** | 去中心发现与 DID 身份直接用 ANP 的能力 |
| **W3C DID/VC** | `executor`/`verifier`/`Capability.subject` 用 DID；能力用 VC |
| **W3C PROV-O** | `provenance`/`evidence` 是 PROV-O 的一个剖面 |
| **Croissant 1.1** | `inputs` 的数据集用 Croissant 描述，许可可机器判定（ODRL/DUO） |
| **W3C WoT Thing Description** | 参考其"能力描述"的写法（能力位已在 05 有对应物） |

**一句话：我们不替换它们，我们补它们上面空着的那层，并且能落在它们下层的任何一个传输上。**

---

## 10. 验收线（TASK-V）

| 编号 | 标准 |
|---|---|
| TASK-V1 | **任务可寻址**：同一定义 ⇒ 同 `taskId`；改写 ⇒ 新 `taskId`；可引用、可 fork |
| TASK-V2 | **验收可复算**：任一 `acceptance.probeRef` 可被第三方重放并得到容差内一致的判定 |
| TASK-V3 | **四权分离可强制**：构造 `executor == verifier` 的任务 ⇒ 非 L0 声明时被拒 |
| TASK-V4 | **资格闸门是硬前置**：不合格执行者即使排序最高也 MUST `no-route` |
| TASK-V5 | **深度与权限衰减**：超过 `maxDepth` 的委派被拒；子任务权限 ⊄ 父权限时被拒 |
| TASK-V6 | **组合保义**：`Task<A→B>` 与 `Task<B→C>` 组合后，B 的 schema 不匹配即被拒 |
| TASK-V7 | **诚实标注**：`acceptance.kind='human'` 的任务未被包装成自动验收（`honestNote` 必填） |
| TASK-V8 | **跨实现可执行**：同一 `taskId` 能被第二个实现（PROTO-V1）加载并推进到同一 `state` |

---

## 11. 开放问题

| # | 问题 | 我的默认立场 |
|---|---|---|
| T① | `Task` 与 15 的 `Recipe` 是合并还是分开？ | **合并**：Recipe 是 `Task` 的一个特化（输出=模型） |
| T② | 任务市场要不要（悬赏 / 竞标）？ | 先不做，只做"公开任务 + 自主接单"（15 §9） |
| T③ | 验收探针由谁定？ | 多套并存 + 一个公共基准套件（对齐 13 ⑮） |
| T④ | 人作为 executor 要不要与 agent 用同一套凭证？ | 要（同构是核心价值），但人的凭证走不同 issuer 信任链 |
| T⑤ | `authority.mustNever` 谁来审计？ | 由 verifier 抽检 + 账本留痕；**运行时强制**是缺口（§8.4） |
| T⑥ | 任务 DAG 需要全局调度器吗？ | 不需要（去中心）；调度是各节点的本地策略 |
