# 模型定义设计 v2：JSX 组件树 + 自动微分（经真实模型压测）

> 状态：v2 定稿（2026-10-07）· 相对 v1 的变更：**放弃兼容 PyTorch**
> （历史包袱太重），改为 Web 原生的「组件树」范式；并用 2024-2026 八个
> 真实模型家族做结构压测，据此补了 6 个核心抽象（§5）。
> 适用范围：本机/单浏览器先行；联邦分发后置（§9）。

---

## 1. 设计理念

**为会写代码的人设计，用 Web 自己的范式。**

PyTorch 的 API 是二十年的沉积，兼容它只会得到四不像。Web 前端早就解决过
同一个问题——把复杂 UI 写成可组合的声明式树。模型和 UI 在结构上同构：
层级结构 + 可复用块 + 数据流动。差别只在叶子上算的是数而不是 DOM。

三粒种子：

| 种子 | 含义 |
|---|---|
| **声明式树** | 模型即组件树，结构一眼可读，diff/体检/序列化都是树遍历 |
| **组件 + props** | 可复用模型块（Attn/FeedForward/Residual…），props=超参，build 期决定参数形状 |
| **数据流** | 张量顺树流动；forward=遍历，backward=逆序遍历 |

三条铁律：

1. **Code-first**：模型是 TS 代码不是 JSON——for/条件/复用都是原生表达
2. **eager + 记录**：op 调用立即执行并记录；`backward()` 时拓扑逆序调 bwd；
   控制流随便写，调试就是 print
3. **渐进披露**：组件（组合）→ 叶原语（算子）→ `defineOp`（WGSL 逃生门），
   每层不挡路

## 2. 一次压测驱动的核心抽象（v2）

先给结论，推导过程见 §4。除 v1 的元素/组件/props 外，压测逼出六个抽象：

| 抽象 | 一句话 | 被谁逼出来 |
|---|---|---|
| **组件上下文 `ctx`** | 组件除了 props 还拿到 `ctx`：token 流、位置、相位 | Qwen 的 N-gram 嵌入要从 token 侧查表注入 |
| **执行相位 `phase`** | forward 带相位：`train / prefill / decode`，组件可按相位走不同分支 | DeepSeek CED：20 层编码器管 prefill、20 层解码器管 decode，**同一模型两种行为** |
| **总线 `bus`（provide/inject）** | 命名共享通道，跨层读写张量 | DeepSeek CSA2 跨层复用 KV；U-Net 编码器→解码器跳连 |
| **`<Scan>` 循环** | 沿轴携带状态循环的子组件 | Mamba/GDN/RWKV/LSTM 的递归结构 |
| **多分支残差流** | 残差流可加宽成 N 条并行分支，动态门控读写 | Qwen Gated Residual（4 分支）、DeepSeek mHC（双随机流形约束） |
| **外部记忆 `<Memory>`** | 带自身参数的查表注入组件 | Qwen N-gram 嵌入（51B）、DeepSeek Engram（196B 稀疏记忆） |

外加两组**叶子扩展**：CNN 叶（conv2d/pool/patchify，给扩散/世界模型）、
条件注入组件 `<AdaLn>` / `<CrossAttn>`（给扩散与世界模型）。

## 3. 构建与执行

```
JSX 元素树（纯数据对象，同 React element）
   │  build(Component, props)                      ← 一次
   ▼
① 分配参数（requires_grad 的张量进 .params()）
② 把树线性化为前向序列（深度优先）＋反向序列（逆序，可含 Scan 展开）
③ 训练：每次 forward(x, phase) 沿序列执行并记录 trace
   loss.backward() → 逆序遍历调叶 bwd → 梯度入 .grad
④ opt.step() / zeroGrad()
```

白送的优点：**树是静态的，反向序列一次编译、每步复用**（比 PyTorch eager
每步重建图还省）。无 `fp32 atomicAdd` 的约束由既有模式解决：每参数唯一写者
+ 规约 kernel；共享参数（tied embedding）的多个 grad 流在 backward 末尾按
参数 ID 归并（`sum_rows`/split-reduce 同款）。

**循环外置**：扩散采样多步、世界模型按 chunk 自回归——这些循环是**用户代码**
包在模型外面，树只描述一步。这是特性不是限制：train 与 sample 循环本来就该
用户看得见。

## 4. 设计压测：八个真实模型家族

以下全部按公开架构写成**设计草图**（非可运行代码）。每个家族：架构要点 →
组件树写法 → 暴露的缺口 → 吸收的改进。写完八个后，§5 汇总。

### 4.1 经典 GPT（基线）

```jsx
const GPT = ({ vocab, dim, layers, heads }) => (
  <>
    <Embed vocab={vocab} dim={dim} />
    {range(layers).map(i => <Block key={i} dim={dim} heads={heads} />)}
    <RMSNorm dim={dim} />
    <Head dim={dim} vocab={vocab} tie="embed" />
  </>
);
const Block = ({ dim, heads }) => (
  <Residual><Attn dim={dim} heads={heads} /><FeedForward dim={dim} /></Residual>
);
```

缺口：无。基线通过。

### 4.2 Qwen3.8-Flash-Next（2026-08，混合线性/稀疏注意力 + MoE + 门控残差 + N-gram）

架构要点：48 层 = 36 GDN（Gated DeltaNet 线性注意力）+ 12 QSA（稀疏注意力）
按 3:1 交错；MoE 512 专家 top-10；Gated Residual 把残差流加宽成 4 分支动态
门控；51B N-gram 嵌入按局部上下文查表注入；每层还能挂 Per-Layer Embedding；
Muon 优化器；MTP 多头预测。

```jsx
const FlashNext = ({ vocab, dim, layers }) => (
  <Model>
    <Embed vocab={vocab} dim={dim} />
    <Memory kind="ngram" dim={dim} />                    {/* 缺口⑥外部记忆：读 ctx.tokens 注入 */}
    {cycle([ 'gdn', 'qsa' ], { gdn: 3, qsa: 1 }).map((kind, i) => (
      <Block key={i} kind={kind} dim={dim} branches={4} />   {/* 缺口⑤残差流 4 分支门控 */}
    ))}
    <Memory kind="ple" />                                {/* Per-Layer Embedding：同⑥ */}
    <Mtp heads={3}>                                       {/* MTP：多头 loss，训练期生效 */}
      <Head dim={dim} vocab={vocab} tie="embed" />
    </Mtp>
  </Model>
);
const Block = ({ kind, dim, branches }) => (
  <Residual branches={branches} gate="dynamic">          {/* 缺口⑤：Gated Residual/mHC 的容器 */}
    {kind === 'gdn'
      ? <GatedDeltaNet dim={dim} />                      {/* Scan 内实现，见 4.5 */}
      : <QsaSparseAttn dim={dim} blocks={microBlock} />}  {/* 叶原语：索引+选块+精确注意力 */}
    <Moe experts={512} top={10}><FeedForward dim={dim} /></Moe>
  </Residual>
);
// 训练：矩阵参数走 Muon，向量/门控走 AdamW（缺口⑨ 优化器注册表）
const opt = [ muon(model.matrices()), adamw(model.vectors()) ];
```

**缺口与改进**：⑤ 残差流宽度与门控是组件契约的一部分，不是特例；
⑥ 模型需要**不经过主残差流**的第二输入通路（查表注入），这就是 `<Memory>`；
`cycle()` 小助手表达"3:1 交错"；⑨ 优化器按参数分组注册。

### 4.3 DeepSeek-V4.1-Flash（2026-09，因果编码器-解码器 + 跨层 KV 复用 + Engram）

架构要点：552B MoE（384 路由专家 top-6 + 1 共享）+ 196B Engram 稀疏记忆；
CED 非对称：20 层因果编码器 + 20 层解码器，**prefill 只激活 8B、decode 激活
16B**（同一模型按执行相位走不同层）；每层混合全局注意力 + 滑动窗注意力
（前 2 层仅 SWA）；CSA2：跨层复用 KV + FP4 KV + 层级稀疏索引；mHC 残差；
Muon + MTP。

```jsx
const V41Flash = ({ vocab, dim }) => (
  <Model>
    <Embed vocab={vocab} dim={dim} />
    <Memory kind="engram" />                              {/* ⑥：196B 稀疏记忆，按需查 */}
    {/* ②/①：相位决定走哪套层 —— prefill 走编码器，decode 走解码器 */}
    {ctx.phase === 'prefill'
      ? range(20).map(i => <EncLayer key={i} dim={dim} swa={i < 2} />)
      : range(20).map(i => <DecLayer key={i} dim={dim} />)}
    <Norm dim={dim} />
    <Head dim={dim} vocab={vocab} />
  </Model>
);
const EncLayer = ({ dim, swa }) => (
  <Residual manifold="doublyStochastic">                 {/* ⑤：mHC 流形约束=残差参数化 */}
    <Attn dim={dim} kind={swa ? 'swa' : 'global+swa'} />
    <Moe experts={384} top={6} shared={1}><FeedForward dim={dim} /></Moe>
  </Residual>
);
const DecLayer = ({ dim }) => (
  <>
    <Residual><Attn dim={dim} kind="global" /></Residual>
    <Residual>
      <CrossAttn src={<Inject bus="enc-out" />} />        {/* ③总线：跨层/跨栈读取 */}
    </Residual>
    <Residual><Moe experts={384} top={6} shared={1}><FeedForward dim={dim} /></Moe></Residual>
  </>
);
// CSA2 跨层 KV 复用：编码器层把 KV 写进总线，解码器层按需读同一份
const EncLayerWithBus = (p) => (
  <Provide bus="kv-shared"><EncLayer {...p} /></Provide>
);
const DecLayerWithBus = (p) => (
  <DecLayer {...p} kv={<Inject bus="kv-shared" layer={-2} />} />   {/* 读倒数第 2 层的 KV */}
);
```

**缺口与改进**：② **执行相位**是一等概念（CED 的 prefill/decode 不对称、
MTP 只在训练期、dropout 只在训练期，都归它管）；③ **总线**让"跨层 KV 复用"
从打破树结构的特例变成正交抽象——这正是 React Context / Vue provide-inject
的老问题老解法；⑤ mHC 也只是残差组件的一种参数化。

### 4.4 扩散模型 DiT / 视频扩散（2024-2026 主流生成底座）

架构要点：DiT = patch 化（CNN）+ N 个 DiT 块（注意力 + MLP，条件经
AdaLN-Zero 注入时间步/类别）；VAE 把图像压到潜空间（CNN 编解码器）；
视频扩散（Wan/HunyuanVideo/LTX）再加时空注意力与文本交叉注意力。

```jsx
// 潜空间扩散：树=去噪器的一步；采样循环外置（用户代码）
const DiT = ({ patch, dim, layers }) => (
  <>
    <PatchEmbed patch={patch} dim={dim} />                {/* ⑦ CNN 叶：卷积切 patch */}
    {range(layers).map(i => (
      <DiTBlock key={i} dim={dim}>
        <AdaLn cond={tEmbed} />                           {/* ⑦ 条件注入：调制 scale/shift/gate */}
      </DiTBlock>
    )}
    <Unpatchify patch={patch} />
  </>
);
const DiTBlock = ({ dim, children }) => (
  <Residual>
    <Attn dim={dim} modulate={children} />                {/* AdaLn 调制注入注意力 */}
    <FeedForward dim={dim} modulate={children} />
  </Residual>
);
// 训练循环（用户代码，循环外置）：
for (const [x0, cond] of loader) {
  const [x_t, t, noise] = addNoise(x0);
  const loss = mse(model(x_t, t, cond), noise);           // 一次 forward 一个 trace
  await loss.backward();
}
// VAE（CNN + U-Net 跳连）—— ③总线的第二个用例：
const VAE = () => (
  <Model>
    <Provide bus="skips">
      <ConvNet blocks={[[64,2],[128,2],[256,2]]} on={['e1','e2','e3']} />  {/* 下行每级 provide */}
    </Provide>
    <ConvNet blocks={[[256],[128],[64]]} skip="skips" />                    {/* 上行同级 inject 拼接 */}
  </Model>
);
```

**缺口与改进**：⑦ CNN 叶 + 条件注入组件（AdaLn/CrossAttn）——扩散与语言
模型共享同一棵树；U-Net 跳连再次证明总线抽象的必要性；**循环外置**原则
（采样 T 步、按 chunk 自回归）写成明文。

### 4.5 循环/线性注意力：Mamba-2 / GDN / RWKV / xLSTM / LSTM

架构要点：Mamba-2（SSD 矩阵态多头）、Gated DeltaNet（Qwen 的 GDN：增量规则
+ 显式覆写）、RWKV-7（时间混合+通道混合+状态）、xLSTM（矩阵态线性注意力
mLSTM + 指数门控 sLSTM）；混合模型（Samba/Nemotron-H/Kimi Linear/Jamba）=
多数线性层 + 少数全注意力层。

```jsx
// 循环=组件；eager trace 天然支持 BPTT；优化路径=fused scan 叶原语
const Mamba2 = ({ dim, layers }) => (
  <>
    <Embed />
    {range(layers).map(i => <MambaBlock key={i} dim={dim} />)}
    <Head />
  </>
);
const MambaBlock = ({ dim }) => (
  <Residual>
    <GatedDeltaNet dim={dim} heads={8} state={64} />      {/* 叶：内部=Scan+chunk 矩阵态 */}
    <FeedForward dim={dim} />
  </Residual>
);
// 通用递归原语（xLSTM 的 sLSTM / 自定义 cell 直接用它）：
const XLstmBlock = ({ dim }) => (
  <Residual>
    <Scan axis="t" state={[h0, c0]}>                      {/* ④Scan：沿时间轴携状态循环 */}
      {(x_t, [h, c]) => [mLstm(x_t, h, c), [h1, c1]]}     {/* 子函数=单步计算，照样被 trace */}
    </Scan>
    <FeedForward dim={dim} />
  </Residual>
);
// 混合层（3:1）与 Qwen 同款 cycle([...])
const HybridLM = ({ layers }) => (
  <>{cycle(['mamba', 'attn'], { mamba: 3, attn: 1 }).map((kind, i) =>
    <Block key={i} kind={kind} />)}</>
);
```

**缺口与改进**：④ `<Scan>`（Flax scan / Keras RNN 的 Web 版）——递归模型
从"特例"变成"一个组件"。两个实现档：**解释档**（Scan 就是 traced for 循环，
BPTT 自动成立，先跑通）与**融合档**（注册 fused selective-scan 叶原语提性
能）。eager+trace 在这里体现红利：不需要为递归单独设计任何机制。

### 4.6 传统 RNN（LSTM/GRU）—— 最古老的模型

```jsx
const LSTM = ({ dim }) => (
  <>
    <Embed />
    <Scan axis="t" state={[h0, c0]}>
      {(x_t, [h, c]) => {
        const g = matmul(cat([x_t, h]), Wg);             // 四个门一次矩阵乘
        const [i, f, o, g_] = split(g, 4);
        const c1 = sigmoid(f) * c + sigmoid(i) * tanh(g_);
        return [tanh(c1) * sigmoid(o), [h1, c1]];
      }}
    </Scan>
    <Head />
  </>
);
```

缺口：无新增。证明 Scan 对最古老的模型同样自然。

### 4.7 传统 CNN（ResNet/ConvNeXt）—— 非序列模型

```jsx
const ResNet = () => (
  <Model>
    <Conv2d kernel={7} stride={2} out={64} />             {/* ⑦ CNN 叶 */}
    {range(4).map((s, i) => <Stage key={i} out={64 * 2 ** i} stride={s === 0 ? 1 : 2} />)}
    <Pool kind="avg" />
    <Linear out={1000} />
  </Model>
);
const Stage = ({ out, stride }) => (
  <>{range(2).map(i =>
    <Residual key={i}>{i === 0
      ? <Conv2d kernel={3} stride={stride} out={out} />   {/* 首块降采样 */}
      : <Conv2d kernel={3} out={out} />}</Residual>)}</>
);
```

缺口：无新增。证明树形抽象与"序列模型"无关，conv/pool 只是另一种叶。

### 4.8 交互式视频世界模型（minWM / BiWM / AlayaWorld 路线）

架构要点：双向视频扩散底座 → 控制（相机/动作）微调 → 因果/双向自回归
分块 rollout（chunk≈1.3s）→ 少步蒸馏（DMD/consistency，4 步去噪）→ 实时
交互；滑窗历史条件 + 长期记忆；潜空间 VAE + DiT 主干 + 文本/动作交叉注意力。

```jsx
// 世界模型 = DiT 主干 + 动作条件 + 历史窗口 + 记忆；chunk 循环外置
const WorldModel = ({ dim }) => (
  <>
    <Vae />                                               {/* 潜空间（CNN，复用 4.7/4.4） */}
    <DiTBackbone dim={dim}>                               {/* 4.4 的 DiT，条件多加两路 */}
      <AdaLn cond={tEmbed} />                             {/* 去噪时间步 */}
      <CrossAttn cond={textEmbed} />                      {/* 文本 */}
      <CrossAttn cond={actionEmbed} />                    {/* 动作/相机轨迹（离散动作→嵌入） */}
      <CrossAttn cond={<Inject bus="history" />} />        {/* ③总线：滑窗历史 latent */}
    </DiTBackbone>
    <Memory kind="longterm" />                            {/* ⑥：分钟级记忆 */}
  </>
);
// 实时 rollout（用户代码；少步蒸馏后每 chunk 4 步去噪）
let world = init();
for (const chunk of chunks) {
  const latents = denoiseSteps(world.latents, 4, { action: chunk.action, text: world.prompt });
  world = world.step(latents);                            // 滑窗更新 + 记忆写入
}
// 少步蒸馏训练：teacher N 步 → student 4 步，loss 照常 backward
```

缺口：无新增——它是前七个家族的自然**复合**。这正是组件树最好的证明：
世界模型 = VAE + DiT + AdaLn + CrossAttn + Memory + 外置 rollout 循环，
没有一样新机制。

## 5. 压测固化的改进清单（v1 → v2）

| # | 改进 | 来源 | v1 状态 |
|---|---|---|---|
| ① | 执行相位 `ctx.phase`（train/prefill/decode） | DS V4.1 CED、MTP、dropout | ❌ 无 |
| ② | 组件上下文 `ctx`（token 流/位置/相位） | Qwen N-gram 从 token 侧查表 | ❌ 无 |
| ③ | 总线 `bus`（provide/inject 跨层张量） | DS CSA2 跨层 KV 复用、U-Net 跳连 | ❌ 无 |
| ④ | `<Scan>` 循环组件（含融合叶原语逃生门） | Mamba/GDN/RWKV/xLSTM/LSTM | ❌ 无 |
| ⑤ | 多分支残差流 + 门控 + 参数化（mHC/GR） | Qwen Gated Residual、DS mHC | ❌ 无 |
| ⑥ | `<Memory>` 外部记忆组件（ngram/engram/longterm） | Qwen 51B N-gram、DS 196B Engram | ❌ 无 |
| ⑦ | CNN 叶 + 条件注入组件（AdaLn/CrossAttn） | DiT/视频扩散/世界模型 | ❌ 无 |
| ⑧ | `cycle()` 层型交错、`tie`/`Mtp` 等小组合子 | Qwen 3:1 混合、MTP | ❌ 无 |
| ⑨ | 优化器注册表 + 参数分组（Muon for matrices / AdamW for rest） | DS/Qwen 全面转向 Muon | ❌ 无 |
| ⑩ | 循环外置原则成文（扩散采样、chunk rollout） | 世界模型/扩散 | 隐含 |

v1（PyTorch 对齐 op API）的整体方向被推翻重写为组件树；不变的只有
eager+trace 的 autodiff 内核与双层融合的性能策略。

## 6. 通用性 / 易用性自评（诚实打分）

| 维度 | 评分 | 依据 |
|---|---|---|
| 结构通用性 | **高** | 8 个家族全覆盖，无一需要"引擎特例"；新家族=新组件组合 |
| 控制流表达 | **高** | for/条件/Scan 原生；循环外置原则明确 |
| 跨层/侧路连接 | **高** | 总线抽象覆盖 KV 复用、跳连、条件注入、记忆四类 |
| 上手成本（会 React/TS 的人） | **低** | 三个概念：组件/props/ctx；其余全在叶子里 |
| 上手成本（只会 PyTorch 的人） | **中低** | 心智可迁移（组件≈Module、props≈__init__ 超参、ctx≈forward 额外参数） |
| 性能可控性 | **中** | 叶原语=已融合 kernel；**新算子仍需 WGSL 工作**（逃生门存在但不免费） |
| MoE/稀疏路由效率 | **中低** | 表达免费，高效 all-to-all/专家并行是系统工程（联邦下还要跨设备） |
| 超大规模记忆（Engram 51B/196B） | **中低** | `<Memory>` 表达自由，主机 offload/异步预取是部署层硬活 |

明确宣称的边界：本设计解决"**把模型写出来、训起来**"的表达问题；
不解决极致的推理系统问题（FP4 KV、投机解码、专家 offload）——后者属于
部署层，按需以融合叶原语/部署配置渗入。

## 7. 不变量与性能验收

继承并强化既有铁律：

1. 训练全程 fp32；无 fp32 atomicAdd → 每参数唯一写者 + 规约
2. 同 pass dispatch 禁 RAW/WAR/WAW；大帧发送必须背压（BUG-008）
3. arena + offset 绑定；单 kernel ≤4 bind group
4. 跨设备权重比较用容差（1e-3）

新增验收线（实现 Ph.3 时执行）：

| 项 | 标准 |
|---|---|
| tiny-GPT 重写为组件树 | 与手写版**数值对拍一致** + 训练 loss 曲线重合 |
| 训练吞吐 | 相对手写 trainer **退化 <10%**（否则融合策略错） |
| Scan 解释档 BPTT | 与手写循环反向**对拍一致** |
| 新家族接入成本 | 每个新架构 ≤100 行组件代码 + 0 行 WGSL（除真正的新算子） |

## 8. 联邦扩展点（后置，接口预留）

- `manifest.modelCode`（源码）+ `codeHash` 进指纹；全网同码 → 形状一致 →
  FedAvg 数学不变；共识探针/账本照常
- host 经 DataChannel 分发代码（几十 KB）；URL/内容寻址后续
- `<Lora rank={r} target={Block}/>` 作为包装组件：联邦只传 adapter 参数
  （`model.loraParams()`），通信量降一个量级
- 安全债务：用户代码 eval = 同页执行；对外开放前 worker 沙箱 + CSP

## 9. 实现计划（每步独立验收）

| 阶段 | 内容 | 验收 |
|---|---|---|
| P1 叶原语 | 现有 kernels 包成 op（matmul/gelu/silu/ln/rms/embed/ce/add/softmax/attn-tile）+ `defineOp` 契约 | 逐 op 对拍 reference/ + gradcheck |
| P2 JSX 运行时 | 元素即数据；`build()` 两阶段（参数分配+反向序列编译）；ctx/phase/bus/Scan/interpreter | Node 侧可测；一棵示例树走通 forward+backward |
| P3 组件库 | Embed/Attn/FeedForward/Residual(多分支)/Moe/Memory/AdaLn/CrossAttn/Conv2d/Scan；GPT 参考实现 | **与 tiny-gpt 对拍一致 + 吞吐退化 <10%** |
| P4 训练接线 | loader/ce/optimizer 注册表(Muon+AdamW)/循环外置示例；页面自检新增"组件树模型"项 | 44 项旧自检不回归；新项 PASS |
| P5 家族样板 | §4 的 8 个家族各一个可跑 small 版（对拍 or 收敛曲线） | 每家族一个自检项 |
| P6 清债 | 删 `trainer.ts` 手写反向；tiny-gpt/qwen 迁组件树 | verify 全家桶 + 真机 verify:phone 全绿 |

## 10. 风险与开放问题

| 风险 | 缓解 |
|---|---|
| 组件树抽象打散融合、吞吐退化 | 双层融合（叶=已融合）；P3 验收线 <10% 卡死 |
| Scan 解释档 BPTT 显存/速度差 |  fused scan 叶原语按需下沉；显式 state 保留策略 |
| 总线跨层读写失控（乱耦合） | bus 为只读快照 + 显式 Provide/Inject 拓扑检查（build 期报环） |
| WGSL 逃生门把"易用"打穿 | 文档明示：新算子是「贡献」不是「使用」；组件库覆盖 90% |
| MoE/扩散家族性能上限 | 明确只做表达层；部署层优化单独立项 |
| 用户代码 eval 安全 | 自有设备默认关；开放前 worker+CSP |

开放问题（P2 前拍板）：① `state_dict()` 与 checkpoint/权重交换共用编解码？
② 动态 shape（变长 batch/变 chunk）在 op 契约的表达？③ bus 的快照语义
（注册时固定 vs forward 时取最新）？④ Scan 的 chunked/fused 档 API 形状？

---

## 附录：压测对象（2024-2026 公开架构）

| 家族 | 代表 | 架构要点（本设计吸收处） |
|---|---|---|
| 经典 LLM | GPT-2 / LLaMA / tiny-gpt | 基线（§4.1） |
| MoE + 混合注意力 + KV 压缩 | DeepSeek-V4.1-Flash（2026-09，MIT） | CED 相位①、CSA2 跨层 KV③、mHC⑤、Engram⑥、384+1 MoE |
| 混合线性/稀疏 + 门控残差 + N-gram | Qwen3.8-Flash-Next（2026-08） | 36GDN+12QSA 交错④⑤、Gated Residual 4 分支⑤、51B N-gram⑥、512 top-10 MoE |
| 线性注意力/SSM 混合 | Mamba-2 / GDN / RWKV-7 / xLSTM；Samba/Nemotron-H/Kimi Linear/Jamba | Scan④、矩阵态叶、cycle 交错⑧ |
| 传统 RNN | LSTM/GRU | Scan④（§4.6） |
| 传统 CNN | ResNet/ConvNeXt | CNN 叶⑦（§4.7） |
| 扩散生成 | DiT / SD3 / Wan2.x / HunyuanVideo / LTX | AdaLn/CrossAttn⑦、循环外置⑩ |
| 交互式世界模型 | minWM（2026-05）/ BiWM（2026-06）/ AlayaWorld（2026-07）/ HY-World 2.0 / Cosmos | 复合：DiT+CrossAttn(动作)+Memory+总线(历史)+chunk rollout |
