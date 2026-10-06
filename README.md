# webllm

浏览器内自研 LLM 引擎。不依赖 WebLLM / transformers.js，用 **TypeScript + WebGPU + WGSL** 从零实现训练与推理闭环，数据不出本地。

两条主线：

1. **训练** —— 从零预训练 tiny-GPT（nanoGPT 级），自研前向 + 反向 + AdamW
2. **推理** —— 既跑自训 tiny-GPT，也加载现成小模型（Qwen2.5-0.5B-Instruct-GPTQ-Int4）

## 快速开始

```bash
npm install
npm run fetch:model   # 下载模型权重（438MB，被 .gitignore 排除）
npm run dev      # 打开 http://127.0.0.1:5173
```

打开页面后自检会自动运行，逐项显示 GPU vs CPU 对拍结果。

也可以直接起**联邦联训 Demo**（会同时拉起一台零依赖的信令服务器）：

```bash
npm run demo     # 终端会打印本机地址 + 手机可用的局域网地址
```

> 不需要模型也能跑 —— tiny-GPT 的训练与自检不依赖 Qwen 权重，
> 只是「加载现成模型」相关的测试会失败。`fetch:model` 可随时补。

## 环境要求

WebGPU 支持的浏览器（Chrome / Edge 113+）。开发环境实测基线见 [docs/实施计划.md](docs/实施计划.md)，
可用 [webgpu-check.html](webgpu-check.html) 在自己的机器上复测。

## 架构

单个 kernel 最多 4 个 bind group，因此**不按张量绑定**，改用 arena + 显式 offset：

| bind group | 内容 |
|---|---|
| `group0` | 权重 arena（单一大 storage buffer，runtime-sized array + offset 索引） |
| `group1` | 激活 arena |
| `group2` | 激活 / KV 备用 arena |
| `group3` | uniform（各张量 offset、shape、超参） |

```
src/
├── gpu/          # device / arena / buffer / pipeline + kernels/*.wgsl（35 个自研算子）
├── model/        # tiny-gpt 前反向图、权重初始化、Qwen 架构
├── train/        # 训练循环、AdamW、语料 batch、SFT/DPO 指令数据
├── infer/        # 生成循环 + KV cache、temperature/top-k/top-p采样
├── tokenizer/    # Qwen BPE
├── weights/      # safetensors 解析、GPTQ int4 反量化、权重上传
├── store/        # IndexedDB checkpoint（权重 + AdamW 状态）
├── reference/    # CPU 参考实现（GPU 对拍基准）
├── fed/          # 公共训练网络：协议 / 引擎抽象 / P2P 传输 / 联邦训练节点 / 自检
└── tests/        # 算子对拍、梯度检验、SFT/DPO/checkpoint、推理对拍
```

关键约束（详细推导见实施计划）：

- **训练全程 fp32**，f16 仅走推理路径，且必须先通过 fp32 logits 回归对拍
- **无 fp32 atomicAdd**（WGSL atomic 仅整型）→ 权重梯度禁止原子累加，用「每输出元素唯一线程 + 规约」
- **2 GiB 硬顶**（`maxBufferSize` 只能降不能升）→ 多 buffer 分片 + grid 切分
- 训练放主线程但用 `await` 驱动循环，不依赖 rAF，避免标签页后台节流

## 公共训练网络（联邦联训 Demo）

[fed.html](fed.html) 是一个独立 Demo：**PC 与手机在同一个 WiFi 下打开网页，就能共同训练同一个模型**。
数据不出本地，只交换权重，全程浏览器内完成、P2P 直连。

```
Host（汇聚节点）── 开轮 / 校验 / FedAvg / 广播全局权重
   │   WebRTC DataChannel（网状直连，训练数据不经任何服务器）
Peer  Peer  Peer   各自在本地数据分片上训练，只上报权重
```

- **默认引擎是纯 JS 小模型**（embedding + 单隐层 MLP 的字符级 LM）：不依赖 WebGPU，
  因此局域网 `http://192.168.x.x` 也能跑 —— WebGPU 要求安全上下文，这一点决定了
  「手机同一 WiFi 开箱即用」能不能成立。
- **信任层 v0**：房间清单指纹（所有节点对「训什么模型、用什么数据」达成一致）
  + **共识探针复算**（主机用节点提交的权重自己算一遍 loss，与节点自报值比对，不符即剔除）
  + 贡献账本（样本数 / token 数 / Δ 范数 / 聚合份额 / 裁决，随模型卡下载）
  + 可复现（固定种子 + 确定性评估）。
- **数据两种来源**：任务提供（内置语料按分片切给各节点）+ 用户贡献
  （过滤到房间字符表后并入本地训练池，不上传）。

完整规划（三层架构、信任层路线、市场、里程碑）见
[docs/公共AI网络-规划.md](docs/公共AI网络-规划.md)。

```bash
npm run demo            # 一键起信令 + dev server，并打印手机可用地址
npm run verify:fed      # 联邦训练核心自检（16 项）
npm run verify:signal   # 信令链路自检（7 项，含跨 64KB 长帧）
```

自检关键结论（真实语料 Tiny Shakespeare，1,115,394 字符）：初始探针 loss 4.182 →
联邦训练 2.824，优于单节点独训均值 2.979；权重帧 54,300 字节逐位无损往返，篡改 1 字节即被拦下。

## 验证标准

| 层级 | 手段 | 通过标准 |
|---|---|---|
| 单算子 | GPU vs CPU 参考对拍 | rel < 1e-5 |
| 反向 | 逐层数值梯度（finite difference） | rel < 1e-3 |
| 前向整体 | 与 CPU 参考 logits 对拍 | 一致 |
| 训练 | 小语料过拟合 | loss → 0 |
| 推理 | 贪心解码与参考对拍 | 逐token一致 |

跑测试：`npm run typecheck` 做静态检查，运行时自检在页面内执行。

## 里程碑

M0 基建 → M1 前向算子 → M2 训练闭环 → M3 性能优化 → M4 加载 Qwen → M5 解码
优化（KV cache / GEMV / split-K / dispatch 合并）→ M6 后训练管线。

M0–M5 已完成；M6 已落地：**SFT 指令微调**（masked CE，loss 只计 completion）、
**IndexedDB checkpoint**（权重 + AdamW 状态无损往返）、**DPO 偏好对齐**
（冻结 reference + 逐行加权 CE 反向）。全部浏览器端，端到端实测：
SFT loss 3.5→0.06（held-out 泛化 7/8）、checkpoint 往返逐位一致、
DPO margin 0→5.1。解码吞吐从 1.2 → 10+ tokens/s，实测数据见
[docs/性能实测记录.md](docs/性能实测记录.md)，bug 档案见
[docs/bug记录.md](docs/bug记录.md)，各阶段验证方式见
[docs/实施计划.md](docs/实施计划.md)。

### 公共训练网络路线

M-N0 协议与最小房间（**已完成**）→ M-N1 WebGPU 引擎接入 → M-N2 分层拓扑与断线续训
→ M-N3 信任 v1（SHA-256 + 身份签名）→ M-N4 LoRA 引擎（浏览器参与 0.5B~3B 微调）
→ M-N5 任务市场 → M-N6 公共模型仓库 → M-N7 信任 v2/v3（鲁棒聚合 / 质押 / DP）。
详见 [docs/公共AI网络-规划.md](docs/公共AI网络-规划.md)。

## License

MIT