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
└── tests/        # 算子对拍、梯度检验、SFT/DPO/checkpoint、推理对拍
```

关键约束（详细推导见实施计划）：

- **训练全程 fp32**，f16 仅走推理路径，且必须先通过 fp32 logits 回归对拍
- **无 fp32 atomicAdd**（WGSL atomic 仅整型）→ 权重梯度禁止原子累加，用「每输出元素唯一线程 + 规约」
- **2 GiB 硬顶**（`maxBufferSize` 只能降不能升）→ 多 buffer 分片 + grid 切分
- 训练放主线程但用 `await` 驱动循环，不依赖 rAF，避免标签页后台节流

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

## License

MIT