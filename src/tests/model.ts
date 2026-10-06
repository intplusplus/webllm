import type { GpuContext } from '../gpu/device';
import { checkTolerance, makeRng } from '../reference/cpu-ref';
import { gptForwardRef } from '../reference/gpt-ref';
import { DEFAULT_CONFIG } from '../model/config';
import { initWeights } from '../model/init';
import { TinyGpt } from '../model/tiny-gpt';
import type { SelfTest } from './kernels';
import { testGradientCheck } from './ops/gradcheck';
import { testTraining } from './train/train';
import { testPretrain } from './train/pretrain';
import { testGenerate } from './infer/generate';
import { testGptForwardF16 } from './ops/f16-logits';
import { testPerformanceBench } from './perf/bench';
import { testWeightsFormat } from './qwen/weights';
import { testQwenWeightFormat } from './qwen/qwen-format';
import { testTokenizer } from './qwen/tokenizer';
import { testQwenArch, testQwenKvCache } from './qwen/qwen-arch';
import { testQwenInfer } from './infer/qwen-infer';
import { testSft, testCheckpoint, testDpo } from './train/posttrain';

/**
 * M1 关键验证：完整 GPT 前向的 GPU logits 与独立 CPU 参考实现逐元素对拍。
 * 若任一算子或拼接关系（布局 / 残差 / 权重顺序）有误，这里必然偏差。
 */
async function testGptForward(gpu: GpuContext): Promise<string> {
  const baseConfig = DEFAULT_CONFIG;
  const B = 2;
  const T = 8;
  const M = B * T;

  const rng = makeRng(999);
  const tokens = new Uint32Array(M);
  for (let i = 0; i < M; i++) tokens[i] = Math.floor(rng() * baseConfig.vocabSize);

  // 逐层加深二分：0 层只验 embedding+layernorm+lm_head；1 层加 attention+mlp；以此类推。
  const lines: string[] = [];
  let allOk = true;
  for (const nLayer of [0, 1, 2, 3]) {
    const config = { ...baseConfig, nLayer };
    const weights = initWeights(config, 20241005);
    const model = new TinyGpt(gpu, config, weights);
    const got = await model.readTensor(model.forward(tokens, B, T));
    const expected = gptForwardRef(weights, config, { tokens, B, T }).logits;
    const r = checkTolerance(got, expected, 1e-3, 1e-2);
    if (!r.ok) allOk = false;
    lines.push(
      `nLayer=${nLayer}: maxAbs=${r.maxAbs.toExponential(2)} maxRel=${r.maxRel.toExponential(2)}${r.ok ? '' : ' FAIL'}`,
    );
  }

  if (!allOk) throw new Error('逐层偏差：' + lines.join(' | '));
  return `B=${B} T=${T} C=${baseConfig.nEmbd} vocab=${baseConfig.vocabSize}；${lines.join(' | ')}`;
}

export const modelTests: SelfTest[] = [
  { name: 'tiny-GPT 前向 logits（GPU vs CPU 参考）', run: testGptForward },
  { name: '梯度检验（数值 vs 解析）', run: testGradientCheck },
  { name: '训练闭环（小语料过拟合 loss→0）', run: testTraining },
  { name: '真实数据预训练监测（Tiny Shakespeare）', run: testPretrain },
  { name: '自回归采样生成（GPU 前向 + sampler）', run: testGenerate },
  { name: 'f16 GEMM logits 回归（fp32 vs f16）', run: testGptForwardF16 },
  { name: 'M3 性能基线（fp32 vs f16 推理吞吐）', run: testPerformanceBench },
  { name: 'M4-1 safetensors + int4 反量化', run: testWeightsFormat },
  { name: 'M4 真实 int4 权重格式探测（Qwen2.5-0.5B GPTQ）', run: testQwenWeightFormat },
  { name: 'M4-2 Qwen2 BPE tokenizer（encode/decode 往返）', run: testTokenizer },
  { name: 'M4-3 Qwen2 架构端到端对拍（GPU vs CPU 参考）', run: testQwenArch },
  { name: 'M5 KV cache 增量解码（vs 无 cache 全量前向）', run: testQwenKvCache },
  { name: 'M4-4 真实 Qwen2.5-0.5B int4 加载与推理', run: testQwenInfer },
  { name: 'M6-1 SFT 指令微调（masked CE + 过拟合 + held-out 生成）', run: testSft },
  { name: 'M6-2 IndexedDB checkpoint 往返（logits 逐位一致）', run: testCheckpoint },
  { name: 'M6-3 DPO 偏好对齐（margin 上升 + 生成不回退）', run: testDpo },
];