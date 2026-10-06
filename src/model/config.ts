/** GPT 配置（nanoGPT / GPT-2 风格：LayerNorm + GELU）。 */
export interface GPTConfig {
  vocabSize: number;
  /** 上下文长度 T */
  blockSize: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  /** Linear / LayerNorm 是否使用 bias */
  bias: boolean;
}

/** 每个注意力头的维度。 */
export function headDim(config: GPTConfig): number {
  if (config.nEmbd % config.nHead !== 0) {
    throw new Error(`nEmbd(${config.nEmbd}) 必须能被 nHead(${config.nHead}) 整除`);
  }
  return config.nEmbd / config.nHead;
}

/** M1 阶段用于对拍的小配置。 */
export const DEFAULT_CONFIG: GPTConfig = {
  vocabSize: 65,
  blockSize: 16,
  nLayer: 3,
  nHead: 4,
  nEmbd: 64,
  bias: true,
};