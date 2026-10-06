/**
 * Qwen2 架构配置。
 *
 * 与 GPT-2 的差异（这些差异决定了 kernel 选择）：
 *   - RMSNorm 代替 LayerNorm
 *   - RoPE（half-split）代替可学习位置嵌入；theta 为 1e6（长上下文）
 *   - GQA：query 头数 > kv 头数
 *   - SwiGLU MLP（gate/up/down 三个投影，silu 激活）
 *   - q/k/v 有 bias，o_proj 与 MLP 无 bias
 *   - tie_word_embeddings：lm_head 与 embed_tokens 共享
 */
export interface QwenConfig
{
    vocabSize: number;
    /** 支持的最大上下文长度 T。 */
    blockSize: number;
    nLayer: number;
    /** query 头数。 */
    nHead: number;
    /** kv 头数（GQA），nHead % nKvHead == 0。 */
    nKvHead: number;
    /** hidden size。 */
    nEmbd: number;
    /** SwiGLU 中间层维度。 */
    intermediate: number;
    rmsEps: number;
    ropeTheta: number;
}

/** 每个注意力头的维度。 */
export function qwenHeadDim ( config: QwenConfig ): number
{
    if ( config.nEmbd % config.nHead !== 0 ) throw new Error( `nEmbd(${ config.nEmbd }) 必须能被 nHead(${ config.nHead }) 整除` );
    return config.nEmbd / config.nHead;
}

/** kv 投影的输出维度。 */
export function qwenKvDim ( config: QwenConfig ): number
{
    return qwenHeadDim( config ) * config.nKvHead;
}

/** Qwen2.5-0.5B-Instruct（含 GPTQ-Int4 量化版）的架构参数，来自其 config.json。 */
export const QWEN25_05B: QwenConfig = {
    vocabSize: 151936,
    blockSize: 128,
    nLayer: 24,
    nHead: 14,
    nKvHead: 2,
    nEmbd: 896,
    intermediate: 4864,
    rmsEps: 1e-6,
    ropeTheta: 1000000,
};
