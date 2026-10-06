import type { QwenConfig } from '../model/qwen-config';
import type { QwenGpt } from '../model/qwen';
import { sampleToken, type SamplerOptions } from './sampler';

export interface QwenGenerateOptions extends SamplerOptions
{
    /** 命中任一 id 即停止生成（不写入结果）。 */
    stopTokens?: number[];
}

export interface QwenGenerateResult
{
    /** prompt + 生成的 token。 */
    ids: number[];
    promptLength: number;
    generated: number;
    /** 生成阶段的吞吐（不含预填充）。 */
    tokensPerSecond: number;
    /** 预填充（处理整段 prompt）耗时，毫秒。 */
    prefillMs: number;
}

/**
 * Qwen 自回归生成（KV cache + 增量解码）。
 *
 * 先用一次 prefill 把整段 prompt 写进 cache，之后每步只前向 1 个 token，
 * 注意力读取整段 cache。相比每步重算全部上下文，解码阶段的计算量从 O(T) 降到 O(1)。
 */
export async function generateQwen (
    model: QwenGpt,
    config: QwenConfig,
    prompt: number[],
    maxNewTokens: number,
    options: QwenGenerateOptions = {},
): Promise<QwenGenerateResult>
{
    if ( prompt.length === 0 ) throw new Error( 'generateQwen: prompt 不能为空' );
    if ( maxNewTokens <= 0 ) throw new Error( 'generateQwen: maxNewTokens 必须为正' );
    if ( prompt.length > config.blockSize )
    {
        throw new Error( `generateQwen: prompt 长度 ${ prompt.length } 超过 blockSize=${ config.blockSize }` );
    }

    const stop = new Set( options.stopTokens ?? [] );
    const context = prompt.slice();
    const vocab = config.vocabSize;

    model.resetCache();

    // 预填充
    const prefillStart = performance.now();
    const promptLogits = model.prefill( Uint32Array.from( prompt ) );
    const promptRow = await model.readTensorSlice(
        promptLogits,
        ( prompt.length - 1 ) * vocab,
        vocab,
    );
    const prefillMs = performance.now() - prefillStart;

    // 增量解码
    const decodeStart = performance.now();
    let generated = 0;
    let next = sampleToken( promptRow, 0, vocab, options );

    while ( generated < maxNewTokens && !stop.has( next ) )
    {
        context.push( next );
        generated += 1;
        if ( generated >= maxNewTokens ) break;

        const logits = model.decodeStep( next );
        const row = await model.readTensorSlice( logits, 0, vocab );
        next = sampleToken( row, 0, vocab, options );
    }

    const decodeSeconds = Math.max( 1e-6, ( performance.now() - decodeStart ) / 1000 );
    return {
        ids: context,
        promptLength: prompt.length,
        generated,
        tokensPerSecond: generated / decodeSeconds,
        prefillMs,
    };
}

/** Qwen2.5-Instruct 的默认 system prompt。 */
export const QWEN_DEFAULT_SYSTEM = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.';

/** 特殊 token 常量（与 tokenizer.json 一致）。 */
export const QWEN_IM_START = 151644;
export const QWEN_IM_END = 151645;
export const QWEN_ENDOFTEXT = 151643;

/** 按 Qwen2.5-Instruct 的 chat 模板拼出 prompt 文本。 */
export function buildQwenChatText ( userText: string, systemText: string = QWEN_DEFAULT_SYSTEM ): string
{
    return `<|im_start|>system\n${ systemText }<|im_end|>\n<|im_start|>user\n${ userText }<|im_end|>\n<|im_start|>assistant\n`;
}
