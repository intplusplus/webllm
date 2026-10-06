import type { GpuContext } from '../../gpu/device';
import { BpeTokenizer } from '../../tokenizer/bpe';

/** 覆盖多语言 / 空白 / emoji / 控制字节 / 长词等边界。 */
const CORPUS: string[] = [
  'hello world',
  'The quick brown fox jumps over the lazy dog.',
  '你好，世界！我是一个测试文本。',
  '深度学习的发展历程 machine learning',
  '',
  ' ',
  '  \n\t\r\n ',
  'emoji 😀🚀🎉 and math ∑∫≈≠',
  'def foo(bar):\n    return bar * 2  # comment',
  "I can't believe it's 2026-10-06 already!",
  'Qwen2.5-0.5B-Instruct 的 int4 GPTQ 量化版本。',
  'a',
  'a'.repeat(300),
  'supercalifragilisticexpialidocious unbelievably longwordsequence 1234567890',
  '\u0000\u0001\u007f control bytes 😀',
  '¿Habla español? Oui, très bien.',
];

/**
 * M4-2：Qwen2 BPE tokenizer。
 *
 * 校验三件事：
 *   1. encode→decode 往返无损（覆盖多语言 / emoji / 空白 / 控制字节）
 *   2. 特殊 token（<|im_start|> 等）按原文顺序保留，且可关闭
 *   3. token id 落在合法词表范围内、常用英文词为单 token
 */
export async function testTokenizer(_gpu: GpuContext): Promise<string> {
  const tokenizer = await BpeTokenizer.load();

  const failures: string[] = [];
  for (const text of CORPUS) {
    const back = tokenizer.decode(tokenizer.encode(text));
    if (back !== text) failures.push(`往返失败 text=${JSON.stringify(text.slice(0, 24))} got=${JSON.stringify(back.slice(0, 24))}`);
  }
  if (failures.length > 0) throw new Error(failures.join(' | '));

  // 特殊 token：按原文顺序出现，并可被关闭
  const LT = '<';
  const imStart = `${LT}|im_start|>`;
  const imEnd = `${LT}|im_end|>`;
  const eot = `${LT}|endoftext|>`;
  const idsOf = (text: string, special = true) => tokenizer.encode(text, { specialTokens: special });
  const idOf = (content: string) => {
    const ids = idsOf(content);
    if (ids.length !== 1) throw new Error(`特殊 token ${content} 未映射为单个 id：${ids.length}`);
    return ids[0];
  };
  const imStartId = idOf(imStart);
  const imEndId = idOf(imEnd);
  const eotId = idOf(eot);

  const dialog = `${imStart}system${imEnd}user${imStart}hi${imEnd}${eot}`;
  const dialogIds = idsOf(dialog);
  if (tokenizer.decode(dialogIds) !== dialog) throw new Error('含特殊 token 的文本往返失败');
  const wanted = [imStartId, imEndId, imStartId, imEndId, eotId];
  const seen = dialogIds.filter((id) => wanted.includes(id));
  if (seen.length !== wanted.length || seen.some((v, i) => v !== wanted[i])) {
    throw new Error(`特殊 token 顺序错误：got=[${seen}] want=[${wanted}]`);
  }
  const plain = idsOf(imEnd, false);
  if (plain.length <= 1 || plain.includes(imEndId)) throw new Error('specialTokens=false 时未按普通文本切分');

  // id 范围与词汇覆盖率
  const allIds = new Set<number>();
  for (const text of CORPUS) for (const id of tokenizer.encode(text)) allIds.add(id);
  for (const id of allIds) {
    if (!Number.isInteger(id) || id < 0 || id >= tokenizer.vocabSize) throw new Error(`非法 token id ${id}`);
  }

  const commonWords = ['the', 'and', 'of', 'is', 'to', 'in', 'that', 'world', 'hello'];
  const multi = commonWords.filter((w) => tokenizer.encode(w).length !== 1);
  if (multi.length > 0) throw new Error(`常用词未映射为单 token：${multi.join(',')}`);

  const sample = '深度学习的发展历程';
  const sampleIds = tokenizer.encode(sample);
  return [
    `vocab=${tokenizer.vocabSize}（151643 BPE + 22 special）；往返 ${CORPUS.length} 例全部无损`,
    `特殊 token：<|im_start|>=${imStartId} <|im_end|>=${imEndId} <|endoftext|>=${eotId}，顺序保留且可关闭`,
    `示例 "${sample}" → ${sampleIds.length} tokens [${sampleIds}]；常用英文词均为单 token`,
  ].join('；');
}
