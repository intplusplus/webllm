/** 训练数据加载：Karpathy 的 Tiny Shakespeare 字符级语料（已镜像到本地 public/data）。 */

/**
 * 本地副本，由 Vite 静态服务提供。
 * 与 Qwen 权重同样放在 public 下：避免外网抖动导致自检不稳定。
 */
const DATA_URL = '/data/tinyshakespeare.txt';

export interface Corpus {
  /** 字符 id 序列 */
  ids: Uint32Array;
  vocab: string[];
}

/** 加载真实文本并按字符建词表；返回可复现的 id 序列。 */
export async function loadTinyShakespeare(): Promise<Corpus> {
  const res = await fetch(DATA_URL);
  if (!res.ok) {
    throw new Error(`加载训练数据失败：HTTP ${res.status} ${res.statusText} ${DATA_URL}`);
  }
  const text = await res.text();
  if (text.length < 1e5) {
    throw new Error(`语料过短：${text.length} chars`);
  }

  const vocab = [...new Set(text)].sort();
  const stoi = new Map<string, number>();
  vocab.forEach((ch, i) => stoi.set(ch, i));
  const ids = new Uint32Array(text.length);
  for (let i = 0; i < text.length; i++) ids[i] = stoi.get(text[i]) ?? 0;
  return { ids, vocab };
}
