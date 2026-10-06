/**
 * Qwen2 BPE tokenizer（不依赖 transformers.js / tokenizers）。
 *
 * 数据来自 HuggingFace tokenizers 的 tokenizer.json：
 *   model.type = BPE，utf8 字节级映射（ByteLevel），151643 个 vocab + 22 个特殊 token。
 *
 * 三段管线：
 *   1. pre-tokenize：按 HF 的 Split 正则切分（含 Greedy-BPE 的英文缩写/数字/空白规则）
 *   2. byte-level：每个片段按 UTF-8 字节流映射为「可见 unicode 字符」形式
 *   3. BPE：按 merges 的 rank 反复贪心合并，再查 vocab 得到 token id
 *
 * 参照 Qwen2.5 tokenizer.json 实测：
 *   normalizer = NFC、pre_tokenizer = Sequence(Split, ByteLevel)、post_processor = ByteLevel
 */

/** HF `tokenizers` 中 ByteLevel 的字节→可见字符映射表（GPT-2 bytes_to_unicode）。 */
function buildByteToUnicode(): Map<number, string> {
  const bs: number[] = [];
  for (let i = 0x21; i <= 0x7e; i++) bs.push(i);
  for (let i = 0xa1; i <= 0xac; i++) bs.push(i);
  for (let i = 0xae; i <= 0xff; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const map = new Map<number, string>();
  for (let i = 0; i < bs.length; i++) map.set(bs[i], String.fromCharCode(cs[i]));
  return map;
}

const BYTE_TO_CHAR = buildByteToUnicode();
const CHAR_TO_BYTE = new Map<string, number>();
for (const [b, c] of BYTE_TO_CHAR) CHAR_TO_BYTE.set(c, b);

/**
 * Qwen2 的 pre-tokenizer Split 正则。
 *
 * 注意：HF 原文写作 `(?i:'s|'t|...)`。JS/V8 的 RegExp 不支持 `(?i:` 环视字面量（ES2025
 * modifier groups），这里把大小写不敏感提到整条 pattern 的 `i` 标志上——对本 pattern
 * 的其余各分支（`\p{L}` / `\p{N}` / `\s` / 否定字符类）语义完全等价。
 */
const SPLIT_PATTERN =
    "'s|'t|'re|'ve|'m|'ll|'d|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/** tokenizer.json 中本实现需要的子集。 */
export interface TokenizerJson {
  model: {
    type: string;
    vocab: Record<string, number>;
    merges: string[];
  };
  added_tokens?: { id: number; content: string; special: boolean }[];
}

/** 把一段 UTF-8 字节流转成 ByteLevel 的可见字符形式。 */
function bytesToVisible(bytes: number[]): string {
  let out = '';
  for (const b of bytes) out += BYTE_TO_CHAR.get(b)!;
  return out;
}

/** 把可见字符形式转回 UTF-8 字节流。 */
function visibleToBytes(text: string): number[] {
  const out: number[] = [];
  for (const ch of text) {
    const b = CHAR_TO_BYTE.get(ch);
    if (b === undefined) throw new Error(`tokenizer: 字符 ${JSON.stringify(ch)} 不在 ByteLevel 映射表内`);
    out.push(b);
  }
  return out;
}

/** UTF-8 编码（等价 TextEncoder，但避免依赖）。 */
function utf8Bytes(text: string): number[] {
  return Array.from(new TextEncoder().encode(text));
}

/** 贪心 BPE：反复找出 rank 最小的相邻 pair 并合并，直到没有可合并的 pair。 */
function applyBpe(piece: string, ranks: Map<string, number>, cache: Map<string, string>): string[] {
  const cached = cache.get(piece);
  if (cached !== undefined) return cached.split(' ');

  let symbols: string[] = Array.from(piece);

  while (symbols.length > 1) {
    let bestRank = Infinity;
    let bestIndex = -1;
    for (let i = 0; i < symbols.length - 1; i++) {
      const rank = ranks.get(`${symbols[i]} ${symbols[i + 1]}`);
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestIndex = i;
      }
    }
    if (bestIndex < 0) break;
    symbols = [...symbols.slice(0, bestIndex), symbols[bestIndex] + symbols[bestIndex + 1], ...symbols.slice(bestIndex + 2)];
  }

  cache.set(piece, symbols.join(' '));
  return symbols;
}

export interface EncodeOptions
{
    /** 是否把特殊 token 作为独立 token 匹配（默认 true）。 */
    specialTokens?: boolean;
}

/**
 * Qwen2 BPE 分词器。encode 返回 token id 数组，decode 反向还原文本。
 */
export class BpeTokenizer
{
  private readonly vocab: Map<string, number>;
  private readonly idToToken: string[];
  private readonly ranks = new Map<string, number>();
  private readonly specials: { content: string; id: number }[] = [];
  private readonly specialsPattern: RegExp | null = null;
  private readonly splitRe: RegExp;
  private readonly cache = new Map<string, string>();
  readonly vocabSize: number;

  constructor(json: TokenizerJson)
  {
    if (json.model?.type !== 'BPE') throw new Error(`tokenizer: 期望 BPE，实际 ${json.model?.type}`);

    this.vocab = new Map(Object.entries(json.model.vocab));
    for (const t of json.added_tokens ?? []) {
      if (t.id !== undefined && !this.vocab.has(t.content)) this.vocab.set(t.content, t.id);
    }

    let maxId = -1;
    for (const id of this.vocab.values()) maxId = Math.max(maxId, id);
    this.idToToken = new Array<string>(maxId + 1).fill('');
    for (const [token, id] of this.vocab) this.idToToken[id] = token;
    this.vocabSize = maxId + 1;

    json.model.merges.forEach((pair, rank) => this.ranks.set(pair, rank));

    for (const t of json.added_tokens ?? []) {
      if (t.special) this.specials.push({ content: t.content, id: t.id });
    }
    if (this.specials.length > 0) {
      const escaped = this.specials
        .map((s) => s.content.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .sort((a, b) => b.length - a.length)
        .join('|');
      // 必须带 g 标志：否则 exec 会忽略 lastIndex，在同一个匹配位置死循环。
      this.specialsPattern = new RegExp(`(${escaped})`, 'g');
    }

    // 整条 pattern 用 i 标志实现 HF 的 (?i:...) 分组；对 Split 其余分支语义等价。
    this.splitRe = new RegExp( SPLIT_PATTERN, 'giu' );
  }

  /** 从 URL 加载 tokenizer.json（默认本地 public 路径）。 */
  static async load(url = '/models/qwen2.5-0.5b-int4/tokenizer.json'): Promise<BpeTokenizer> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`tokenizer: 加载失败 HTTP ${res.status} ${url}`);
    return new BpeTokenizer(await res.json() as TokenizerJson);
  }

  /** 文本 → token id 序列。 */
  encode(text: string, options: EncodeOptions = {}): number[] {
    const specialTokens = options.specialTokens ?? true;
    const ids: number[] = [];

    const emitPiece = (piece: string): void => {
      if (piece.length === 0) return;
      const visible = bytesToVisible(utf8Bytes(piece));
      for (const symbol of applyBpe(visible, this.ranks, this.cache)) {
        const id = this.vocab.get(symbol);
        if (id === undefined) throw new Error(`tokenizer: ${JSON.stringify(symbol)} 不在 vocab 中`);
        ids.push(id);
      }
    };

    // 先切出特殊 token，其余走 BPE。
    const segments: { text: string; special: boolean }[] = [];
    if (specialTokens && this.specialsPattern) {
      let cursor = 0;
      this.specialsPattern.lastIndex = 0;
      let m: RegExpExecArray | null;
      while ((m = this.specialsPattern.exec(text)) !== null) {
        if (m.index > cursor) segments.push({ text: text.slice(cursor, m.index), special: false });
        segments.push({ text: m[0], special: true });
        cursor = m.index + m[0].length;
        if (m[0].length === 0) this.specialsPattern.lastIndex++;
      }
      if (cursor < text.length) segments.push({ text: text.slice(cursor), special: false });
    } else {
      segments.push({ text, special: false });
    }

    for (const seg of segments) {
      if (seg.special) {
        const id = this.vocab.get(seg.text);
        if (id === undefined) throw new Error(`tokenizer: 特殊 token ${seg.text} 不在 vocab 中`);
        ids.push(id);
        continue;
      }
      this.splitRe.lastIndex = 0;
      let m: RegExpExecArray | null;
      while ((m = this.splitRe.exec(seg.text)) !== null) {
        emitPiece(m[0]);
      }
    }

    return ids;
  }

  /** token id 序列 → 文本。 */
  decode(ids: number[]): string {
    const visible: string[] = [];
    for (const id of ids) {
      const token = this.idToToken[id];
      if (token === undefined || token === '') throw new Error(`tokenizer: id ${id} 不在 vocab 中`);
      visible.push(token);
    }
    return new TextDecoder().decode(new Uint8Array(visibleToBytes(visible.join(''))));
  }
}
