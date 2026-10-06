/**
 * 语料加载、字符表构建、数据分片。
 *
 * 「数据」在公共训练网络里有两个来源，本文件都覆盖：
 *   1. 任务提供的数据 —— 内置 Tiny Shakespeare（随 app 静态托管，离线可用）
 *   2. 用户自己贡献的数据 —— 经字符表过滤后并入该节点本地训练池
 *
 * 关键约束：字符表（vocab）必须全网一致，否则各节点算的 logits 维度不同、
 * 权重无法聚合。因此字符表由房间语料唯一确定，并写进 manifest 指纹。
 */
import { hashString } from './protocol';

export interface Corpus
{
  name: string;
  text: string;
  /** 有序字符表 —— 索引即 token id */
  vocab: string[];
}

const BUILTIN_URL = '/data/tinyshakespeare.txt';

/** 有序去重字符表。 */
export function vocabOf ( text: string ): string[]
{
  return [ ...new Set( text ) ].sort();
}

export function buildStoi ( vocab: string[] ): Map<string, number>
{
  return new Map( vocab.map( ( ch, i ) => [ ch, i ] ) );
}

/** 加载内置语料（Karpathy Tiny Shakespeare，镜像在 public/data）。 */
export async function loadBuiltinCorpus (): Promise<Corpus>
{
  const res = await fetch( BUILTIN_URL );
  if ( !res.ok ) throw new Error( `加载内置语料失败：HTTP ${ res.status } ${ BUILTIN_URL }` );
  const text = await res.text();
  if ( text.length < 1e5 ) throw new Error( `内置语料过短：${ text.length } 字符` );
  return { name: 'Tiny Shakespeare（任务提供）', text, vocab: vocabOf( text ) };
}

/** 把文本编码为 id 序列，词表外字符直接丢弃。 */
export function encodeTo ( text: string, stoi: Map<string, number> ): Uint32Array
{
  const out: number[] = [];
  for ( const ch of text )
  {
    const id = stoi.get( ch );
    if ( id !== undefined ) out.push( id );
  }
  return Uint32Array.from( out );
}

export interface FilterResult
{
  text: string;
  /** 被丢弃的字符数（词表外） */
  dropped: number;
}

/** 用户贡献的文本先过一遍词表，只有词表内的字符会进入训练。 */
export function filterToVocab ( text: string, stoi: Map<string, number> ): FilterResult
{
  let kept = '';
  let dropped = 0;
  for ( const ch of text )
  {
    if ( stoi.has( ch ) ) kept += ch;
    else dropped += 1;
  }
  return { text: kept, dropped };
}

/** 连续分片：把语料切成 count 份，取第 index 份（末片兜底取余）。 */
export function shardText ( text: string, index: number, count: number ): string
{
  if ( count <= 0 ) throw new Error( `shardText: count=${ count }` );
  if ( index < 0 || index >= count ) throw new Error( `shardText: index=${ index } 超出 0..${ count - 1 }` );
  const per = Math.floor( text.length / count );
  if ( per < 1024 ) throw new Error( `语料过短，无法切成 ${ count } 片` );
  const start = index * per;
  const end = index === count - 1 ? text.length : start + per;
  return text.slice( start, end );
}

export function corpusDigest ( corpus: Corpus ): string
{
  return hashString( `${ corpus.name }|${ corpus.text.length }|${ corpus.text.slice( 0, 4096 ) }|${ corpus.vocab.join( '' ) }` );
}

/** 共识探针窗口：所有节点都能读到的固定区间，用于交叉校验提交的权重。 */
export function pickProbe ( corpus: Corpus, size: number ): { offset: number; size: number }
{
  // 取语料中后段，避开开头（开头常被分片 0 独占）
  const offset = Math.max( 0, Math.floor( corpus.text.length * 0.5 ) );
  return { offset, size: Math.min( size, corpus.text.length - offset - 1 ) };
}
