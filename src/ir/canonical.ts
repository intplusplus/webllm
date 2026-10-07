/**
 * 规范序列化（canonical）与 specHash 投影（design/03-IR与类型系统.md §4）。
 *
 * 联邦按"同哈希 = 同结构"聚合，所以序列化必须是规范的：
 *   1. Key 排序（所有 map 按字典序）；
 *   2. 无空白（紧凑 JSON）；
 *   3. 去不确定字段（meta.created、注释、无 id 的手写顺序）；
 *   4. 归一化（默认值不写入）；
 *   5. **后端无关**（device/shard/precision 不进 specHash，否则换设备就断了聚合，INV-11）。
 *
 * 第 3/5 条不是洁癖：Marin 的实证是"把环境特定信息混进寻址 ⇒ 同一份东西在不同机器
 * 上得到不同身份 ⇒ 内容寻址/去重/跨设备复用全部失效"（03 §4.1）。
 */

import type { Annot, Model, ModelNode, OpRef } from './types';
import { contractFingerprint, type OpRegistry } from './op';

// ---------------------------------------------------------------------------
// 通用 canonical JSON
// ---------------------------------------------------------------------------

/** 深度排序 key、丢弃 undefined 的值，返回可 JSON.stringify 的纯对象。 */
export function sortDeep ( value: unknown ): unknown
{
  if ( Array.isArray( value ) ) return value.map( sortDeep );
  if ( value !== null && typeof value === 'object' )
  {
    const src = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for ( const k of Object.keys( src ).sort() )
    {
      const v = src[ k ];
      if ( v === undefined ) continue;
      out[ k ] = sortDeep( v );
    }
    return out;
  }
  return value;
}

/** 紧凑 + 排序 key 的 canonical JSON。 */
export function canonicalJson ( value: unknown ): string
{
  return JSON.stringify( sortDeep( value ) );
}

// ---------------------------------------------------------------------------
// 归一化：剔除默认值与不确定字段
// ---------------------------------------------------------------------------

function normalizeAnnot ( a: Annot | undefined ): Annot | undefined
{
  if ( !a ) return undefined;
  const out: Annot = {};
  // phase：语义字段，保留（逆时针不影响语义）；但"未声明"=默认，不写。
  if ( a.phase !== undefined ) out.phase = a.phase;
  // device / shard / precision：**进哈希会破坏跨设备聚合**（INV-11、IR-V4）
  // —— 它们都是"环境/执行策略"，不是语义。03 §4.1 把 precision 与 device 并列。
  if ( a.recompute === true ) out.recompute = true; // 默认 false 不写
  // effects：默认 pure 不写（03 §4 第 4 条）。
  if ( a.effects === 'effectful' ) out.effects = 'effectful';
  if ( a.cost !== undefined ) out.cost = a.cost;
  return Object.keys( out ).length > 0 ? out : undefined;
}

function normalizeNode ( n: ModelNode ): Record<string, unknown>
{
  const out: Record<string, unknown> = {
    id: n.id,
    op: n.op,
    props: n.props ?? {},
  };
  if ( n.children && n.children.length > 0 ) out.children = [ ...n.children ];
  if ( n.wiring && n.wiring.length > 0 ) out.wiring = n.wiring;
  if ( n.slot && Object.keys( n.slot ).length > 0 ) out.slot = n.slot;
  const annot = normalizeAnnot( n.annot );
  if ( annot ) out.annot = annot;
  // meta（note/span）是注释，不进哈希（03 §4 第 3 条）。
  return out;
}

/**
 * 产出用于求哈希的规范化模型投影：
 *   - 剔除 meta.created / id_suffix / 注释；
 *   - 剔除全部 annot.device / annot.shard（INV-11）；
 *   - 归一化默认值；
 *   - 节点表按 id 字典序（canonicalJson 自动完成）。
 *
 * 注意：**不含** trainer（strategy 与 schemaVersion 单独拼接，见 specHashInput）。
 */
export function canonicalModelProjection ( model: Model ): string
{
  const nodes = model.nodes ?? {};
  const normalized: Record<string, unknown> = {};
  for ( const id of Object.keys( nodes ).sort() )
    normalized[ id ] = normalizeNode( nodes[ id ] );

  const meta: Record<string, unknown> = {};
  if ( model.meta?.author ) meta.author = model.meta.author;
  if ( model.meta?.parentHash ) meta.parentHash = model.meta.parentHash;
  if ( model.meta?.datasetHash ) meta.datasetHash = model.meta.datasetHash;
  if ( model.meta?.license ) meta.license = model.meta.license;
  if ( model.meta?.tags && model.meta.tags.length > 0 ) meta.tags = [ ...model.meta.tags ].sort();
  // meta.created 明确剔除（不确定字段）。

  const projection = {
    schemaVersion: model.schemaVersion,
    graph: normalizeNode( model.graph ),
    nodes: normalized,
    meta,
  };
  return canonicalJson( projection );
}

// ---------------------------------------------------------------------------
// specHash 输入
// ---------------------------------------------------------------------------

/**
 * 收集模型里用到的全部 op 指纹（opRef → version + codeHash + depVersions）。
 * 03 §4.1：算子实现版本与依赖版本**必须进哈希**，否则"同一个 op"在不同部署里
 * 行为不同，specHash 的同一性就是假的。
 */
export function opFingerprints ( model: Model, registry?: OpRegistry ): string[]
{
  const refs = new Set<OpRef>();
  for ( const n of Object.values( model.nodes ?? {} ) ) refs.add( n.op );
  const out: string[] = [];
  for ( const ref of [ ...refs ].sort() )
  {
    const def = registry?.get( ref );
    out.push( def ? contractFingerprint( def.contract ) : `${ ref }#unregistered` );
  }
  return out;
}

/**
 * specHash 的输入字符串（对结果取 SHA-256 得到 specHash）。
 *
 *   specHash := SHA-256( canonical(Model \ 不确定字段)
 *                        + TrainerSpec.strategy + schemaVersion
 *                        + Σ(opRef → version, depVersions) )
 *
 * 实现提示（03 §4）：TrainerSpec.strategy 进哈希，而 strategy.params（如 pop 大小）不进，
 * 保证"同结构不同超参仍可对齐"。
 */
export function specHashInput ( model: Model, registry?: OpRegistry ): string
{
  return [
    canonicalModelProjection( model ),
    `strategy:${ model.trainer.strategy }`,
    `schema:${ model.schemaVersion }`,
    `ops:${ opFingerprints( model, registry ).join( '|' ) }`,
  ].join( '\n' );
}
