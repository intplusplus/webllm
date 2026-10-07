/**
 * JSX ↔ Spec IR 双向投影（design/03-IR与类型系统.md §7，验收线 IR-V3）。
 *
 * IR 是权威表示（JSON），JSX 只是**可 round-trip 的糖**。因此这里的每条路径都围绕
 * 「同一份语义，写法不同」设计：
 *
 *   1. buildModel（JSX → IR）：id 由**遍历路径确定性**生成（纯函数），保证同一棵树
 *      两次 build 得到相同 id 全集；op 落成注册表里的 `Name@version`（opRef）；
 *   2. toJsx（IR → JSX）：从 model.graph 递归，沿 children 走平铺表；
 *   3. renderJsx：把树渲染成**逐字节稳定**的受限 JSX 文本（props 用 canonicalJson，
 *      key 排序 + 紧凑，保证同输入同输出）；
 *   4. parseJsx：只认 renderJsx 产出的那套受限语法，读回 id/props/children。
 *
 * round-trip 保义的钥匙是 **id 原样保留**：renderJsx 一定写出 id，parseJsx 一定读回 id，
 * 于是 buildModel 走「显式 id」分支，不再重新推导 ⇒ 两次 build 的节点表/子表逐字相同
 * ⇒ 经 canonical 投影后 specHash 不变（trainer/meta 由调用方原样传回即可）。
 */

import {
  IR_SCHEMA_VERSION,
  type Model,
  type ModelMeta,
  type ModelNode,
  type NodeId,
  type OpRef,
  type PropValue,
  type Props,
  type TrainerSpec,
} from './types';
import { builtinRegistry, opRef } from './op';
import { ensureBuiltinOps } from './ops';
import { canonicalJson } from './canonical';

// ---------------------------------------------------------------------------
// 表达式树节点与 pragma
// ---------------------------------------------------------------------------

/** JSX 表达式树节点（`h` 的产物）。 */
export interface JsxNode
{
  op: string;                 // 不带版本，如 'RMSNorm'
  props: Props;
  children: JsxNode[];
  /** 显式 id；缺省由 buildModel 按遍历路径确定性地生成。 */
  id?: string;
}

/** JSX pragma：`h('RMSNorm', { dim: 128 }, child1, child2)`。 */
export function h (
  op: string,
  props?: Props | null,
  ...children: Array<JsxNode | JsxNode[] | null | undefined | false>
): JsxNode
{
  // 叶子/条件分支用 null / undefined / false 占位，这里统一拍平、剔除假值。
  const kids: JsxNode[] = [];
  for ( const c of children )
  {
    if ( c === null || c === undefined || c === false ) continue;
    if ( Array.isArray( c ) ) kids.push( ...c );
    else kids.push( c );
  }
  return { op, props: props ?? {}, children: kids };
}

// ---------------------------------------------------------------------------
// props 深拷贝（不进 IR 就该共享引用：一份 props 可能被写回、被多处引用）
// ---------------------------------------------------------------------------

function clonePropValue ( v: PropValue ): PropValue
{
  if ( Array.isArray( v ) ) return v.map( clonePropValue );
  if ( v !== null && typeof v === 'object' )
  {
    // Expr / 嵌套对象：逐 key 复制，保留 key 顺序（不做 canonical 排序，忠实入参）。
    const src = v as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for ( const k of Object.keys( src ) ) out[ k ] = clonePropValue( src[ k ] as PropValue );
    return out as unknown as PropValue;
  }
  return v;
}

function cloneProps ( p: Props ): Props
{
  const out: Props = {};
  for ( const k of Object.keys( p ) ) out[ k ] = clonePropValue( p[ k ] );
  return out;
}

// ---------------------------------------------------------------------------
// id 生成：纯路径函数（同一棵树 ⇒ 同一 id 全集）
// ---------------------------------------------------------------------------

/** 根 id 的固定种子。 */
const ROOT_ID = 'root';

/** 第 k 个子节点的确定性 id：`${父id}/${op}#${k}`。 */
function childId ( parentId: NodeId, op: string, k: number ): NodeId
{
  return `${ parentId }/${ op }#${ k }`;
}

// ---------------------------------------------------------------------------
// JSX → IR
// ---------------------------------------------------------------------------

/** 表达式树 → 权威 IR（Model）。id 必须**确定性地**按遍历路径生成。 */
export function buildModel ( root: JsxNode, trainer: TrainerSpec, meta?: ModelMeta ): Model
{
  // 先确保内置算子已灌入注册表，否则合法 op 会被误判为未注册。
  ensureBuiltinOps();

  const nodes: Record<NodeId, ModelNode> = {};

  const walk = ( node: JsxNode, fallbackId: NodeId ): ModelNode =>
  {
    const def = builtinRegistry.latest( node.op );
    if ( !def ) throw new Error( `未注册的 op: ${ node.op }` );

    const id = node.id ?? fallbackId;
    // 用 hasOwnProperty 而非真值判断：避免 `__proto__` 等原型键被误判为已占用。
    if ( Object.prototype.hasOwnProperty.call( nodes, id ) )
      throw new Error( `节点 id 冲突（重复或与生成 id 撞车）: ${ id }` );

    const mnode: ModelNode = {
      id,
      op: opRef( def.contract ),
      props: cloneProps( node.props ?? {} ),
    };
    // 先登记自身，令子节点在列举时能看到父节点已占用的 id。
    nodes[ id ] = mnode;

    const childIds: NodeId[] = [];
    node.children.forEach( ( child, k ) =>
    {
      const cid = child.id ?? childId( id, child.op, k );
      childIds.push( cid );
      walk( child, cid );
    } );
    // 无子节点则不写 children（与 canonical 归一化一致：空数组等价于缺省）。
    if ( childIds.length > 0 ) mnode.children = childIds;
    return mnode;
  };

  const graph = walk( root, root.id ?? ROOT_ID );

  return {
    schemaVersion: IR_SCHEMA_VERSION,
    graph,
    nodes,
    trainer,
    meta: meta ?? {},
  };
}

// ---------------------------------------------------------------------------
// IR → JSX
// ---------------------------------------------------------------------------

/** 从 `Name@version` 取回不带版本的 op 名。 */
function opName ( ref: OpRef ): string
{
  const at = ref.lastIndexOf( '@' );
  return at >= 0 ? ref.slice( 0, at ) : ref;
}

/** 权威 IR → 表达式树（model.graph 为根，按 children 递归）。 */
export function toJsx ( model: Model ): JsxNode
{
  const nodes = model.nodes ?? {};

  const build = ( n: ModelNode ): JsxNode =>
  {
    const children: JsxNode[] = [];
    for ( const cid of n.children ?? [] )
    {
      const child = nodes[ cid ];
      if ( !child ) throw new Error( `IR 节点表缺少子节点: ${ cid }` );
      children.push( build( child ) );
    }
    return { op: opName( n.op ), props: cloneProps( n.props ?? {} ), children, id: n.id };
  };

  return build( model.graph );
}

// ---------------------------------------------------------------------------
// 渲染（稳定的受限 JSX 文本）
// ---------------------------------------------------------------------------

const INDENT = '  '; // 2 空格

function isModel ( v: JsxNode | Model ): v is Model
{
  // Model 有 graph 而无 op；JsxNode 反之。避免把两者混为一谈。
  return ( v as Model ).graph !== undefined && ( v as JsxNode ).op === undefined;
}

function renderNode ( node: JsxNode, depth: number ): string
{
  const pad = INDENT.repeat( depth );
  let head = `<${ node.op }`;
  // id 必须写出：round-trip 靠它锚定身份，缺省时才省略。
  if ( node.id !== undefined ) head += ` id=${ JSON.stringify( node.id ) }`;
  // props 用 canonicalJson 序列化（key 排序 + 紧凑），保证逐字节稳定。
  head += ` props=${ canonicalJson( node.props ?? {} ) }`;

  if ( node.children.length === 0 ) return `${ pad }${ head }/>`;

  const lines: string[] = [ `${ pad }${ head }>` ];
  for ( const c of node.children ) lines.push( renderNode( c, depth + 1 ) );
  lines.push( `${ pad }</${ node.op }>` );
  return lines.join( '\n' );
}

/** 表达式树（或 Model）→ 稳定的 JSX 文本（同一输入必须逐字节相同）。 */
export function renderJsx ( node: JsxNode | Model ): string
{
  const jsx = isModel( node ) ? toJsx( node ) : node;
  return renderNode( jsx, 0 );
}

// ---------------------------------------------------------------------------
// 解析（受限语法的递归下降）
// ---------------------------------------------------------------------------

/**
 * 受限语法解析器：renderJsx 的输出必须能 parse 回等价树。
 *
 * 文法（自 renderJsx 反推，刻意做窄）：
 *   element := '<' name attr* ( '/>' | '>' element* '</' name '>' )
 *   attr    := 'id' '=' string | 'props' '=' '{' json '}'
 *   json    := JSON 值（对象/数组/字符串/数字/布尔/null）
 */
export function parseJsx ( src: string ): JsxNode
{
  let pos = 0;

  const lineCol = ( at: number ): { line: number; column: number } =>
  {
    let line = 1, column = 1;
    for ( let i = 0; i < at; i++ )
    {
      if ( src[ i ] === '\n' ) { line++; column = 1; }
      else column++;
    }
    return { line, column };
  };

  const fail = ( msg: string ): never =>
  {
    const { line, column } = lineCol( Math.min( pos, src.length ) );
    throw new Error( `parseJsx: ${ msg }（第 ${ line } 行第 ${ column } 列）` );
  };

  const skipWs = (): void =>
  {
    while ( pos < src.length && /\s/.test( src[ pos ] ) ) pos++;
  };

  const readName = (): string =>
  {
    skipWs();
    if ( !/[A-Za-z_]/.test( src[ pos ] ?? '' ) ) fail( '期望标识符' );
    const start = pos;
    while ( pos < src.length && /[A-Za-z0-9_]/.test( src[ pos ] ) ) pos++;
    return src.slice( start, pos );
  };

  /** 读取一段双引号字符串（含引号、含转义），交给 JSON.parse 反解。 */
  const readStringRaw = (): string =>
  {
    if ( src[ pos ] !== '"' ) fail( '期望字符串' );
    const start = pos;
    pos++;
    while ( pos < src.length )
    {
      const c = src[ pos ];
      if ( c === '\\' ) { pos += 2; continue; }
      if ( c === '"' ) { pos++; return src.slice( start, pos ); }
      pos++;
    }
    return fail( '字符串未闭合' );
  };

  const parseJson = ( token: string ): unknown =>
  {
    try { return JSON.parse( token ); }
    catch { return fail( `JSON 解析失败: ${ token }` ); }
  };

  /** 读取一个 JSON 值（从当前游标起，只吃属于它的那段）。 */
  const readJsonValue = (): unknown =>
  {
    skipWs();
    const ch = src[ pos ];
    if ( ch === '{' || ch === '[' )
    {
      const start = pos;
      let depth = 0;
      while ( pos < src.length )
      {
        const c = src[ pos ];
        if ( c === '"' ) { readStringRaw(); continue; }
        if ( c === '{' || c === '[' ) { depth++; pos++; continue; }
        if ( c === '}' || c === ']' ) { depth--; pos++; if ( depth === 0 ) break; continue; }
        pos++;
      }
      if ( depth !== 0 ) fail( 'JSON 值未闭合' );
      return parseJson( src.slice( start, pos ) );
    }
    if ( ch === '"' ) return parseJson( readStringRaw() );
    // 原语（数字/true/false/null）：读到下一个分隔符为止。
    const start = pos;
    while ( pos < src.length && !/[}\s,]/.test( src[ pos ] ) ) pos++;
    return parseJson( src.slice( start, pos ) );
  };

  const parseElement = (): JsxNode =>
  {
    if ( src[ pos ] !== '<' ) fail( '期望 <' );
    pos++;
    if ( src[ pos ] === '/' ) fail( '意外的闭合标签' );
    const name = readName();

    let id: string | undefined;
    let props: Props = {};

    // ---- 属性 ----
    for (;;)
    {
      skipWs();
      if ( pos >= src.length ) fail( `标签 <${ name }> 未闭合` );
      if ( src.startsWith( '/>', pos ) ) { pos += 2; return finishSelf(); }
      if ( src[ pos ] === '>' ) { pos++; break; }

      const attr = readName();
      skipWs();
      if ( src[ pos ] !== '=' ) fail( `属性 ${ attr } 缺少 =` );
      pos++;
      skipWs();

      if ( attr === 'id' )
      {
        id = parseJson( readStringRaw() ) as string;
      }
      else if ( attr === 'props' )
      {
        // 与 renderJsx 对齐：`props=` 后**直接**是一个 JSON 对象（不再多套一层 `{}`）。
        if ( src[ pos ] !== '{' ) fail( 'props 后应为 JSON 对象 {' );
        const v = readJsonValue();
        if ( v === null || typeof v !== 'object' || Array.isArray( v ) )
          fail( 'props 必须是 JSON 对象' );
        props = v as Props;
      }
      else
      {
        fail( `未知属性 ${ attr }` );
      }
    }

    // ---- 子节点 ----
    const children: JsxNode[] = [];
    for (;;)
    {
      skipWs();
      if ( pos >= src.length ) fail( `缺少闭合标签 </${ name }>` );
      if ( src.startsWith( '</', pos ) )
      {
        pos += 2;
        const close = readName();
        skipWs();
        if ( src[ pos ] !== '>' ) fail( `闭合标签 </${ close }> 缺少 >` );
        pos++;
        if ( close !== name ) fail( `闭合标签不匹配: <${ name }> vs </${ close }>` );
        break;
      }
      children.push( parseElement() );
    }
    return { op: name, props, children, id };

    function finishSelf (): JsxNode
    {
      return { op: name, props, children: [], id };
    }
  };

  // ---- 顶层 ----
  skipWs();
  if ( pos >= src.length ) fail( '空输入' );
  if ( src[ pos ] !== '<' ) fail( '期望以 < 开头的元素' );
  const root = parseElement();
  skipWs();
  if ( pos < src.length ) fail( '根元素后存在多余内容' );
  return root;
}
