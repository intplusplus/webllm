/**
 * 受限表达式的静态求值（design/03-IR与类型系统.md §3.3）。
 *
 * props 允许受限表达式以支持循环/条件/参数绑定，但表达式**必须可静态求值**：
 * 禁止 lambda / 闭包 / 外部调用。这里提供确定性求值器，供 infer 的符号求解
 * 与各 op 的 shape 函数使用。
 */

import type { Dim, Expr, PropValue, Props } from './types';
import { isExpr } from './types';

export interface ExprEnv
{
  /** 符号维取值表（`B`→4、`T`→128）；未绑定为 undefined。 */
  symbols: Record<string, number | undefined>;
  /** 当前节点的 props（供 `props.xxx` 引用）。 */
  props?: Props;
  /** 模型级 meta（供 `meta.B` 之类引用）。 */
  meta?: Record<string, unknown>;
}

export class ExprError extends Error {}

function getPath ( env: ExprEnv, path: string ): unknown
{
  const parts = path.split( '.' );
  const head = parts[ 0 ];
  let cur: unknown;
  if ( head === 'meta' ) cur = env.meta;
  else if ( head === 'props' ) cur = env.props;
  else if ( head === 'symbols' || head === 'sym' ) cur = env.symbols;
  else
  {
    // 裸符号 `B` / `T` 视作符号维。
    const v = env.symbols[ path ];
    if ( v !== undefined ) return v;
    throw new ExprError( `无法解析引用：${ path }` );
  }
  for ( let i = 1; i < parts.length; i++ )
  {
    if ( cur === null || cur === undefined || typeof cur !== 'object' )
      throw new ExprError( `引用路径中断：${ path }` );
    cur = ( cur as Record<string, unknown> )[ parts[ i ] ];
  }
  return cur;
}

export function evalExpr ( e: Expr, env: ExprEnv ): number | string | boolean
{
  switch ( e.kind )
  {
    case 'lit':
      return e.value;
    case 'ref':
    {
      const v = getPath( env, e.path );
      if ( typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean' ) return v;
      // 引用到 Expr 时递归求值。
      if ( isExpr( v ) ) return evalExpr( v, env );
      throw new ExprError( `引用不是标量：${ e.path }` );
    }
    case 'bin':
    {
      const a = Number( evalExpr( e.a, env ) );
      const b = Number( evalExpr( e.b, env ) );
      switch ( e.op )
      {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return b === 0 ? NaN : a / b;
        case '%': return b === 0 ? NaN : a % b;
      }
      throw new ExprError( `未知运算符 ${ e.op }` );
    }
    case 'range':
    {
      const from = Number( evalExpr( e.from, env ) );
      const to = Number( evalExpr( e.to, env ) );
      const step = e.step ? Number( evalExpr( e.step, env ) ) : 1;
      const out: number[] = [];
      if ( step === 0 ) throw new ExprError( 'range step 不能为 0' );
      for ( let i = from; step > 0 ? i < to : i > to; i += step ) out.push( i );
      return out as unknown as number;
    }
    case 'cond':
      return evalExpr( e.test, env ) ? evalExpr( e.then, env ) : evalExpr( e.else, env );
  }
}

/** 求一个 prop 值（可能是字面量或表达式）。 */
export function evalProp ( v: PropValue, env: ExprEnv ): PropValue
{
  if ( isExpr( v ) ) return evalExpr( v, env );
  if ( Array.isArray( v ) ) return v.map( ( x ) => evalProp( x, env ) );
  return v;
}

export function propRaw ( props: Props, key: string ): PropValue | undefined
{
  return props[ key ];
}

export function propNumber ( props: Props, key: string, env: ExprEnv, fallback?: number ): number | undefined
{
  const raw = props[ key ];
  if ( raw === undefined ) return fallback;
  const v = evalProp( raw, env );
  if ( typeof v === 'number' ) return v;
  if ( typeof v === 'string' && v.trim() !== '' && !Number.isNaN( Number( v ) ) ) return Number( v );
  return fallback;
}

export function propBool ( props: Props, key: string, env: ExprEnv, fallback = false ): boolean
{
  const raw = props[ key ];
  if ( raw === undefined ) return fallback;
  const v = evalProp( raw, env );
  if ( typeof v === 'boolean' ) return v;
  if ( typeof v === 'number' ) return v !== 0;
  if ( typeof v === 'string' ) return v === 'true' || v === '1';
  return fallback;
}

export function propString ( props: Props, key: string, env: ExprEnv, fallback?: string ): string | undefined
{
  const raw = props[ key ];
  if ( raw === undefined ) return fallback;
  const v = evalProp( raw, env );
  return typeof v === 'string' ? v : fallback;
}

/**
 * 把 DimList 解析为具体数字；遇到无法绑定的符号维返回 undefined，
 * 并把该符号名记入 `unresolved`（供 `DYNAMIC_SHAPE_UNRESOLVED`）。
 */
export function resolveDims (
  dims: Dim[] | undefined,
  env: ExprEnv,
  unresolved: string[] = [],
): number[] | undefined
{
  if ( !dims ) return undefined;
  const out: number[] = [];
  for ( const d of dims )
  {
    if ( typeof d === 'number' )
    {
      out.push( d );
      continue;
    }
    const v = env.symbols[ d ];
    if ( v === undefined )
    {
      if ( !unresolved.includes( d ) ) unresolved.push( d );
      return undefined;
    }
    out.push( v );
  }
  return out;
}
