/**
 * 运行时张量绑定（design/06 §1 的 `run()` 输入之一）。
 *
 * IR 是**数据**：它不持有任何运行时张量（INV-15：节点不得反向持有上下文）。
 * 于是"权重/输入从哪来"必须显式绑定——本文件定义这个绑定的最小契约。
 *
 * 约定（与 `src/ir/exec.ts` 的端口解析规则配套）：
 *   - 节点的**端口**（`io.in`）按 `slot > wiring > children 位置 > props.bind` 解析；
 *   - 权重一类**非数据流**资源不进端口，放 `props.bind`（形如 `{ w: 'layers.0.wq.w' }`）；
 *   - 两者最终都落在同一张 `TensorTable` 上，于是"参数共享（tie）"自然成立：
 *     两个节点绑同一个名字 ⇒ 同一份张量（`lmHead.w` 复用 `wte` 就是这个机制）。
 */

import type { DType } from './types';

export type TensorData = Float32Array | Uint32Array | Int32Array;

export interface TensorValue
{
  data: TensorData;
  /** 具体数字形状（run 期间一切形状都已求值，不再有符号维）。 */
  shape: number[];
  dtype: DType;
}

export interface TensorTable
{
  get ( name: string ): TensorValue | undefined;
  set ( name: string, t: TensorValue ): void;
  has ( name: string ): boolean;
  names (): string[];
}

export function tv ( data: TensorData, shape: number[], dtype: DType = 'f32' ): TensorValue
{
  return { data, shape, dtype };
}

export function f32 ( data: Float32Array, shape: number[] ): TensorValue
{
  return { data, shape, dtype: 'f32' };
}

export function u32 ( data: Uint32Array, shape: number[] ): TensorValue
{
  return { data, shape, dtype: 'u32' };
}

export function numel ( shape: number[] ): number
{
  return shape.reduce( ( a, b ) => a * b, 1 );
}

/** 建一张可写张量表；同名重复 set 会覆盖（后来的绑定赢）。 */
export function createTensorTable ( init?: Record<string, TensorValue> ): TensorTable
{
  const map = new Map<string, TensorValue>();
  if ( init ) for ( const k of Object.keys( init ) ) map.set( k, init[ k ] );
  return {
    get: ( name ) => map.get( name ),
    set: ( name, t ) => { map.set( name, t ); },
    has: ( name ) => map.has( name ),
    names: () => [ ...map.keys() ].sort(),
  };
}

/** 取张量；缺失即抛（run 期缺失绑定属于**硬错误**，不能静默变 0）。 */
export function requireTensor ( table: TensorTable, name: string, who: string ): TensorValue
{
  const t = table.get( name );
  if ( !t ) throw new Error( `${ who } 绑定了不存在的张量 "${ name }"` );
  return t;
}
