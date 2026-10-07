/**
 * 用 Spec IR 描述一个 GPT-2 风格的小模型（nanoGPT 结构）。
 *
 * 目的：让 IR 的编译产物（infer → plan → emit → run）跑出来的 logits，
 * 能与 `src/reference/gpt-ref.ts` 的**手写实现逐位对拍**。所以本文件只做一件事：
 * 把 `GPTConfig + GPTWeights` 翻译成一份 Model + 一张 TensorTable，语义严格照抄
 * `gptForwardRef` 的算子顺序（LayerNorm → Q/K/V → RoPE → 因果注意力 → attnProj 残差
 * → LayerNorm → fc → GELU → mlpProj 残差 → … → lnF → lmHead）。
 *
 * 两处关键约定（对拍能否对上的前提）：
 *   1. 参数通过 `props.bind` 指向张量名，而非端口；于是 tie 只需让 `lmHead.w` 与 `wte`
 *      在表里**指向同一份 Float32Array**（binding.ts §约定：两个节点绑同一个名字 ⇒ 同一张量）。
 *   2. 权重按名字逐条 set；bias 关闭时既不进表也不写 bind，这样 run 期的 Matmul/LayerNorm
 *      会走"无 bias"分支，与参考实现 `L.wq.b ?? undefined` 一致。
 */

import type { Model, ModelNode, NodeId } from './types';
import { IR_SCHEMA_VERSION } from './types';
import { builtinRegistry } from './op';
import { createTensorTable, f32, u32, type TensorTable } from './binding';
import type { RunBinding } from './exec';
import type { GPTConfig } from '../model/config';
import { headDim } from '../model/config';
import type { GPTWeights } from '../model/init';

// ---------------------------------------------------------------------------
// 对外类型
// ---------------------------------------------------------------------------

export interface GptIr
{
  model: Model;
  /** 权重 + 输入张量表；张量命名见文件顶注 / 任务约定。 */
  tensors: TensorTable;
  config: GPTConfig;
}

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------

/**
 * 取注册表里该 op 的最新 opRef（`Name@version`）。
 * 注册表可能尚未灌入内置算子（本文件按约束不 import './ops'），故回退到 `Name@1.0`
 * ——与全部内置算子的实际版本一致，保证结果确定。
 */
function opRefOf ( name: string ): string
{
  const def = builtinRegistry.latest( name );
  return def ? `${ def.contract.op }@${ def.contract.version }` : `${ name }@1.0`;
}

/** Matmul 节点的 bind：`b` 必给；bias 仅在启用且给了 key 时写，避免 run 期误取到 bias。 */
function matmulBind ( weightKey: string, biasKey: string | undefined ): Record<string, string>
{
  const bind: Record<string, string> = { b: weightKey };
  if ( biasKey !== undefined ) bind[ 'bias' ] = biasKey;
  return bind;
}

/** 通用矩阵乘（NT：权重按 [out, in] 存，输出特征数由 props.n 决定）。 */
function matmul (
  id: NodeId,
  opts: { n: number; a: NodeId; weightKey: string; biasKey?: string; hasBias: boolean },
): ModelNode
{
  return {
    id,
    op: opRefOf( 'Matmul' ),
    props: {
      transposeB: true,
      hasBias: opts.hasBias,
      n: opts.n,
      bind: matmulBind( opts.weightKey, opts.hasBias ? opts.biasKey : undefined ),
    },
    slot: { a: opts.a },
  };
}

/** 层归一化（可选仿射 bias）。 */
function layernorm (
  id: NodeId,
  opts: { dim: number; eps: number; x: NodeId; weightKey: string; biasKey?: string },
): ModelNode
{
  const bind: Record<string, string> = { weight: opts.weightKey };
  if ( opts.biasKey !== undefined ) bind[ 'bias' ] = opts.biasKey;
  return {
    id,
    op: opRefOf( 'LayerNorm' ),
    props: { dim: opts.dim, eps: opts.eps, bind },
    slot: { x: opts.x },
  };
}

/** 词/位置嵌入查表（ids 与 weight 都走 bind）。 */
function embed (
  id: NodeId,
  opts: { vocab: number; dim: number; idsKey: string; weightKey: string },
): ModelNode
{
  return {
    id,
    op: opRefOf( 'Embed' ),
    props: {
      vocab: opts.vocab,
      dim: opts.dim,
      bind: { ids: opts.idsKey, weight: opts.weightKey },
    },
  };
}

/** 旋转位置编码（GPT-NeoX 相邻成对；T 由 run 期 symbols.T 提供，seqLen 兜底）。 */
function rope (
  id: NodeId,
  opts: { heads: number; headDim: number; base: number; seqLen: number; x: NodeId },
): ModelNode
{
  return {
    id,
    op: opRefOf( 'RoPE' ),
    props: {
      heads: opts.heads,
      headDim: opts.headDim,
      base: opts.base,
      seqLen: opts.seqLen,
    },
    slot: { x: opts.x },
  };
}

/** 因果自注意力（q/k/v 三插槽）。 */
function attention (
  id: NodeId,
  opts: { heads: number; headDim: number; q: NodeId; k: NodeId; v: NodeId },
): ModelNode
{
  return {
    id,
    op: opRefOf( 'Attention' ),
    props: { heads: opts.heads, headDim: opts.headDim, causal: true },
    slot: { q: opts.q, k: opts.k, v: opts.v },
  };
}

/** GELU（默认 tanh 近似，与参考实现一致）。 */
function gelu ( id: NodeId, x: NodeId ): ModelNode
{
  return { id, op: opRefOf( 'GELU' ), props: {}, slot: { x } };
}

/** 残差：out = x + child。 */
function residual ( id: NodeId, opts: { x: NodeId; child: NodeId } ): ModelNode
{
  return {
    id,
    op: opRefOf( 'Residual' ),
    props: {},
    slot: { x: opts.x },
    children: [ opts.child ],
  };
}

// ---------------------------------------------------------------------------
// 构建
// ---------------------------------------------------------------------------

/** 由 GPTConfig + GPTWeights 构建 IR（权重按名字放进张量表）。 */
export function buildGptIr (
  config: GPTConfig,
  weights: GPTWeights,
  opts?: { eps?: number; ropeBase?: number },
): GptIr
{
  // 默认值与 gptForwardRef 保持一致，否则数值对拍会在 LayerNorm / RoPE 上漂移。
  const eps = opts?.eps ?? 1e-5;
  const ropeBase = opts?.ropeBase ?? 10000;

  const C = config.nEmbd;
  const D = headDim( config );
  const bias = config.bias;

  const nodes: Record<NodeId, ModelNode> = {};
  const put = ( n: ModelNode ): void => { nodes[ n.id ] = n; };

  const table = createTensorTable();
  const setWeight = ( name: string, data: Float32Array, shape: number[] ): void =>
  {
    table.set( name, f32( data, shape ) );
  };

  // ---- 权重入表 -----------------------------------------------------------
  setWeight( 'wte', weights.wte, [ config.vocabSize, C ] );
  setWeight( 'wpe', weights.wpe, [ config.blockSize, C ] );
  setWeight( 'lnFW', weights.lnFW, [ C ] );
  if ( bias ) setWeight( 'lnFB', weights.lnFB, [ C ] );
  // tie_word_embeddings：lmHead.w 与 wte 放进**同一份 Float32Array 引用**，
  // 这样两个节点经 props.bind 指向 'lmHead.w' / 'wte' 时拿到的是同一张量（参数共享自然成立）。
  setWeight( 'lmHead.w', weights.wte, [ config.vocabSize, C ] );
  if ( bias && weights.lmHead.b ) setWeight( 'lmHead.b', weights.lmHead.b, [ config.vocabSize ] );

  // ---- 嵌入：tok + pos → Fan(sum) ----------------------------------------
  put( embed( 'tok', { vocab: config.vocabSize, dim: C, idsKey: 'input.tokens', weightKey: 'wte' } ) );
  put( embed( 'pos', { vocab: config.blockSize, dim: C, idsKey: 'input.pos', weightKey: 'wpe' } ) );
  put( {
    id: 'emb',
    op: opRefOf( 'Fan' ),
    props: { mode: 'sum' },
    children: [ 'tok', 'pos' ],
  } );

  // ---- 逐层堆叠 -----------------------------------------------------------
  let prev: NodeId = 'emb';
  for ( let i = 0; i < config.nLayer; i++ )
  {
    const L = weights.layers[ i ];
    const p = `layers.${ i }.`;
    const id = ( s: string ): NodeId => `L${ i }.${ s }`;

    const linear = (
      key: string,
      lin: { w: Float32Array; b: Float32Array | null },
      outD: number,
      inD: number,
    ): void =>
    {
      setWeight( `${ p }${ key }.w`, lin.w, [ outD, inD ] );
      if ( bias && lin.b ) setWeight( `${ p }${ key }.b`, lin.b, [ outD ] );
    };

    setWeight( `${ p }ln1W`, L.ln1W, [ C ] );
    setWeight( `${ p }ln2W`, L.ln2W, [ C ] );
    // 关闭 bias 时 LayerNorm 只做缩放（与参考实现 hasBias=false 一致）：bias 张量不入表。
    if ( bias )
    {
      setWeight( `${ p }ln1B`, L.ln1B, [ C ] );
      setWeight( `${ p }ln2B`, L.ln2B, [ C ] );
    }
    linear( 'wq', L.wq, C, C );
    linear( 'wk', L.wk, C, C );
    linear( 'wv', L.wv, C, C );
    linear( 'attnProj', L.attnProj, C, C );
    linear( 'fc', L.fc, 4 * C, C );
    linear( 'mlpProj', L.mlpProj, C, 4 * C );

    // --- attention 子块 ---
    put( layernorm( id( 'ln1' ), {
      dim: C, eps, x: prev,
      weightKey: `${ p }ln1W`,
      biasKey: bias ? `${ p }ln1B` : undefined,
    } ) );
    put( matmul( id( 'q' ), { n: C, a: id( 'ln1' ), weightKey: `${ p }wq.w`, biasKey: `${ p }wq.b`, hasBias: bias } ) );
    put( matmul( id( 'k' ), { n: C, a: id( 'ln1' ), weightKey: `${ p }wk.w`, biasKey: `${ p }wk.b`, hasBias: bias } ) );
    put( matmul( id( 'v' ), { n: C, a: id( 'ln1' ), weightKey: `${ p }wv.w`, biasKey: `${ p }wv.b`, hasBias: bias } ) );
    put( rope( id( 'ropeQ' ), { heads: config.nHead, headDim: D, base: ropeBase, seqLen: config.blockSize, x: id( 'q' ) } ) );
    put( rope( id( 'ropeK' ), { heads: config.nHead, headDim: D, base: ropeBase, seqLen: config.blockSize, x: id( 'k' ) } ) );
    put( attention( id( 'attn' ), { heads: config.nHead, headDim: D, q: id( 'ropeQ' ), k: id( 'ropeK' ), v: id( 'v' ) } ) );
    put( matmul( id( 'proj' ), { n: C, a: id( 'attn' ), weightKey: `${ p }attnProj.w`, biasKey: `${ p }attnProj.b`, hasBias: bias } ) );
    put( residual( id( 'res1' ), { x: prev, child: id( 'proj' ) } ) );

    // --- mlp 子块 ---
    put( layernorm( id( 'ln2' ), {
      dim: C, eps, x: id( 'res1' ),
      weightKey: `${ p }ln2W`,
      biasKey: bias ? `${ p }ln2B` : undefined,
    } ) );
    put( matmul( id( 'fc' ), { n: 4 * C, a: id( 'ln2' ), weightKey: `${ p }fc.w`, biasKey: `${ p }fc.b`, hasBias: bias } ) );
    put( gelu( id( 'act' ), id( 'fc' ) ) );
    put( matmul( id( 'mlp' ), { n: C, a: id( 'act' ), weightKey: `${ p }mlpProj.w`, biasKey: `${ p }mlpProj.b`, hasBias: bias } ) );
    put( residual( id( 'res2' ), { x: id( 'res1' ), child: id( 'mlp' ) } ) );

    prev = id( 'res2' );
  }

  // ---- 收尾：lnF → lmHead ------------------------------------------------
  put( layernorm( 'lnf', {
    dim: C, eps, x: prev,
    weightKey: 'lnFW',
    biasKey: bias ? 'lnFB' : undefined,
  } ) );
  put( matmul( 'head', { n: config.vocabSize, a: 'lnf', weightKey: 'lmHead.w', biasKey: 'lmHead.b', hasBias: bias } ) );

  const model: Model = {
    schemaVersion: IR_SCHEMA_VERSION,
    graph: nodes[ 'head' ],
    nodes,
    trainer: {
      strategy: 'backprop',
      params: {},
      schedule: { localSteps: 1, syncEvery: 1, precision: 'fp32' },
    },
    meta: { author: 'webllm', tags: [ 'tiny-gpt', 'ir' ] },
  };

  return { model, tensors: table, config };
}

// ---------------------------------------------------------------------------
// 批输入绑定
// ---------------------------------------------------------------------------

/**
 * 把一批 token 写进张量表（含位置索引），返回 run() 需要的绑定。
 * 通过 `set` 覆盖同名张量 ⇒ 支持"编译一次、多步复用"（树是静态的，输入每步换）。
 */
export function bindBatch ( gpt: GptIr, tokens: Uint32Array, B: number, T: number ): RunBinding
{
  const need = B * T;
  // pos 由 tokens 派生，其长度必然等于 B*T；真正要守的是 tokens 的长度契约。
  if ( tokens.length !== need )
    throw new Error(
      `bindBatch：tokens 长度 ${ tokens.length } 与 B*T=${ need } 不一致（pos 长度也必须等于 B*T）`,
    );

  const pos = new Uint32Array( need );
  for ( let i = 0; i < need; i++ ) pos[ i ] = i % T;

  gpt.tensors.set( 'input.tokens', u32( tokens, [ need ] ) );
  gpt.tensors.set( 'input.pos', u32( pos, [ need ] ) );

  return { tensors: gpt.tensors, symbols: { B, T } };
}
