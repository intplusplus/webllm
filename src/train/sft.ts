/**
 * SFT/DPO 训练辅助：批次构造与贪心生成。
 * 供 tests/posttrain.ts（管线验证）与 ui/app.ts（训练台）共用。
 */
import { encodeSft, type SftSample } from './sft-data';
import type { Trainer } from './trainer';

export interface Batch
{
  tokens: Uint32Array;
  targets: Uint32Array;
  weights: Float32Array;
  B: number;
  T: number;
  /** 每条序列的 completion 长度（logp 归一与 DPO 权重用） */
  completionLens: number[];
  prompts: string[];
}

/** 把一组样本拼成定长批次；prompt 位置 w=0（不计 loss），completion 位置 w=1。 */
export function buildBatch ( samples: SftSample[], padId: number ): Batch
{
  const fulls = samples.map( ( s ) => s.prompt + s.completion );
  const T = Math.max( ...fulls.map( ( s ) => s.length ) );
  const B = samples.length;
  const tokens = new Uint32Array( B * T );
  const targets = new Uint32Array( B * T );
  const weights = new Float32Array( B * T );
  const completionLens: number[] = [];
  const prompts: string[] = [];

  for ( let b = 0; b < B; b++ )
  {
    const s = samples[ b ];
    const ids = [ ...encodeSft( s.prompt + s.completion ) ];
    const promptLen = s.prompt.length;
    const len = ids.length;
    completionLens.push( len - promptLen );
    prompts.push( s.prompt );
    for ( let t = 0; t < T; t++ )
    {
      const base = b * T + t;
      if ( t < len )
      {
        tokens[ base ] = ids[ t ];
        // 位置 t 预测 full[t+1]：只有当目标是 completion 的一部分时才计 loss
        if ( t + 1 < len && t + 1 >= promptLen )
        {
          targets[ base ] = ids[ t + 1 ];
          weights[ base ] = 1;
        }
        else
        {
          targets[ base ] = 0;
          weights[ base ] = 0;
        }
      }
      else
      {
        tokens[ base ] = padId;
        targets[ base ] = 0;
        weights[ base ] = 0;
      }
    }
  }
  return { tokens, targets, weights, B, T, completionLens, prompts };
}

/** 贪心生成 completion（到换行为止），返回不含换行的文本。 */
export async function greedyCompletion (
  trainer: Trainer,
  prompt: string,
  maxNew: number,
  vocab: string[],
  blockSize: number,
): Promise<string>
{
  const V = vocab.length;
  const ids = [ ...encodeSft( prompt ) ];
  let out = '';
  for ( let i = 0; i < maxNew; i++ )
  {
    const T = Math.min( ids.length, blockSize );
    const window = Uint32Array.from( ids.slice( ids.length - T ) );
    trainer.forward( window, 1, T );
    const logits = await trainer.readLogits();
    const last = logits.slice( ( T - 1 ) * V, T * V );
    let best = 0;
    for ( let v = 1; v < V; v++ ) if ( last[ v ] > last[ best ] ) best = v;
    const ch = vocab[ best ];
    if ( ch === '\n' ) break;
    out += ch;
    ids.push( best );
  }
  return out;
}
