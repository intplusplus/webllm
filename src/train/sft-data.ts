/**
 * SFT / DPO 合成数据：字符级指令任务（不依赖预训练词表，自包含）。
 *
 * 两个任务（对 3~6 字符的小写串）：
 *   Reverse: abcd -> dcba
 *   Sort:    bda  -> abd
 *
 * 训练样本 = prompt + completion；SFT 的 loss 只计 completion（completion mask），
 * DPO 的 chosen = 正确 completion、rejected = 典型错误（未反转 / 未排序）。
 */

/** SFT/DPO 专用字符表：小写字母 + 任务模板所需符号。 */
export const SFT_VOCAB = [
  ...'abcdefghijklmnopqrstuvwxyz',
  ...'RS:-> \n',
] as string[];

export const SFT_STOI = new Map<string, number>( SFT_VOCAB.map( ( ch, i ) => [ ch, i ] ) );

export function encodeSft ( text: string ): Uint32Array
{
  const ids = new Uint32Array( text.length );
  for ( let i = 0; i < text.length; i++ )
  {
    const id = SFT_STOI.get( text[ i ] );
    if ( id === undefined ) throw new Error( `SFT 数据含词表外字符 ${ JSON.stringify( text[ i ] ) }` );
    ids[ i ] = id;
  }
  return ids;
}

export interface SftSample
{
  prompt: string;
  completion: string;
}

/** 可复现的伪随机串。 */
function randStr ( rng: () => number, len: number ): string
{
  let s = '';
  for ( let i = 0; i < len; i++ ) s += 'abcdefghijklmnopqrstuvwxyz'[ Math.floor( rng() * 26 ) ];
  return s;
}

function makePair ( rng: () => number, task: 'reverse' | 'sort' ): SftSample
{
  // 固定 3 字符输入：对 2 层 64d 的 tiny 模型，长度泛化比任务本身更难学；
  // 固定长度让自检聚焦验证 masked CE / DPO 管线的正确性
  const src = randStr( rng, 3 );
  const dst = task === 'reverse' ? [ ...src ].reverse().join( '' ) : [ ...src ].sort().join( '' );
  const tag = task === 'reverse' ? 'Reverse' : 'Sort';
  return { prompt: `${ tag }: ${ src } -> `, completion: `${ dst }\n` };
}

/** n 条训练样本（两类任务交替）。 */
export function makeSftSamples ( seed: number, n: number ): SftSample[]
{
  let s = seed >>> 0;
  const rng = () =>
  {
    s = ( s * 1664525 + 1013904223 ) >>> 0;
    return s / 4294967296;
  };
  const out: SftSample[] = [];
  for ( let i = 0; i < n; i++ ) out.push( makePair( rng, i % 2 === 0 ? 'reverse' : 'sort' ) );
  return out;
}

/** DPO 偏好对：chosen = 正确 completion，rejected = 典型错误（未做任务，直接复制输入）。 */
export function makeDpoPairs ( seed: number, n: number ): { sample: SftSample; rejected: string }[]
{
  let s = ( seed ^ 0x5f3759df ) >>> 0;
  const rng = () =>
  {
    s = ( s * 1664525 + 1013904223 ) >>> 0;
    return s / 4294967296;
  };
  const out: { sample: SftSample; rejected: string }[] = [];
  for ( let i = 0; i < n; i++ )
  {
    const task = i % 2 === 0 ? 'reverse' : 'sort';
    const sample = makePair( rng, task );
    // rejected：模型最典型的失败模式 —— 没做任务，把输入原样（或降序）抄下来
    const src = sample.prompt.split( ' -> ' )[ 0 ].split( ': ' )[ 1 ];
    const bad = task === 'reverse' ? src : [ ...src ].sort().reverse().join( '' );
    out.push( { sample, rejected: `${ bad }\n` } );
  }
  return out;
}
