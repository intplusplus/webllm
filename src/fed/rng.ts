/** 可复现伪随机数。单独成文件，是为了让 engine.ts 与 gpu-engine.ts 之间不产生运行时循环依赖。 */

/** mulberry32：小而稳的可复现 PRNG。 */
export function mulberry32 ( seed: number ): () => number
{
  let a = seed >>> 0;
  return () =>
  {
    a = ( a + 0x6d2b79f5 ) >>> 0;
    let t = a;
    t = Math.imul( t ^ ( t >>> 15 ), t | 1 );
    t ^= t + Math.imul( t ^ ( t >>> 7 ), t | 61 );
    return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;
  };
}
