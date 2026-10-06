/**
 * 公共训练网络 · 线路协议与权重编解码。
 *
 * 设计取向：
 * - 控制消息走 JSON（可读、好调试），权重走二进制（体积小、无精度损失）。
 * - 一条 DataChannel 上：string = 控制帧，ArrayBuffer = 权重帧。RTCDataChannel
 *   本身保证有序可靠，所以「元信息随权重同帧下发」不会错配。
 * - 权重帧头部带 FNV-1a 摘要，用于校验传输完整性。
 *   注意：FNV-1a 只能防误码，不能防恶意篡改；生产环境应换成 SHA-256 + 签名，
 *   见 docs/公共AI网络-规划.md 的「信任层」一节。
 */

/** 训练引擎标识。同一房间内所有节点必须一致。 */
export type EngineId = 'mlp' | 'gpu-tinygpt';

/** 模型规格 —— 写进 manifest，所有节点据此构造完全相同的初始模型。 */
export interface ModelSpec
{
  engine: EngineId;
  /** 字符表大小 */
  vocabSize: number;
  /** 上下文窗口（字符数） */
  ctx: number;
  /** 字符嵌入维度 */
  embDim: number;
  /** 隐层宽度 */
  hidden: number;
  /** 初始权重种子 —— 保证所有节点从同一个模型出发 */
  seed: number;
}

/** 语料引用：名称 + 指纹 + 字符表。指纹决定「大家训的是不是同一份数据」。 */
export interface CorpusRef
{
  name: string;
  digest: string;
  chars: number;
  vocab: string[];
}

/** 共识探针：一段所有节点都能读到的固定窗口，用于交叉验证提交的权重。 */
export interface ProbeRef
{
  offset: number;
  size: number;
}

/** 房间清单 —— 房间的「宪法」，所有节点的训练内容由它唯一确定。 */
export interface RoomManifest
{
  roomId: string;
  taskName: string;
  taskBrief: string;
  model: ModelSpec;
  /** 总轮次 */
  rounds: number;
  /** 每轮每节点本地步数 */
  localSteps: number;
  /** 每步序列条数 */
  batchSize: number;
  lr: number;
  /** 数据分片数（= 预期节点数上限） */
  shards: number;
  corpus: CorpusRef;
  probe: ProbeRef;
  createdAt: number;
  /** 清单自身的指纹 */
  fingerprint: string;
}

/** 参数名 → f32 切片。用扁平命名（如 `L0.wq.w`）保证跨引擎可迁移。 */
export type NamedWeights = Record<string, Float32Array>;

/** 随权重一起上报的元信息。 */
export interface WeightMeta
{
  round: number;
  peerId: string;
  /** 本轮本地消耗的样本数（= 序列条数 × 步数） */
  samples: number;
  /** 本轮本地消耗的 token 数 */
  tokens: number;
  /** 本地训练集上的 loss */
  localLoss: number;
  /** 共识探针上的 loss —— 供聚合方交叉验证 */
  probeLoss: number;
  /** ||w_local - w_global_prev||₂，衡量本轮贡献强度 */
  deltaNorm: number;
  /** 权重帧摘要 */
  digest: string;
}

/** 设备能力快照（只用于展示与调度，不参与正确性）。 */
export interface DevCap
{
  kind: string;
  webgpu: boolean;
  cores: number;
  memoryGB: number;
  ua: string;
}

/** 单节点单轮在账本上的记录。 */
export interface LedgerEntry
{
  round: number;
  peerId: string;
  name: string;
  deviceKind: string;
  samples: number;
  tokens: number;
  probeLoss: number;
  localLoss: number;
  deltaNorm: number;
  /** 该节点在 FedAvg 中的权重份额 */
  share: number;
  verdict: 'ok' | 'suspect' | 'timeout';
  note: string;
}

/** 一轮结束后的汇总，广播给所有节点。 */
export interface RoundStats
{
  round: number;
  /** 聚合后全局模型在探针上的 loss */
  globalLoss: number;
  /** 上一轮全局 loss */
  prevGlobalLoss: number;
  /** 在线节点数（含主机） */
  online: number;
  /** 实际参与加权的节点数 */
  aggregated: number;
  /** 本轮上下行总字节 */
  bytes: number;
  elapsedMs: number;
  entries: LedgerEntry[];
}

/** 控制帧联合类型。 */
export type ControlMessage =
  | { t: 'hello'; peerId: string; name: string; device: DevCap }
  | { t: 'assign'; shardIndex: number; manifest: RoomManifest }
  | { t: 'round/open'; round: number }
  | { t: 'round/close'; stats: RoundStats }
  | { t: 'credit'; peerId: string; chars: number; digest: string }
  | { t: 'bye'; reason: string };

// ------------------------------------------------------------------ 哈希

/** FNV-1a 32 位 → 8 位十六进制。快、无依赖；非密码学安全。 */
export function fnv1aHex ( bytes: Uint8Array ): string
{
  let h = 0x811c9dc5;
  for ( let i = 0; i < bytes.length; i++ )
  {
    h ^= bytes[ i ];
    h = Math.imul( h, 0x01000193 );
  }
  return ( h >>> 0 ).toString( 16 ).padStart( 8, '0' );
}

export function hashString ( s: string ): string
{
  return fnv1aHex( new TextEncoder().encode( s ) );
}

/** 对象指纹：键排序后序列化再哈希，保证跨节点一致。 */
export function fingerprintOf ( value: unknown ): string
{
  const canon = ( v: unknown ): unknown =>
  {
    if ( Array.isArray( v ) ) return v.map( canon );
    if ( v && typeof v === 'object' )
    {
      const o = v as Record<string, unknown>;
      const out: Record<string, unknown> = {};
      for ( const k of Object.keys( o ).sort() ) out[ k ] = canon( o[ k ] );
      return out;
    }
    return v;
  };
  return hashString( JSON.stringify( canon( value ) ) );
}

// ------------------------------------------------------------------ 权重编解码

const MAGIC = 0x574c4644; // 'WLFD' —— 公共训练网络权重帧

function align4 ( n: number ): number
{
  return ( n + 3 ) & ~3;
}

/**
 * 序列化权重帧：
 *   [u32 magic][u32 headerLen][header JSON (补空格到 4 字节对齐)][f32 载荷]
 * header 形如 { v, names, lengths, digest, meta }。
 */
export function encodeWeights ( w: NamedWeights, meta?: WeightMeta ): ArrayBuffer
{
  const names = Object.keys( w ).sort();
  const lengths = names.map( ( n ) => w[ n ].length );
  const total = lengths.reduce( ( a, b ) => a + b, 0 );
  if ( total === 0 ) throw new Error( 'encodeWeights: 空权重' );

  const payload = new Float32Array( total );
  let off = 0;
  for ( let i = 0; i < names.length; i++ )
  {
    const arr = w[ names[ i ] ];
    payload.set( arr, off );
    off += lengths[ i ];
  }

  // 摘要先算，写入 header，接收端重算比对
  const payloadBytes = new Float32Array( payload ); // 独立副本，避免后续视图干扰
  const digest = fnv1aHex( new Uint8Array( payloadBytes.buffer ) );

  const headerRaw = new TextEncoder().encode(
    JSON.stringify( { v: 1, names, lengths, digest, meta: meta ?? null } ),
  );
  const headerLen = align4( headerRaw.byteLength );

  const buf = new ArrayBuffer( 8 + headerLen + total * 4 );
  const dv = new DataView( buf );
  dv.setUint32( 0, MAGIC, true );
  dv.setUint32( 4, headerLen, true );
  const headerView = new Uint8Array( buf, 8, headerLen );
  headerView.set( headerRaw );
  headerView.fill( 0x20, headerRaw.byteLength ); // 空格补齐，JSON 容忍尾部空白

  new Float32Array( buf, 8 + headerLen, total ).set( payloadBytes );
  return buf;
}

export interface DecodedWeights
{
  weights: NamedWeights;
  meta: WeightMeta | null;
  digest: string;
  bytes: number;
}

/** 反序列化并校验摘要；失败时抛出，调用方应把该节点标记为不可信。 */
export function decodeWeights ( buf: ArrayBuffer ): DecodedWeights
{
  if ( buf.byteLength < 8 ) throw new Error( '权重帧过短' );
  const dv = new DataView( buf );
  if ( dv.getUint32( 0, true ) !== MAGIC ) throw new Error( '权重帧魔数不匹配' );
  const headerLen = dv.getUint32( 4, true );
  if ( 8 + headerLen > buf.byteLength ) throw new Error( '权重帧头部越界' );
  const header = JSON.parse(
    new TextDecoder().decode( new Uint8Array( buf, 8, headerLen ) ),
  ) as { names: string[]; lengths: number[]; digest: string; meta: WeightMeta | null };

  const total = header.lengths.reduce( ( a, b ) => a + b, 0 );
  if ( 8 + headerLen + total * 4 !== buf.byteLength ) throw new Error( '权重帧长度不一致' );

  const payloadBytes = new Uint8Array( buf, 8 + headerLen, total * 4 );
  const digest = fnv1aHex( payloadBytes );
  if ( digest !== header.digest ) throw new Error( `权重帧摘要不匹配（期待 ${ header.digest }，实得 ${ digest }）` );

  const weights: NamedWeights = {};
  let off = 0;
  for ( let i = 0; i < header.names.length; i++ )
  {
    const len = header.lengths[ i ];
    // slice 出独立副本，避免长期持有整块 ArrayBuffer
    weights[ header.names[ i ] ] = new Float32Array( buf.slice( 8 + headerLen + off * 4, 8 + headerLen + ( off + len ) * 4 ) );
    off += len;
  }
  return { weights, meta: header.meta, digest, bytes: buf.byteLength };
}

/** 权重整体摘要（用于账本与模型卡的可复现性记录）。 */
export function weightsDigest ( w: NamedWeights ): string
{
  const dec = decodeWeights( encodeWeights( w ) );
  return dec.digest;
}

/** 逐参数 L2 距离 ||a - b||₂。 */
export function l2Distance ( a: NamedWeights, b: NamedWeights ): number
{
  let sq = 0;
  for ( const k of Object.keys( a ) )
  {
    const x = a[ k ];
    const y = b[ k ];
    if ( !y || y.length !== x.length ) throw new Error( `l2Distance: 参数 ${ k } 形状不一致` );
    for ( let i = 0; i < x.length; i++ )
    {
      const d = x[ i ] - y[ i ];
      sq += d * d;
    }
  }
  return Math.sqrt( sq );
}

/**
 * 加权联邦平均（FedAvg）。weights[i] 按 sizes[i] 加权。
 * 与顺序无关（浮点误差量级），便于各节点复核。
 */
export function fedAvg ( list: NamedWeights[], sizes: number[] ): NamedWeights
{
  if ( list.length === 0 ) throw new Error( 'fedAvg: 无可聚合节点' );
  if ( list.length !== sizes.length ) throw new Error( 'fedAvg: 权重与样本数长度不一致' );
  const totalSize = sizes.reduce( ( a, b ) => a + b, 0 );
  if ( totalSize <= 0 ) throw new Error( 'fedAvg: 总样本数为 0' );

  const out: NamedWeights = {};
  for ( const k of Object.keys( list[ 0 ] ) )
  {
    const acc = new Float32Array( list[ 0 ][ k ].length );
    for ( let m = 0; m < list.length; m++ )
    {
      const w = sizes[ m ] / totalSize;
      const src = list[ m ][ k ];
      if ( !src || src.length !== acc.length ) throw new Error( `fedAvg: 参数 ${ k } 形状不一致` );
      for ( let i = 0; i < acc.length; i++ ) acc[ i ] += w * src[ i ];
    }
    out[ k ] = acc;
  }
  return out;
}
