/**
 * 训练 checkpoint 的 IndexedDB 存取（浏览器端持久化，数据不出本机）。
 *
 * 保存内容：配置 + 全部权重 + AdamW m/v + 步数 —— 足以无损恢复训练与推理。
 * 往返正确性由 tests/posttrain.ts 的 checkpoint 测试保证（保存→加载→logits 完全一致）。
 */

const DB_NAME = 'webllm-checkpoints';
const STORE = 'checkpoints';

interface CheckpointRecord
{
  key: string;
  savedAt: number;
  config: unknown;
  step: number;
  /** 参数名 → f32 数组 */
  params: Record<string, Float32Array>;
  /** AdamW 状态：参数名 → {m, v} */
  optimizer: Record<string, { m: Float32Array; v: Float32Array }>;
}

function openDb (): Promise<IDBDatabase>
{
  return new Promise( ( resolve, reject ) =>
  {
    const req = indexedDB.open( DB_NAME, 1 );
    req.onupgradeneeded = () =>
    {
      if ( !req.result.objectStoreNames.contains( STORE ) ) req.result.createObjectStore( STORE, { keyPath: 'key' } );
    };
    req.onsuccess = () => resolve( req.result );
    req.onerror = () => reject( req.error ?? new Error( 'IndexedDB 打开失败' ) );
  } );
}

function txDone ( tx: IDBTransaction ): Promise<void>
{
  return new Promise( ( resolve, reject ) =>
  {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject( tx.error ?? new Error( 'IndexedDB 事务失败' ) );
    tx.onabort = () => reject( tx.error ?? new Error( 'IndexedDB 事务中止' ) );
  } );
}

export interface CheckpointPayload
{
  config: unknown;
  step: number;
  params: Record<string, Float32Array>;
  optimizer: Record<string, { m: Float32Array; v: Float32Array }>;
}

export async function saveCheckpoint ( key: string, payload: CheckpointPayload ): Promise<void>
{
  const db = await openDb();
  const record: CheckpointRecord = { key, savedAt: Date.now(), ...payload };
  const tx = db.transaction( STORE, 'readwrite' );
  tx.objectStore( STORE ).put( record );
  await txDone( tx );
  db.close();
}

export async function loadCheckpoint ( key: string ): Promise<CheckpointPayload & { savedAt: number }>
{
  const db = await openDb();
  const tx = db.transaction( STORE, 'readonly' );
  const req = tx.objectStore( STORE ).get( key );
  const record = await new Promise<CheckpointRecord | undefined>( ( resolve, reject ) =>
  {
    req.onsuccess = () => resolve( req.result );
    req.onerror = () => reject( req.error ?? new Error( 'IndexedDB 读取失败' ) );
  } );
  db.close();
  if ( !record ) throw new Error( `checkpoint 不存在: ${ key }` );
  return { savedAt: record.savedAt, config: record.config, step: record.step, params: record.params, optimizer: record.optimizer };
}

export async function deleteCheckpoint ( key: string ): Promise<void>
{
  const db = await openDb();
  const tx = db.transaction( STORE, 'readwrite' );
  tx.objectStore( STORE ).delete( key );
  await txDone( tx );
  db.close();
}
