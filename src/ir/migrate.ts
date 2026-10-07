/**
 * IR 版本迁移（design/03-IR与类型系统.md §6）。
 *
 * 规则：
 *   - 1.x 内只新增可选字段 ⇒ 迁移是"补默认值"；
 *   - 2.0 才允许不兼容变更，届时在此加分支；
 *   - 迁移器是**纯函数**，可离线重算哈希 ⇒ 公地里的历史模型永不失效（IR-V5）。
 */

import type { Model, ModelNode, Precision, StrategyId, TrainerSpec } from './types';
import { IR_SCHEMA_VERSION } from './types';
import { builtinRegistry } from './op';
import { ensureBuiltinOps } from './ops';

const DEFAULT_TRAINER: TrainerSpec = {
  strategy: 'backprop',
  params: {},
  schedule: { localSteps: 1, syncEvery: 1, precision: 'fp32' },
};

const VALID_PRECISION: ReadonlySet<string> = new Set( [ 'fp32', 'fp16', 'int4', 'bf16' ] );

function asRecord ( v: unknown ): Record<string, unknown>
{
  return v !== null && typeof v === 'object' ? ( v as Record<string, unknown> ) : {};
}

/** 迁移期的宽松转换：历史 props 可能是任意结构，交给后续 infer 校验。 */
function asProps ( v: unknown ): ModelNode['props']
{
  return asRecord( v ) as ModelNode['props'];
}

/** 把 op 名补成 opRef（`Name` → `Name@1.0`，按注册表最新版本）。 */
function normalizeOp ( raw: unknown ): string
{
  const s = typeof raw === 'string' ? raw : '';
  if ( s.includes( '@' ) ) return s;
  const latest = builtinRegistry.latest( s );
  return latest ? `${ latest.contract.op }@${ latest.contract.version }` : `${ s }@1.0`;
}

function normalizeNode ( raw: unknown, fallbackId: string ): ModelNode
{
  const r = asRecord( raw );
  const node: ModelNode = {
    id: typeof r.id === 'string' && r.id !== '' ? r.id : fallbackId,
    op: normalizeOp( r.op ),
    props: asProps( r.props ),
  };
  if ( Array.isArray( r.children ) && r.children.length > 0 )
    node.children = ( r.children as unknown[] ).filter( ( x ): x is string => typeof x === 'string' );
  if ( Array.isArray( r.wiring ) && r.wiring.length > 0 ) node.wiring = r.wiring as ModelNode['wiring'];
  if ( r.slot !== undefined ) node.slot = asRecord( r.slot ) as Record<string, string>;
  if ( r.annot !== undefined ) node.annot = r.annot as ModelNode['annot'];
  if ( r.meta !== undefined ) node.meta = r.meta as ModelNode['meta'];
  return node;
}

/**
 * 把任意历史形态的 IR 升级到当前 schema。
 * 幂等：`migrate(migrate(m))` 与 `migrate(m)` 的 specHash 相同。
 */
export function migrate ( input: unknown ): Model
{
  ensureBuiltinOps();
  const r = asRecord( input );

  const rawNodes = asRecord( r.nodes );
  const nodes: Record<string, ModelNode> = {};
  for ( const id of Object.keys( rawNodes ) ) nodes[ id ] = normalizeNode( rawNodes[ id ], id );

  // 根节点：优先 graph，其次找无人引用的节点。
  let graph = normalizeNode( r.graph, 'root' );
  if ( !nodes[ graph.id ] ) nodes[ graph.id ] = graph;
  else graph = nodes[ graph.id ];

  // trainer 补默认值。
  const tr = asRecord( r.trainer );
  const sched = asRecord( tr.schedule );
  const prec = typeof sched.precision === 'string' && VALID_PRECISION.has( sched.precision )
    ? ( sched.precision as Precision )
    : DEFAULT_TRAINER.schedule.precision;
  const trainer: TrainerSpec = {
    strategy: ( typeof tr.strategy === 'string' ? tr.strategy : DEFAULT_TRAINER.strategy ) as StrategyId,
    params: asProps( tr.params ),
    schedule: {
      localSteps: typeof sched.localSteps === 'number' ? sched.localSteps : DEFAULT_TRAINER.schedule.localSteps,
      syncEvery: typeof sched.syncEvery === 'number' ? sched.syncEvery : DEFAULT_TRAINER.schedule.syncEvery,
      precision: prec,
    },
  };
  if ( Array.isArray( tr.requires ) ) trainer.requires = tr.requires as TrainerSpec['requires'];

  const meta = asRecord( r.meta ) as Model['meta'];

  const version = typeof r.schemaVersion === 'string' ? r.schemaVersion : IR_SCHEMA_VERSION;
  const major = version.split( '.' )[ 0 ];
  // 1.x 与未知 major 一律归一到当前 schema（将来 2.0 在此加分支）。
  const schemaVersion = major === IR_SCHEMA_VERSION.split( '.' )[ 0 ] ? version : IR_SCHEMA_VERSION;

  return { schemaVersion, graph, nodes, trainer, meta };
}
