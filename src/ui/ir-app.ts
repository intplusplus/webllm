/**
 * 模型编写台视图：把 Spec IR 的 infer / plan / emit / run / backward 摊开给人看。
 *
 * 设计原则（为什么这么写）：
 *   1. 所有数字都来自 studio 门面层的真实调用，页面里没有一处演示常量；
 *   2. 指纹实验与「应用修复」把「预期 vs 实际」显式写出来 —— 让页面自带自检，
 *      一旦架构/权重/注解没有正确分离，页面会红字报警，而不是默默展示；
 *   3. 纯 DOM 构造，不引任何框架或图表库；训练折线用 SVG <polyline> 手搓。
 *
 * 本页计算走 CPU 后端（与手写 gpt-forward-ref 对拍的那条路径），不依赖 WebGPU。
 */
import {
  listTemplates,
  openTemplate,
  reweight,
  annotateEnv,
  analyze,
  probeBackend,
  applyFix,
  loadCorpus,
  makeBatch,
  forwardOnce,
  trainSession,
  type AnalyzeResult,
  type CorpusView,
  type FixOutcome,
  type StudioSession,
} from '../ir/studio';
import type { BackendCapability } from '../ir/plan';
import type { Diag, DiagLevel } from '../ir/types';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** 极简建元素工具：与训练台 app.ts 保持同一种写法。 */
function el<K extends keyof HTMLElementTagNameMap>( tag: K, cls?: string, text?: string ): HTMLElementTagNameMap[K]
{
  const node = document.createElement( tag );
  if ( cls ) node.className = cls;
  if ( text !== undefined ) node.textContent = text;
  return node;
}

function mb ( bytes: number ): string
{
  return `${ ( bytes / 1024 / 1024 ).toFixed( 3 ) } MB`;
}

export function renderIrApp ( root: HTMLElement ): void
{
  root.innerHTML = '';

  // ------------------------------------------------------------------ 标题
  const header = el( 'header' );
  header.append( el( 'h1', undefined, '模型编写台' ) );
  header.append( el( 'p', 'sub', 'Spec IR 的推断 / 规划 / 发射 / 执行 —— 页面上的每个数字都来自真实计算，没有写死的演示值。' ) );
  root.append( header );

  // ------------------------------------------------------------------ 卡片骨架
  // ① 环境
  const envCard = el( 'section', 'card' );
  envCard.append( el( 'h2', undefined, '① 环境（后端能力位探测）' ) );
  const envBody = el( 'div' );
  envCard.append( envBody );
  root.append( envCard );

  // ② 模型
  const modelCard = el( 'section', 'card' );
  modelCard.append( el( 'h2', undefined, '② 模型（架构）' ) );
  const modelBody = el( 'div' );
  modelCard.append( modelBody );
  root.append( modelCard );

  // ③ 指纹实验
  const fpCard = el( 'section', 'card' );
  fpCard.append( el( 'h2', undefined, '③ 指纹实验（架构 / 权重 / 注解分离）' ) );
  fpCard.append( el( 'div', 'muted', '预期：换一套权重、或给所有节点打上 device/shard/precision 注解后，specHash 都不应变化。下面每次点击都会把「预期 vs 实际」写出来，作为页面自带的自检。' ) );
  const fpControls = el( 'div', 'tb-controls' );
  fpControls.style.marginTop = '10px';
  const btnReweight = el( 'button', 'tb-btn', '换一套权重' );
  const btnAnnotate = el( 'button', 'tb-btn', '打上 device/shard/precision 注解' );
  fpControls.append( btnReweight, btnAnnotate );
  fpCard.append( fpControls );
  const fpOut = el( 'div' );
  fpCard.append( fpOut );
  root.append( fpCard );

  // ④ 诊断
  const diagCard = el( 'section', 'card' );
  diagCard.append( el( 'h2', undefined, '④ infer() 诊断' ) );
  const diagCounts = el( 'div', 'muted', '（未分析）' );
  const diagFixLog = el( 'div', 'muted' );
  diagFixLog.style.minHeight = '1.2em';
  const diagList = el( 'div' );
  diagCard.append( diagCounts, diagFixLog, diagList );
  root.append( diagCard );

  // ⑤a 计划
  const planCard = el( 'section', 'card' );
  planCard.append( el( 'h2', undefined, '⑤a 执行计划 plan()' ) );
  const planBody = el( 'div' );
  planCard.append( planBody );
  root.append( planCard );

  // ⑤b 产物
  const artCard = el( 'section', 'card' );
  artCard.append( el( 'h2', undefined, '⑤b 发射产物 emit()' ) );
  const artBody = el( 'div' );
  artCard.append( artBody );
  root.append( artCard );

  // ⑥ 真跑
  const runCard = el( 'section', 'card' );
  runCard.append( el( 'h2', undefined, '⑥ 真跑（仅 GPT 模板）' ) );
  const runBody = el( 'div' );
  runCard.append( runBody );
  root.append( runCard );

  // 底部：诚实的边界说明
  const footCard = el( 'section', 'card' );
  footCard.append( el( 'h2', undefined, '边界说明' ) );
  footCard.append( el( 'div', 'muted', '反向传播已落到 IR（本页「训练 50 步」用的就是它，走 cpu-grads 的真实 VJP）；但 WebGPU 后端尚未接通 —— run() 目前只有 CPU 实现表（builtinCpuImpls），所以本页的执行全部在 CPU 上完成。' ) );
  root.append( footCard );

  // ------------------------------------------------------------------ 状态
  let probe: { backend: BackendCapability; report: string[] } | null = null;
  let session: StudioSession | null = null;
  let result: AnalyzeResult | null = null;
  let corpus: CorpusView | null = null;
  let corpusPromise: Promise<CorpusView> | null = null;
  let training = false;
  let runUi: { fwd: HTMLButtonElement; train: HTMLButtonElement; tip: HTMLElement } | null = null;

  // ------------------------------------------------------------------ ① 环境
  function renderEnv ( p: { backend: BackendCapability; report: string[] } ): void
  {
    envBody.innerHTML = '';
    envBody.append( el( 'div', 'muted', '本页所有计算都走 CPU 后端（与手写 gpt-forward-ref 对拍的那条路径），不依赖 WebGPU —— 即使探测不到 GPU，页面也完全可用。' ) );
    const pre = el( 'pre', 'mono', p.report.join( '\n' ) );
    pre.style.marginTop = '10px';
    envBody.append( pre );
    envBody.append( el( 'div', 'muted', `analyze()/plan() 使用的后端能力位：backend=${ p.backend.backend }，maxBufferSize=${ ( p.backend.maxBufferSize / 1024 / 1024 ).toFixed( 0 ) } MB，supportsF16=${ String( p.backend.supportsF16 ) }` ) );
  }

  // ------------------------------------------------------------------ ② 模型
  function renderModel (): void
  {
    if ( !session || !result ) return;
    // 先把已收窄的值固化成 const：闭包里的 let 在任意函数调用后会被 TS 重新放宽为可空。
    const s = session;
    const r = result;
    modelBody.innerHTML = '';

    const select = el( 'select', 'tb-input' );
    for ( const t of listTemplates() )
    {
      const opt = el( 'option' );
      opt.value = t.id;
      opt.textContent = t.name;
      select.append( opt );
    }
    select.value = s.templateId;
    select.onchange = () => selectTemplate( select.value );
    modelBody.append( select );

    const info = listTemplates().find( ( t ) => t.id === s.templateId );
    modelBody.append( el( 'div', 'muted', info ? info.note : '' ) );

    const hashLabel = el( 'div', 'muted', 'specHash（64 位十六进制，整行显示）' );
    hashLabel.style.marginTop = '12px';
    modelBody.append( hashLabel );
    modelBody.append( el( 'pre', 'mono', r.specHash ) );

    const ops = r.opList.map( ( o ) => `${ o.name } ×${ o.count }` ).join( '   ' );
    const stats = el( 'pre', 'mono' );
    stats.style.marginTop = '12px';
    stats.textContent = [
      `节点数        ：${ r.nodeCount }`,
      `算子清单      ：${ ops || '（无）' }`,
      `参数量        ：${ r.estimate.params.toLocaleString( 'en-US' ) }`,
      `权重字节      ：${ r.estimate.weightBytes } B = ${ mb( r.estimate.weightBytes ) }`,
      `张量个数      ：${ r.estimate.tensors }`,
    ].join( '\n' );
    modelBody.append( stats );
  }

  // ------------------------------------------------------------------ ③ 指纹实验
  function fingerprintResult ( label: string, before: string, after: string, expect: string, extraNote: string ): void
  {
    fpOut.innerHTML = '';
    const box = el( 'div' );
    box.style.marginTop = '10px';
    box.append( el( 'div', 'muted', label ) );
    const pre = el( 'pre', 'mono' );
    pre.textContent =
      `预期 specHash（应保持不变）：\n  ${ before }\n` +
      `实际 specHash（操作后）    ：\n  ${ after }`;
    box.append( pre );
    if ( extraNote ) box.append( el( 'div', 'muted', extraNote ) );
    if ( before === after )
    {
      box.append( el( 'div', undefined, `结论：${ expect }` ) );
    }
    else
    {
      box.append( el( 'pre', 'bad mono', `异常：指纹不应变化（预期 ${ expect }）` ) );
    }
    fpOut.append( box );
  }

  btnReweight.onclick = () =>
  {
    if ( !session || !probe || !result ) return;
    const be = probe.backend;
    const cur = session;
    const before = result.specHash;
    const seed = Math.floor( Math.random() * 1e9 );
    try
    {
      const ns = reweight( cur, seed );
      session = ns;
      const r = analyze( ns, be );
      result = r;
      renderAll();
      const extra = ns.config ? '' : '（该模板没有权重，reweight 为空操作，hash 自然不变）';
      fingerprintResult( `换一套权重（新 seed=${ seed }）`, before, r.specHash, '结构没变 ⇒ 指纹不变（架构与权重分离）', extra );
    }
    catch ( err )
    {
      fpOut.innerHTML = '';
      fpOut.append( el( 'pre', 'bad mono', `换权重失败：${ ( err as Error ).message }` ) );
    }
  };

  btnAnnotate.onclick = () =>
  {
    if ( !session || !probe || !result ) return;
    const be = probe.backend;
    const s = session;
    const before = result.specHash;
    try
    {
      annotateEnv( s, 'cpu', 4, 'fp16' );
      const r = analyze( s, be );
      result = r;
      renderAll();
      fingerprintResult( '打上 device=cpu / shard=tensor(peers=4) / precision=fp16 注解', before, r.specHash, '环境注解不进指纹（INV-11）', '' );
    }
    catch ( err )
    {
      fpOut.innerHTML = '';
      fpOut.append( el( 'pre', 'bad mono', `打注解失败：${ ( err as Error ).message }` ) );
    }
  };

  // ------------------------------------------------------------------ ④ 诊断
  function renderDiags (): void
  {
    if ( !result ) return;
    const r = result;
    diagList.innerHTML = '';
    const order: DiagLevel[] = [ 'error', 'warn', 'info' ];
    const grouped: Record<DiagLevel, Diag[]> = { error: [], warn: [], info: [] };
    for ( const d of r.diags ) grouped[ d.level ].push( d );
    diagCounts.textContent = `error ${ grouped.error.length } · warn ${ grouped.warn.length } · info ${ grouped.info.length }（共 ${ r.diags.length } 条）`;

    const color: Record<DiagLevel, string> = { error: 'var(--bad)', warn: '#d29922', info: 'var(--muted)' };

    if ( r.diags.length === 0 )
    {
      diagList.append( el( 'div', 'muted', '（无诊断，这张图是干净的）' ) );
      return;
    }
    for ( const level of order )
    {
      for ( const d of grouped[ level ] ) diagList.append( diagBlock( d, color[ level ] ) );
    }
  }

  function diagBlock ( d: Diag, color: string ): HTMLElement
  {
    const box = el( 'div' );
    box.style.cssText = `padding:8px 10px;border:1px solid var(--border);border-left:3px solid ${ color };border-radius:6px;background:#0f141c;margin-bottom:8px`;
    const head = el( 'div', undefined, `${ d.level } · ${ d.code }` );
    head.style.color = color;
    box.append( head );
    box.append( el( 'div', undefined, d.message + ( d.node ? `（节点 ${ d.node }）` : '' ) ) );
    if ( d.fix )
    {
      const fixLine = el( 'div', 'muted', `fix.kind = ${ d.fix.kind }` + ( d.fix.to ? ` → ${ d.fix.to }` : '' ) );
      fixLine.style.marginTop = '6px';
      const btn = el( 'button', 'tb-btn', '应用修复' );
      btn.style.marginTop = '6px';
      box.append( fixLine, btn );
      btn.onclick = () => applyDiagFix( d, btn );
    }
    return box;
  }

  function applyDiagFix ( d: Diag, btn: HTMLButtonElement ): void
  {
    if ( !session || !probe ) return;
    const be = probe.backend;
    const s = session;
    let out: FixOutcome;
    try
    {
      out = applyFix( s, d );
    }
    catch ( err )
    {
      diagFixLog.className = 'bad mono';
      diagFixLog.textContent = `应用修复抛错：${ ( err as Error ).message }`;
      return;
    }
    // 先把真实结果落下来（成功与否都照实显示）
    diagFixLog.className = 'muted';
    diagFixLog.textContent = `应用修复（${ d.code }）：${ out.note }`;
    if ( !out.applied )
    {
      // applied=false：灰字说明，不假装成功
      return;
    }
    btn.disabled = true;
    try
    {
      const r = analyze( s, be );
      result = r;
      renderAll();
      diagFixLog.textContent = `应用修复（${ d.code }）：${ out.note }；已重新 analyze，该诊断应已消失`;
    }
    catch ( err )
    {
      diagFixLog.className = 'bad mono';
      diagFixLog.textContent = `重新 analyze 失败：${ ( err as Error ).message }`;
    }
  }

  // ------------------------------------------------------------------ ⑤a 计划
  function renderPlan (): void
  {
    if ( !result ) return;
    const p = result.plan;
    planBody.innerHTML = '';

    const pre = el( 'pre', 'mono' );
    pre.textContent = [
      `后端 backend            ：${ p.backend }`,
      `arena 总字节            ：${ p.arenaBytes } B = ${ mb( p.arenaBytes ) }`,
      `bind group 数           ：${ p.bindGroups }`,
      `内存预算 budgetMB       ：${ p.budgetMB } MB`,
      `通信 plan.comm          ：${ p.comm }`,
      `压缩 plan.compress      ：${ p.compress }`,
      `保留激活 retainActivations：${ p.retainActivations ? '是' : '否' }`,
      `RNG 算子树 rng.ops      ：${ p.rngOps }`,
    ].join( '\n' );
    planBody.append( pre );

    const segLabel = el( 'div', 'muted', '各段字节：' );
    segLabel.style.marginTop = '10px';
    planBody.append( segLabel );
    planBody.append( el( 'pre', 'mono', p.segments.length === 0
      ? '（无段）'
      : p.segments.map( ( s ) => `${ s.name }：${ s.bytes } B = ${ ( s.bytes / 1024 ).toFixed( 1 ) } KB` ).join( '\n' ) ) );

    const softLabel = el( 'div', 'muted', '软判据 plan.soft：' );
    softLabel.style.marginTop = '10px';
    planBody.append( softLabel );
    if ( p.soft.length === 0 )
    {
      planBody.append( el( 'div', undefined, '（无）' ) );
    }
    else
    {
      const ul = el( 'ul', 'tests' );
      for ( const d of p.soft )
      {
        const li = el( 'li' );
        li.append( el( 'span', 'tag', d.level ) );
        li.append( el( 'span', 'name', d.code ) );
        li.append( el( 'span', 'detail', d.message ) );
        ul.append( li );
      }
      planBody.append( ul );
    }
  }

  // ------------------------------------------------------------------ ⑤b 产物
  function renderArtifact (): void
  {
    if ( !result ) return;
    const a = result.artifact;
    artBody.innerHTML = '';
    artBody.append( el( 'div', 'muted', `pass 数：${ a.passes } · passBreak 次数：${ a.passBreaks } · dispatch 总数：${ a.dispatches.length }` ) );

    const wrap = el( 'div' );
    wrap.style.cssText = 'max-height:320px;overflow:auto;border:1px solid var(--border);border-radius:6px;margin-top:10px';
    const table = el( 'table' );
    table.style.cssText = 'border-collapse:collapse;width:100%;font-size:12px';

    const thead = el( 'thead' );
    const hr = el( 'tr' );
    for ( const col of [ 'nodeId', 'kernel', 'workgroups', 'pass' ] )
    {
      const th = el( 'th', undefined, col );
      th.style.cssText = 'text-align:left;padding:6px 8px;position:sticky;top:0;background:#0f141c;border-bottom:1px solid var(--border)';
      hr.append( th );
    }
    thead.append( hr );
    table.append( thead );

    const tbody = el( 'tbody' );
    for ( const d of a.dispatches.slice( 0, 40 ) )
    {
      const tr = el( 'tr' );
      for ( const cell of [ d.nodeId, d.kernel, d.workgroups.join( '×' ), String( d.pass ) ] )
      {
        const td = el( 'td', undefined, cell );
        td.style.cssText = 'padding:4px 8px;border-bottom:1px solid var(--border)';
        tr.append( td );
      }
      tbody.append( tr );
    }
    table.append( tbody );
    wrap.append( table );
    artBody.append( wrap );
    if ( a.dispatches.length > 40 ) artBody.append( el( 'div', 'muted', `（仅显示前 40 条，共 ${ a.dispatches.length } 条）` ) );
  }

  // ------------------------------------------------------------------ ⑥ 真跑
  function setupRun (): void
  {
    // 非 GPT 模板没有权重 / 张量表，真跑无意义 —— 整块隐藏。
    if ( !session || !session.config )
    {
      runCard.style.display = 'none';
      runUi = null;
      return;
    }
    runCard.style.display = 'block';
    runBody.innerHTML = '';

    const corpusLine = el( 'div', 'muted', '语料：加载中…' );
    runBody.append( corpusLine );

    const controls = el( 'div', 'tb-controls' );
    controls.style.marginTop = '10px';
    const btnFwd = el( 'button', 'tb-btn tb-btn-primary', '跑一步前向' );
    const btnTrain = el( 'button', 'tb-btn', '训练 50 步' );
    controls.append( btnFwd, btnTrain );
    runBody.append( controls );

    const tip = el( 'div', 'muted' );
    tip.style.minHeight = '1.2em';
    runBody.append( tip );

    const runStatus = el( 'div', 'tb-status', '就绪' );
    runBody.append( runStatus );
    const runOut = el( 'div' );
    runBody.append( runOut );

    runUi = { fwd: btnFwd, train: btnTrain, tip };
    updateRunAvailability();

    if ( !corpusPromise ) corpusPromise = loadCorpus();
    corpusPromise.then( ( c ) =>
    {
      corpus = c;
      corpusLine.textContent = `语料：Tiny Shakespeare 已加载，共 ${ c.chars.toLocaleString( 'en-US' ) } 个字符，词表 ${ c.vocab.length } 项`;
      updateRunAvailability();
    } ).catch( ( err: unknown ) =>
    {
      corpusLine.className = 'bad mono';
      corpusLine.textContent = `语料加载失败：${ ( err as Error ).message }`;
    } );

    btnFwd.onclick = () => void runForward( runStatus, runOut );
    btnTrain.onclick = () => void runTrain( runStatus, runOut );
  }

  function updateRunAvailability (): void
  {
    if ( !runUi ) return;
    const ui = runUi;
    const hasError = !!result && result.diags.some( ( d ) => d.level === 'error' );
    const disabled = training || hasError || corpus === null;
    ui.fwd.disabled = disabled;
    ui.train.disabled = disabled;
    ui.tip.className = 'muted';
    ui.tip.style.color = hasError ? 'var(--bad)' : '';
    ui.tip.textContent = hasError
      ? '先修掉 error 级诊断，才能跑前向 / 训练。'
      : disabled
        ? ( training ? '训练进行中…' : '语料加载中…' )
        : '';
  }

  function setTraining ( value: boolean ): void
  {
    training = value;
    updateRunAvailability();
  }

  async function runForward ( runStatus: HTMLElement, runOut: HTMLElement ): Promise<void>
  {
    if ( !session ) return;
    if ( !corpus )
    {
      runStatus.textContent = '语料尚未就绪';
      return;
    }
    const s = session;
    const c = corpus;
    setTraining( true );
    runStatus.textContent = '跑一步前向…';
    try
    {
      const B = 2;
      const T = Math.min( 16, s.config!.blockSize );
      const batch = makeBatch( c.ids, B, T, 12345 );
      const rep = forwardOnce( s, batch.tokens, batch.targets, B, T );
      const cmp = rep.compare;
      runOut.innerHTML = '';
      const pre = el( 'pre', 'mono' );
      pre.textContent = [
        `loss（真实计算）    ：${ rep.loss.toFixed( 6 ) }`,
        `耗时                ：${ rep.ms.toFixed( 2 ) } ms`,
        `logits 形状         ：[ ${ rep.logitsShape.join( ', ' ) } ]`,
        cmp
          ? `对拍 IR vs 手写 gpt-forward-ref：ok=${ cmp.ok } · maxAbs=${ cmp.maxAbs.toExponential( 3 ) } · maxRel=${ cmp.maxRel.toExponential( 3 ) } · 比对元素 ${ cmp.elements }`
          : '对拍：不可用（未返回 compare）',
      ].join( '\n' );
      runOut.append( pre );
      runOut.append( el( 'div', 'muted', '这里的 maxAbs 是 IR 执行路径与手写 gpt-forward-ref 逐元素（同一批权重、同一批 token）比出来的最大绝对误差。' ) );
      runStatus.textContent = '前向完成';
    }
    catch ( err )
    {
      runStatus.textContent = '前向失败';
      runOut.innerHTML = '';
      runOut.append( el( 'pre', 'bad mono', `前向失败：${ ( err as Error ).message }` ) );
    }
    finally
    {
      setTraining( false );
    }
  }

  async function runTrain ( runStatus: HTMLElement, runOut: HTMLElement ): Promise<void>
  {
    if ( !session ) return;
    if ( !corpus )
    {
      runStatus.textContent = '语料尚未就绪';
      return;
    }
    const s = session;
    const c = corpus;
    setTraining( true );
    runStatus.textContent = '训练 50 步…';
    runOut.innerHTML = '';

    const chartBox = el( 'div' );
    chartBox.style.marginTop = '10px';
    const chart = buildChart();
    chartBox.append( chart );
    const listBox = el( 'div' );
    listBox.style.cssText = 'max-height:150px;overflow:auto;border:1px solid var(--border);border-radius:6px;padding:8px;margin-top:10px';
    const summary = el( 'pre', 'mono' );
    summary.style.marginTop = '10px';
    runOut.append( chartBox, listBox, summary );

    const losses: number[] = [];
    try
    {
      const reports = await trainSession( s, c.ids, {
        steps: 50,
        lr: 0.01,
        B: 4,
        T: 16,
        onStep: ( st ) =>
        {
          losses.push( st.loss );
          listBox.append( el( 'div', undefined, `step ${ st.step }：loss ${ st.loss.toFixed( 5 ) } · |grad| ${ st.gradNorm.toFixed( 4 ) } · ${ st.ms.toFixed( 1 ) } ms` ) );
          listBox.scrollTop = listBox.scrollHeight;
          drawChart( chart, losses );
          runStatus.textContent = `训练中 ${ st.step }/50，loss ${ st.loss.toFixed( 4 ) }`;
        },
      } );
      const first = reports[ 0 ].loss;
      const last = reports[ reports.length - 1 ].loss;
      const totalMs = reports.reduce( ( acc, rp ) => acc + rp.ms, 0 );
      const dropPct = first !== 0 ? ( ( first - last ) / first * 100 ).toFixed( 2 ) : '0.00';
      summary.textContent = [
        `首 loss ：${ first.toFixed( 5 ) }`,
        `末 loss ：${ last.toFixed( 5 ) }`,
        `下降幅度：${ ( first - last ).toFixed( 5 ) }（${ dropPct } %）`,
        `平均每步：${ ( totalMs / reports.length ).toFixed( 2 ) } ms（共 ${ totalMs.toFixed( 1 ) } ms / ${ reports.length } 步）`,
      ].join( '\n' );
      runStatus.textContent = '训练完成';
    }
    catch ( err )
    {
      runStatus.textContent = '训练失败';
      runOut.append( el( 'pre', 'bad mono', `训练失败：${ ( err as Error ).message }` ) );
    }
    finally
    {
      setTraining( false );
    }
  }

  // ------------------------------------------------------------------ 折线图（手搓 SVG）
  function buildChart (): SVGSVGElement
  {
    const svg = document.createElementNS( SVG_NS, 'svg' );
    svg.setAttribute( 'viewBox', '0 0 640 200' );
    svg.setAttribute( 'class', 'tb-chart' );
    return svg;
  }

  function drawChart ( svg: SVGSVGElement, losses: number[] ): void
  {
    while ( svg.firstChild ) svg.removeChild( svg.firstChild );

    const W = 640;
    const H = 200;
    const padL = 52;
    const padR = 16;
    const padT = 16;
    const padB = 28;
    const innerW = W - padL - padR;
    const innerH = H - padT - padB;

    // 坐标轴
    const yAxis = document.createElementNS( SVG_NS, 'line' );
    yAxis.setAttribute( 'x1', String( padL ) );
    yAxis.setAttribute( 'y1', String( padT ) );
    yAxis.setAttribute( 'x2', String( padL ) );
    yAxis.setAttribute( 'y2', String( padT + innerH ) );
    yAxis.setAttribute( 'stroke', '#232b38' );
    svg.append( yAxis );
    const xAxis = document.createElementNS( SVG_NS, 'line' );
    xAxis.setAttribute( 'x1', String( padL ) );
    xAxis.setAttribute( 'y1', String( padT + innerH ) );
    xAxis.setAttribute( 'x2', String( padL + innerW ) );
    xAxis.setAttribute( 'y2', String( padT + innerH ) );
    xAxis.setAttribute( 'stroke', '#232b38' );
    svg.append( xAxis );

    if ( losses.length === 0 ) return;

    // 自动定标：按数据范围留 10% 余量；全相等时给个人工范围，避免除零。
    let lo = Math.min( ...losses );
    let hi = Math.max( ...losses );
    if ( !( hi > lo ) )
    {
      hi = lo + 0.5;
      lo = lo - 0.5;
    }
    const margin = ( hi - lo ) * 0.1 || 0.1;
    const yMin = lo - margin;
    const yMax = hi + margin;

    const n = losses.length;
    const xAt = ( i: number ): number => padL + ( n === 1 ? innerW / 2 : ( innerW * i ) / ( n - 1 ) );
    const yAt = ( v: number ): number => padT + innerH - ( innerH * ( v - yMin ) ) / ( yMax - yMin );

    const poly = document.createElementNS( SVG_NS, 'polyline' );
    poly.setAttribute( 'fill', 'none' );
    poly.setAttribute( 'stroke', '#58a6ff' );
    poly.setAttribute( 'stroke-width', '1.5' );
    poly.setAttribute( 'points', losses.map( ( v, i ) => `${ xAt( i ).toFixed( 1 ) },${ yAt( v ).toFixed( 1 ) }` ).join( ' ' ) );
    svg.append( poly );

    losses.forEach( ( v, i ) =>
    {
      const dot = document.createElementNS( SVG_NS, 'circle' );
      dot.setAttribute( 'cx', String( xAt( i ) ) );
      dot.setAttribute( 'cy', String( yAt( v ) ) );
      dot.setAttribute( 'r', '1.5' );
      dot.setAttribute( 'fill', '#58a6ff' );
      svg.append( dot );
    } );

    const label = ( x: number, y: number, text: string ): void =>
    {
      const t = document.createElementNS( SVG_NS, 'text' );
      t.setAttribute( 'x', String( x ) );
      t.setAttribute( 'y', String( y ) );
      t.setAttribute( 'fill', '#7d8796' );
      t.setAttribute( 'font-size', '10' );
      t.textContent = text;
      svg.append( t );
    };
    label( 4, yAt( hi ) + 4, hi.toFixed( 3 ) );
    label( 4, yAt( lo ) + 4, lo.toFixed( 3 ) );
    label( padL + innerW - 64, H - 8, `step 1 … ${ n }` );
  }

  // ------------------------------------------------------------------ 汇总重绘
  function renderAll (): void
  {
    renderModel();
    renderDiags();
    renderPlan();
    renderArtifact();
    updateRunAvailability();
  }

  function selectTemplate ( id: string ): void
  {
    if ( !probe ) return;
    const be = probe.backend;
    try
    {
      const s = openTemplate( id );
      session = s;
      result = analyze( s, be );
    }
    catch ( err )
    {
      modelBody.innerHTML = '';
      modelBody.append( el( 'pre', 'bad mono', `打开模板失败：${ ( err as Error ).message }` ) );
      diagList.innerHTML = '';
      diagCounts.textContent = '（分析失败）';
      return;
    }
    fpOut.innerHTML = '';
    diagFixLog.textContent = '';
    renderAll();
    setupRun();
  }

  // ------------------------------------------------------------------ 启动
  void ( async () =>
  {
    envBody.append( el( 'div', 'muted', '探测后端能力位…' ) );
    try
    {
      const p = await probeBackend();
      probe = p;
      renderEnv( p );
    }
    catch ( err )
    {
      // probeBackend 内部已兜底，正常不会抛；这里再兜一层保证页面永远可用。
      envBody.innerHTML = '';
      envBody.append( el( 'pre', 'bad mono', `后端探测失败：${ ( err as Error ).message }（仍以 CPU 后端继续）` ) );
      probe = {
        backend: {
          backend: 'cpu',
          maxBufferSize: 512 * 1024 * 1024,
          maxBindGroups: 4,
          maxWorkgroupsPerDimension: 65535,
          supportsF16: false,
          supportsSubgroups: false,
        },
        report: [ '后端探测异常，回退 CPU 后端。' ],
      };
    }
    const templates = listTemplates();
    if ( templates.length > 0 ) selectTemplate( templates[ 0 ].id );
  } )();
}
