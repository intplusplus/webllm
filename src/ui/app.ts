/**
 * 训练台：tiny-GPT 端到端训练与对话的可视化操作台。
 *
 * 三阶段管线（预训练 → SFT → DPO），每阶段：
 *   - 实时 loss / margin 曲线（canvas，逐点推进）
 *   - 阶段指标卡（起始/结束 loss、步数、耗时）
 *   - checkpoint 自动保存 + 手动加载
 * 最后一栏：指令对话（贪心逐字符流式生成）。
 *
 * 训练循环每步都有 GPU 回读（await），天然让出主线程，UI 不冻结；
 * 「停止」通过翻转 stopRequested 实现（合作式中断，下一步开始前检查）。
 */
import type { GpuContext } from '../gpu/device';
import { loadTinyShakespeare } from '../train/data';
import { initWeights } from '../model/init';
import { Trainer } from '../train/trainer';
import { CpuTrainer, type TrainCore } from '../train/cpu-trainer';
import {
  encodeSft,
  makeSftSamples,
  makeDpoPairs,
  SFT_VOCAB,
  SFT_STOI,
  type SftSample,
} from '../train/sft-data';
import { buildBatch } from '../train/sft';
import { saveCheckpoint, loadCheckpoint } from '../store/checkpoint';
import type { GPTConfig } from '../model/config';

const UI_CONFIG: GPTConfig = {
  vocabSize: SFT_VOCAB.length,
  blockSize: 32,
  nLayer: 2,
  nHead: 2,
  nEmbd: 64,
  bias: true,
};

const CKPT_KEY = 'trainbench-tinygpt';
const PAD_ID = SFT_VOCAB.indexOf( ' ' );

/**
 * CPU 模式的步数缩放：CpuTrainer 单步比 GPU 慢一个量级以上，
 * 步数等比缩减，让每个阶段仍在「几十秒~几分钟」的体感范围内跑完。
 * （GPU 侧保持原步数：pretrain 300 / sft 1500 / dpo 120）
 */
const CPU_STEPS = { pretrain: 120, sft: 250, dpo: 40 } as const;

interface CurvePoint { x: number; y: number; }

export function renderTrainApp ( gpu: GpuContext | null, root: HTMLElement ): void
{
  // ------------------------------------------------------------ 状态
  // gpu 为 null ⇒ CPU 模式（CpuTrainer 走 Spec IR 全链路，功能一致、速度较慢）
  const isCpu = gpu === null;
  let trainer: TrainCore | null = null;
  let stopRequested = false;
  let busy = false;

  // 「我教它」/ 偏好池：用户参与训练的数据（localStorage 持久化，刷新不丢）
  const MY_KEY = 'trainbench-my-samples';
  const PREF_KEY = 'trainbench-pref-pairs';
  const mySamples: SftSample[] = loadJson( MY_KEY, [] );
  const prefPairs: Array<{ prompt: string; chosen: string; rejected: string }> = loadJson( PREF_KEY, [] );

  function loadJson<T> ( key: string, fallback: T ): T
  {
    try
    {
      const raw = localStorage.getItem( key );
      return raw ? ( JSON.parse( raw ) as T ) : fallback;
    }
    catch { return fallback; }
  }
  function saveJson ( key: string, value: unknown ): void
  {
    try { localStorage.setItem( key, JSON.stringify( value ) ); } catch { /* 隐私模式等，忽略 */ }
  }

  // 进化时间线：探针 → 对应快照列容器
  const EVOL_PROBES = [ 'Reverse: abc -> ', 'Sort: bca -> ' ] as const;
  const evolCols = new Map<string, HTMLElement>();

  const pretrainCurve: CurvePoint[] = [];
  const sftCurve: CurvePoint[] = [];
  const dpoCurve: CurvePoint[] = [];
  const teachCurve: CurvePoint[] = [];

  // ------------------------------------------------------------ DOM 骨架
  const el = <K extends keyof HTMLElementTagNameMap>( tag: K, cls?: string, text?: string ): HTMLElementTagNameMap[ K ] =>
  {
    const n = document.createElement( tag );
    if ( cls ) n.className = cls;
    if ( text !== undefined ) n.textContent = text;
    return n;
  };

  const wrap = el( 'div', 'tb' );

  // 产品补课：新用户第一眼该知道「这是什么、三个按钮分别是什么」。
  {
    const help = el( 'details', 'tb-help' ) as HTMLDetailsElement;
    help.append( el( 'summary', undefined, '这是什么？三个阶段分别是什么意思？（点开展开）' ) );
    help.append( el( 'p', 'hint', '这是浏览器里自研 LLM 引擎的完整训练台：不联网、不上传，模型和训练全程跑在你这台设备的浏览器里。点「▶ 跑完整管线」从零训一个字符级小模型，也可以分步执行。' ) );
    const ol = el( 'ol' );
    for ( const s of [
      '预训练：海量文本上学会「下一个字是什么」——模型从此会写通顺的字句。',
      'SFT 微调：用问答示例教它「按指令回答」，从补全机器变成助手。',
      'DPO 对齐：用偏好对比教它「哪种回答更好」，收掉胡说八道的脾气。',
    ] ) ol.append( el( 'li', undefined, s ) );
    help.append( ol );
    wrap.append( help );
  }

  // CPU 模式说明：没有 WebGPU 也能完整跑，只是慢（诚实预期管理）
  if ( isCpu )
  {
    const cpuNote = el( 'div', 'tb-cpu-note' );
    cpuNote.append( el( 'p', 'hint',
      '当前浏览器没有可用的 WebGPU（GPUAdapter 被驱动或浏览器策略禁用）。' +
      '训练台已自动切换到 CPU 模式：功能完整（预训练 / SFT / DPO / 对话 / 存档全部可用），' +
      '但步数已缩减、单步更慢——适合体验流程；要跑全量训练请换支持 WebGPU 的浏览器（Chrome / Edge）或设备。' ) );
    wrap.append( cpuNote );
  }

  // 顶部：阶段按钮
  const controls = el( 'div', 'tb-controls' );
  const btnPre = el( 'button', 'tb-btn', '1 · 预训练' ) as HTMLButtonElement;
  const btnSft = el( 'button', 'tb-btn', '2 · SFT 微调' ) as HTMLButtonElement;
  const btnDpo = el( 'button', 'tb-btn', '3 · DPO 对齐' ) as HTMLButtonElement;
  const btnAll = el( 'button', 'tb-btn tb-btn-primary', '▶ 跑完整管线' ) as HTMLButtonElement;
  const btnStop = el( 'button', 'tb-btn tb-btn-danger', '■ 停止' ) as HTMLButtonElement;
  btnStop.disabled = true;
  for ( const b of [ btnAll, btnPre, btnSft, btnDpo, btnStop ] ) controls.append( b );
  wrap.append( controls );

  // 状态条
  const status = el( 'div', 'tb-status',
    isCpu
      ? '空闲（CPU 模式）—— 点击「跑完整管线」开始，或分步执行'
      : '空闲 —— 点击「跑完整管线」开始，或分步执行' );
  wrap.append( status );

  // 曲线
  const chartCard = el( 'section', 'tb-card' );
  chartCard.append( el( 'h2', undefined, '训练曲线（预训练 / SFT loss · DPO margin）' ) );
  const canvas = el( 'canvas', 'tb-chart' ) as HTMLCanvasElement;
  canvas.width = 840;
  canvas.height = 220;
  chartCard.append( canvas );
  const legend = el( 'div', 'tb-legend' );
  legend.append(
    legendDot( '#58a6ff', '预训练 loss' ),
    legendDot( '#3fb950', 'SFT loss' ),
    legendDot( '#a371f7', '教学 loss' ),
    legendDot( '#d29922', 'DPO margin' ),
  );
  chartCard.append( legend );
  wrap.append( chartCard );

  // 进化时间线：训练中定期用当前模型对固定探针生成，看见「它什么时候学会」
  const evolCard = el( 'section', 'tb-card' );
  evolCard.append( el( 'h2', undefined, '它的进化（训练中定期用当前模型做同一道题）' ) );
  const evolProbes = el( 'div', 'tb-evol' );
  evolProbes.append(
    evolColumn( 'Reverse: abc -> ' ),
    evolColumn( 'Sort: bca -> ' ),
  );
  evolCard.append( evolProbes );
  evolCard.append( el( 'div', 'hint', '左边一行 = 同一道题在不同训练步数下的回答。从乱码到正确答案，就是训练在发生。' ) );
  wrap.append( evolCard );

  // 指标卡
  const metrics = el( 'div', 'tb-metrics' );
  wrap.append( metrics );

  // 「我教它」：用户自定义样本 —— 训练台从「演示」变成「实验室」的核心
  const teachCard = el( 'section', 'tb-card' );
  teachCard.append( el( 'h2', undefined, '教它 · 你写的样本，它真的会学会' ) );
  teachCard.append( el( 'div', 'hint',
    '模型是字符级的：只会小写字母和 RS:-> 空格换行。教它任何「输入 -> 输出」的对应关系，' +
    '比如 abc -> cba，或你的暗号 qq -> zzz。添加几条后点「教它」，然后到下面对话框考它。' ) );
  const teachRow = el( 'div', 'tb-teachrow' );
  const teachIn = el( 'input', 'tb-input' ) as HTMLInputElement;
  teachIn.placeholder = '它该学会什么？例如：abc -> cba';
  const teachStepsIn = el( 'input', 'tb-input tb-steps' ) as HTMLInputElement;
  teachStepsIn.type = 'number';
  teachStepsIn.min = '20';
  teachStepsIn.max = '600';
  teachStepsIn.value = '60';
  teachStepsIn.title = '教学步数';
  const teachBtn = el( 'button', 'tb-btn tb-btn-primary', '▶ 教它' ) as HTMLButtonElement;
  teachRow.append( teachIn, teachStepsIn, teachBtn );
  teachCard.append( teachRow );
  const teachErr = el( 'div', 'tb-teach-err' );
  teachErr.style.display = 'none';
  teachCard.append( teachErr );
  const teachList = el( 'div', 'tb-teachlist' );
  teachCard.append( teachList );
  teachCard.append( el( 'div', 'hint', '点列表里的句子可删除。教学 = 在当前模型上继续训练，只学你这些句子。' ) );
  wrap.append( teachCard );

  // 对话（考它 / 对比偏好）
  const chatCard = el( 'section', 'tb-card' );
  chatCard.append( el( 'h2', undefined, '考它（每一步显示模型脑内的候选概率）' ) );
  const chatLog = el( 'div', 'tb-chatlog', '先教它或跑一次 SFT，然后在这里考它。' );
  chatCard.append( chatLog );
  const chatRow = el( 'div', 'tb-chatrow' );
  const chatInput = el( 'input', 'tb-input' ) as HTMLInputElement;
  chatInput.placeholder = 'Reverse: abcd -> ';
  const btnSend = el( 'button', 'tb-btn', '生成' ) as HTMLButtonElement;
  btnSend.disabled = true;
  const btnCompare = el( 'button', 'tb-btn', '⚖ 二选一（教它品味）' ) as HTMLButtonElement;
  btnCompare.disabled = true;
  btnCompare.title = '同一问题出两个回答，你选更好的 —— 攒成偏好对，按你的偏好对齐';
  chatRow.append( chatInput, btnSend, btnCompare );
  chatCard.append( chatRow );
  const compareBox = el( 'div', 'tb-compare' );
  compareBox.style.display = 'none';
  chatCard.append( compareBox );
  const btnAlign = el( 'button', 'tb-btn tb-btn-primary', '▶ 按我的偏好对齐' ) as HTMLButtonElement;
  btnAlign.disabled = true;
  btnAlign.style.display = 'none';
  chatCard.append( btnAlign );
  const chatExamples = el( 'div', 'tb-examples' );
  for ( const ex of [ 'Reverse: abcd -> ', 'Sort: dbca -> ', 'Reverse: xyz -> ' ] )
  {
    const chip = el( 'button', 'tb-chip', ex ) as HTMLButtonElement;
    chip.onclick = () => { chatInput.value = ex; void send(); };
    chatExamples.append( chip );
  }
  chatCard.append( chatExamples );
  wrap.append( chatCard );

  root.append( wrap );

  function legendDot ( color: string, label: string ): HTMLElement
  {
    const d = el( 'span', 'tb-legend-item' );
    const dot = el( 'span', 'tb-legend-dot' ) as HTMLElement;
    dot.style.background = color;
    d.append( dot, el( 'span', undefined, label ) );
    return d;
  }

  // ------------------------------------------------------------ UI 状态
  function setStatus ( text: string ): void { status.textContent = text; }

  function setBusy ( value: boolean ): void
  {
    busy = value;
    btnStop.disabled = !value;
    for ( const b of [ btnAll, btnPre, btnSft, btnDpo, btnSend, btnCompare, teachBtn, btnAlign ] ) ( b as HTMLButtonElement ).disabled = value;
  }

  /** 进化时间线的一列：探针标题 + 快照列表。 */
  function evolColumn ( probe: string ): HTMLElement
  {
    const col = el( 'div', 'tb-evol-col' );
    col.append( el( 'div', 'tb-evol-probe', probe ) );
    const items = el( 'div', 'tb-evol-items' );
    col.append( items );
    evolCols.set( probe, items );
    return col;
  }

  function refreshButtons (): void
  {
    btnPre.disabled = busy;
    btnSft.disabled = busy || trainer === null;
    btnDpo.disabled = busy || trainer === null;
    btnSend.disabled = busy || trainer === null;
    btnCompare.disabled = busy || trainer === null;
    teachBtn.disabled = busy;
    btnAlign.disabled = busy || trainer === null || prefPairs.length === 0;
  }

  function metricCard ( title: string, lines: string[] ): void
  {
    const card = el( 'div', 'tb-metric' );
    card.append( el( 'div', 'tb-metric-title', title ) );
    for ( const line of lines ) card.append( el( 'div', 'tb-metric-line', line ) );
    metrics.prepend( card );
  }

  // ------------------------------------------------------------ 曲线绘制
  function drawChart (): void
  {
    const ctx = canvas.getContext( '2d' );
    if ( !ctx ) return;
    const W = canvas.width;
    const H = canvas.height;
    const pad = 28;
    ctx.clearRect( 0, 0, W, H );
    ctx.fillStyle = '#0f141c';
    ctx.fillRect( 0, 0, W, H );

    const series = [
      { pts: pretrainCurve, color: '#58a6ff', label: 'pre' },
      { pts: sftCurve, color: '#3fb950', label: 'sft' },
      { pts: teachCurve, color: '#a371f7', label: 'teach' },
      { pts: dpoCurve, color: '#d29922', label: 'dpo' },
    ];
    const all = series.flatMap( ( s ) => s.pts );
    if ( all.length < 2 )
    {
      ctx.fillStyle = '#7d8796';
      ctx.font = '12px monospace';
      ctx.fillText( '等待训练数据…', pad, H / 2 );
      return;
    }
    const maxY = Math.max( 0.5, ...all.map( ( p ) => p.y ) );
    const maxX = Math.max( ...all.map( ( p ) => p.x ) );
    const px = ( x: number ) => pad + ( ( W - 2 * pad ) * x ) / maxX;
    const py = ( y: number ) => H - pad - ( ( H - 2 * pad ) * y ) / maxY;

    // 网格 + 轴
    ctx.strokeStyle = '#232b38';
    ctx.beginPath();
    for ( const gy of [ 0, 0.25, 0.5, 0.75, 1 ] )
    {
      ctx.moveTo( pad, py( gy * maxY ) );
      ctx.lineTo( W - pad, py( gy * maxY ) );
    }
    ctx.stroke();
    ctx.fillStyle = '#7d8796';
    ctx.font = '10px monospace';
    ctx.fillText( maxY.toFixed( 2 ), 2, py( maxY ) + 8 );
    ctx.fillText( '0', 2, py( 0 ) + 4 );
    ctx.fillText( `步数 ${ maxX }`, W - 70, H - 8 );

    for ( const s of series )
    {
      if ( s.pts.length < 2 ) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      s.pts.forEach( ( p, i ) =>
      {
        if ( i === 0 ) ctx.moveTo( px( p.x ), py( p.y ) );
        else ctx.lineTo( px( p.x ), py( p.y ) );
      } );
      ctx.stroke();
    }
  }

  // ------------------------------------------------------------ 管线各阶段

  async function ensureTrainer (): Promise<TrainCore>
  {
    if ( trainer ) return trainer;
    const weights = initWeights( UI_CONFIG, 4242 );
    trainer = gpu === null
      ? new CpuTrainer( UI_CONFIG, weights )
      : new Trainer( gpu, UI_CONFIG, weights, 8 );
    refreshButtons();
    return trainer;
  }

  async function runPretrain (): Promise<void>
  {
    setStatus( '预训练：加载 Tiny Shakespeare…' );
    const corpus = await loadTinyShakespeare();
    const t = await ensureTrainer();
    const V = UI_CONFIG.vocabSize;
    if ( corpus.vocab.length !== V )
    {
      setStatus( `语料字符集(${ corpus.vocab.length })与 SFT 词表(${ V })不一致，预训练改用合成序列` );
    }

    const B = 8;
    const T = UI_CONFIG.blockSize;
    const steps = isCpu ? CPU_STEPS.pretrain : 300;
    const opts = { lr: 3e-3, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
    const ids = corpus.ids;
    let firstLoss = 0;
    let last = 0;
    const t0 = performance.now();
    const probeEvery = Math.max( 10, Math.floor( steps / 8 ) );
    await probeSnapshot( 0, 'pre' );

    for ( let s = 0; s < steps; s++ )
    {
      if ( stopRequested ) return;
      const off = ( s * 37 ) % Math.max( 1, ids.length - B * T - 1 );
      const tokens = new Uint32Array( B * T );
      const targets = new Uint32Array( B * T );
      for ( let i = 0; i < B * T; i++ )
      {
        tokens[ i ] = ids[ off + i ] % V;
        targets[ i ] = ids[ off + i + 1 ] % V;
      }
      t.forward( tokens, B, T );
      last = await t.loss( targets );
      if ( s === 0 )
      {
        firstLoss = last;
        pretrainCurve.length = 0;
        pretrainCurve.push( { x: 1, y: last } );
      }
      t.backward( targets );
      t.step( opts );
      if ( s % 5 === 4 || s === steps - 1 )
      {
        pretrainCurve.push( { x: s + 1, y: last } );
        setStatus( `预训练 ${ s + 1 }/${ steps }，loss ${ last.toFixed(3) }` );
        drawChart();
      }
      if ( ( s + 1 ) % probeEvery === 0 || s === steps - 1 ) await probeSnapshot( s + 1, 'pre' );
    }
    metricCard( '预训练', [
      `${ steps } 步 · ${ ( ( performance.now() - t0 ) / 1000 ).toFixed( 1 ) }s`,
      `loss ${ firstLoss.toFixed( 2 ) } → ${ last.toFixed( 3 ) }（Tiny Shakespeare）`,
    ] );
    await autoSave( 'pretrain' );
  }

  function sftSamples (): SftSample[]
  {
    return makeSftSamples( 20260101, 512 );
  }

  async function runSft ( rawSteps = 1500 ): Promise<void>
  {
    const t = await ensureTrainer();
    const steps = isCpu ? Math.min( rawSteps, CPU_STEPS.sft ) : rawSteps;
    const samples = sftSamples();
    const opts = { lr: 0.01, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
    let first = 0;
    let last = 0;
    const t0 = performance.now();
    const probeEvery = Math.max( 10, Math.floor( steps / 8 ) );
    await probeSnapshot( 0, 'sft' );

    for ( let s = 0; s < steps; s++ )
    {
      if ( stopRequested ) return;
      const idx = Array.from( { length: 8 }, ( _, i ) => ( s * 8 + i ) % samples.length );
      const batch = buildBatch( idx.map( ( i ) => samples[ i ] ), PAD_ID );
      t.forward( batch.tokens, batch.B, batch.T );
      last = await t.loss( batch.targets, batch.weights );
      if ( s === 0 )
      {
        first = last;
        sftCurve.length = 0;
        sftCurve.push( { x: 1, y: last } );
      }
      t.backward( batch.targets, batch.weights );
      t.step( opts );
      if ( s % 10 === 9 || s === steps - 1 )
      {
        sftCurve.push( { x: s + 1, y: last } );
        setStatus( `SFT ${ s + 1 }/${ steps }，loss ${ last.toFixed(3) }` );
        drawChart();
      }
      if ( ( s + 1 ) % probeEvery === 0 || s === steps - 1 ) await probeSnapshot( s + 1, 'sft' );
    }
    metricCard( 'SFT 微调', [
      `${ steps } 步 · ${ ( ( performance.now() - t0 ) / 1000 ).toFixed( 1 ) }s`,
      `loss ${ first.toFixed( 3 ) } → ${ last.toFixed( 3 ) }（只计 completion）`,
    ] );
    btnSend.disabled = false;
    await autoSave( 'sft' );
  }

  async function runDpo (): Promise<void>
  {
    await runDpoOn( makeDpoPairs( 31337, 12 ), isCpu ? CPU_STEPS.dpo : 120, 'DPO 对齐' );
  }

  /**
   * DPO 通用驱动：pairs 是「prompt + chosen/rejected completion」的偏好对，
   * 内置任务与用户在对话里攒的偏好都走这里。
   */
  async function runDpoOn (
    pairs: Array<{ sample: SftSample; rejected: string }>,
    steps: number,
    label: string,
  ): Promise<void>
  {
    const t = await ensureTrainer();
    const K = 4;
    const beta = 0.5;
    const opts = { lr: 1e-3, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
    const V = UI_CONFIG.vocabSize;
    const T0 = performance.now();

    // reference：冻结当前权重的推理副本
    // GPU：TinyGpt（WGSL 推理）；CPU：再起一个 CpuTrainer 当冻结副本（只 forward）
    setStatus( `${ label }：冻结 reference 模型…` );
    const refWeights = await t.exportWeights();
    let refLogitsOf: ( tokens: Uint32Array, B: number, T: number ) => Promise<Float32Array>;
    if ( isCpu )
    {
      const { CpuTrainer: RefTrainer } = await import( '../train/cpu-trainer' );
      const ref = new RefTrainer( UI_CONFIG, refWeights );
      refLogitsOf = ( tokens, B, T ) => ref.readForward( tokens, B, T );
    }
    else
    {
      const { TinyGpt } = await import( '../model/tiny-gpt' );
      const ref = new TinyGpt( gpu!, UI_CONFIG, refWeights, 8 );
      refLogitsOf = async ( tokens, B, T ) =>
        await ref.readTensor( ref.forward( tokens, B, T ) );
    }

    const sigmoid = ( x: number ) => 1 / ( 1 + Math.exp( -x ) );
    let m0 = 0;
    let last = 0;

    for ( let s = 0; s < steps; s++ )
    {
      if ( stopRequested ) return;
      const batchPairs = Array.from( { length: K }, ( _, i ) => pairs[ ( s * K + i ) % pairs.length ] );
      const seqs: SftSample[] = [];
      for ( const p of batchPairs ) seqs.push( p.sample, { prompt: p.sample.prompt, completion: p.rejected } );
      const batch = buildBatch( seqs, PAD_ID );
      const B = batch.B;

      t.forward( batch.tokens, B, batch.T );
      const policyLogits = await t.readLogits();
      const refLogits = await refLogitsOf( batch.tokens, B, batch.T );

      const rowLogp = ( logits: Float32Array, row: number ): number =>
      {
        let sum = 0;
        for ( let tt = 0; tt < batch.T; tt++ )
        {
          if ( batch.weights[ row * batch.T + tt ] === 0 ) continue;
          const base = row * batch.T * V + tt * V;
          let mx = -Infinity;
          for ( let v = 0; v < V; v++ ) mx = Math.max( mx, logits[ base + v ] );
          let se = 0;
          for ( let v = 0; v < V; v++ ) se += Math.exp( logits[ base + v ] - mx );
          sum += ( logits[ base + batch.targets[ row * batch.T + tt ] ] - mx ) - Math.log( se );
        }
        return sum;
      };

      const weights = new Float32Array( batch.weights );
      let marginSum = 0;
      for ( let k = 0; k < K; k++ )
      {
        const cRow = 2 * k;
        const rRow = 2 * k + 1;
        const pc = rowLogp( policyLogits, cRow );
        const pcr = rowLogp( refLogits, cRow );
        const pr = rowLogp( policyLogits, rRow );
        const prr = rowLogp( refLogits, rRow );
        const z = beta * ( ( pc - pcr ) - ( pr - prr ) );
        const oneMinusSigma = 1 - sigmoid( z );
        marginSum += z / beta;
        for ( let tt = 0; tt < batch.T; tt++ )
        {
          const mask = batch.weights[ cRow * batch.T + tt ];
          weights[ cRow * batch.T + tt ] = mask === 0 ? 0 : ( beta * oneMinusSigma ) / ( K * batch.completionLens[ cRow ] );
          weights[ rRow * batch.T + tt ] = mask === 0 ? 0 : ( -beta * oneMinusSigma ) / ( K * batch.completionLens[ rRow ] );
        }
      }
      last = marginSum / K;
      if ( s === 0 )
      {
        m0 = last;
        dpoCurve.length = 0;
        dpoCurve.push( { x: 1, y: last } );
      }
      t.backward( batch.targets, weights );
      t.step( opts );
      if ( s % 2 === 1 || s === steps - 1 )
      {
        dpoCurve.push( { x: s + 1, y: last } );
        setStatus( `${ label } ${ s + 1 }/${ steps }，margin ${ last.toFixed(3) }` );
        drawChart();
      }
    }
    metricCard( label, [
      `${ steps } 步 · ${ ( ( performance.now() - T0 ) / 1000 ).toFixed( 1 ) }s`,
      `margin ${ m0.toFixed( 3 ) } → ${ last.toFixed( 3 ) }（相对 reference 的偏好差）`,
    ] );
    await autoSave( 'dpo' );
  }

  // ------------------------------------------------------------ checkpoint
  async function autoSave ( tag: string ): Promise<void>
  {
    if ( !trainer ) return;
    setStatus( `checkpoint 保存中（${ tag }）…` );
    await dumpToIndexedDb( tag );
    setStatus( `已保存 checkpoint（${ tag }）` );
  }

  async function dumpToIndexedDb ( tag: string ): Promise<void>
  {
    if ( !trainer ) throw new Error( '无模型可保存' );
    const params: Record<string, Float32Array> = {};
    const optimizer: Record<string, { m: Float32Array; v: Float32Array }> = {};
    for ( const name of trainer.paramNames() )
    {
      params[ name ] = await trainer.readParam( name );
      optimizer[ name ] = await trainer.readOptimizerState( name );
    }
    await saveCheckpoint( CKPT_KEY, { config: UI_CONFIG, step: trainer.stepCount, params, optimizer, tag } );
  }

  async function loadFromIndexedDb (): Promise<void>
  {
    const saved = await loadCheckpoint( CKPT_KEY );
    const t = await ensureTrainer();
    for ( const name of t.paramNames() )
    {
      if ( !saved.params[ name ] ) continue;
      t.writeParam( name, saved.params[ name ] );
      if ( saved.optimizer[ name ] ) t.writeOptimizerState( name, saved.optimizer[ name ].m, saved.optimizer[ name ].v );
    }
    t.setStepCount( saved.step );
    btnSend.disabled = false;
  }

  // ------------------------------------------------------------ 生成与对话

  /** 词表校验：返回第一条错误提示；合法返回 null。 */
  function vocabError ( text: string ): string | null
  {
    for ( const ch of text )
    {
      if ( !SFT_STOI.has( ch ) )
        return `字符 ${ JSON.stringify( ch ) } 不在词表里（只会小写字母和 RS:-> 空格换行）`;
    }
    return null;
  }

  /**
   * 从 prompt 生成最多 maxChars 个字符。
   * temperature=0 → 贪心；>0 → 按 softmax(v/T) 采样。
   * 每一步都带回 top5 候选与概率 —— 「看见模型的脑子」。
   */
  async function generateFrom (
    prompt: string,
    maxChars: number,
    temperature: number,
  ): Promise<{ text: string; steps: Array<{ ch: string; top: Array<{ ch: string; p: number }> }> }>
  {
    if ( !trainer ) throw new Error( '模型还没初始化' );
    const V = UI_CONFIG.vocabSize;
    const ids = [ ...encodeSft( prompt ) ];
    const out: string[] = [];
    const steps: Array<{ ch: string; top: Array<{ ch: string; p: number }> }> = [];
    for ( let i = 0; i < maxChars; i++ )
    {
      const T = Math.min( ids.length, UI_CONFIG.blockSize );
      const window = Uint32Array.from( ids.slice( ids.length - T ) );
      trainer.forward( window, 1, T );
      const logits = await trainer.readLogits();
      const lastRow = logits.slice( ( T - 1 ) * V, T * V );
      const temp = Math.max( 1e-3, temperature );
      let mx = -Infinity;
      for ( let v = 0; v < V; v++ ) mx = Math.max( mx, lastRow[ v ] );
      const scaled = new Float32Array( V );
      let sum = 0;
      for ( let v = 0; v < V; v++ )
      {
        scaled[ v ] = Math.exp( ( lastRow[ v ] - mx ) / temp );
        sum += scaled[ v ];
      }
      const order = [ ...scaled.keys() ].sort( ( a, b ) => scaled[ b ] - scaled[ a ] );
      const top = order.slice( 0, 5 ).map( ( v ) => ( { ch: SFT_VOCAB[ v ], p: scaled[ v ] / sum } ) );
      let pick: number;
      if ( temperature <= 0 )
      {
        pick = order[ 0 ];
      }
      else
      {
        let r = Math.random() * sum;
        pick = order[ 0 ];
        for ( let v = 0; v < V; v++ ) { r -= scaled[ v ]; if ( r <= 0 ) { pick = v; break; } }
      }
      const ch = SFT_VOCAB[ pick ];
      steps.push( { ch, top } );
      if ( ch === '\n' ) break;
      out.push( ch );
      ids.push( pick );
    }
    return { text: out.join( '' ), steps };
  }

  const dispCh = ( ch: string ): string => ( ch === ' ' ? '␣' : ch === '\n' ? '↵' : ch );

  /** 每一步的候选概率可视化（默认折叠，展开看「模型的脑子」）。 */
  function probStepsDom ( steps: Array<{ ch: string; top: Array<{ ch: string; p: number }> }> ): HTMLElement
  {
    const det = el( 'details', 'tb-probs' ) as HTMLDetailsElement;
    det.append( el( 'summary', undefined, `模型脑内（${ steps.length } 步的候选概率）` ) );
    for ( let i = 0; i < steps.length; i++ )
    {
      const { ch, top } = steps[ i ];
      const row = el( 'div', 'tb-prob-row' );
      row.append( el( 'span', 'tb-prob-pick', `${ i + 1 }→${ dispCh( ch ) }` ) );
      for ( const t of top )
      {
        const item = el( 'span', 'tb-prob' );
        const bar = el( 'span', 'tb-prob-bar' ) as HTMLElement;
        bar.style.width = `${ Math.max( 2, Math.round( t.p * 60 ) ) }px`;
        item.append(
          el( 'span', 'tb-prob-ch', dispCh( t.ch ) ),
          bar,
          el( 'span', 'tb-prob-p', `${ Math.round( t.p * 100 ) }%` ),
        );
        row.append( item );
      }
      det.append( row );
    }
    return det;
  }

  async function send (): Promise<void>
  {
    if ( !trainer || busy ) return;
    const prompt = chatInput.value.trim();
    if ( !prompt ) return;
    const bad = vocabError( prompt );
    if ( bad ) { setStatus( `考题有问题：${ bad }` ); return; }
    chatInput.value = '';
    chatLog.append( el( 'div', 'tb-chat-user', prompt ) );
    const botLine = el( 'div', 'tb-chat-bot' );
    chatLog.append( botLine );
    chatLog.scrollTop = chatLog.scrollHeight;

    const started = performance.now();
    const r = await generateFrom( prompt, 8, 0 );
    botLine.textContent = r.text === '' ? '（生成了空回答）' : r.text;
    const ms = performance.now() - started;
    const wrap0 = el( 'div', 'tb-chat-meta' );
    wrap0.append( el( 'span', undefined, `${ r.text.length } chars · ${ ms.toFixed( 0 ) }ms ` ) );
    wrap0.append( probStepsDom( r.steps ) );
    chatLog.append( wrap0 );
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  // ------------------------------------------------------------ 我教它

  function showTeachErr ( msg: string ): void
  {
    teachErr.textContent = msg;
    teachErr.style.display = 'block';
  }
  function hideTeachErr (): void
  {
    teachErr.style.display = 'none';
  }

  function renderTeachList (): void
  {
    teachList.innerHTML = '';
    if ( mySamples.length === 0 )
    {
      teachList.append( el( 'span', 'hint', '还没有你的样本。' ) );
      return;
    }
    mySamples.forEach( ( s, i ) =>
    {
      const chip = el( 'button', 'tb-chip', `${ s.prompt }${ s.completion.replace( '\n', '' ) }` ) as HTMLButtonElement;
      chip.title = '点击删除这条样本';
      chip.onclick = () =>
      {
        mySamples.splice( i, 1 );
        saveJson( MY_KEY, mySamples );
        renderTeachList();
      };
      teachList.append( chip );
    } );
  }

  function addSample (): void
  {
    const raw = teachIn.value;
    const bad = vocabError( raw );
    if ( bad ) { showTeachErr( bad ); return; }
    if ( !raw.includes( '->' ) ) { showTeachErr( '格式要包含 ->：左边是考题，右边是期望回答，例如 abc -> cba' ); return; }
    const cut = raw.indexOf( '->' );
    const prompt = raw.slice( 0, cut );
    const completion = raw.slice( cut + 2 ) + '\n';
    if ( prompt.trim().length === 0 || completion.trim().length === 0 )
    {
      showTeachErr( '两侧都不能为空' );
      return;
    }
    if ( prompt.length + completion.length > UI_CONFIG.blockSize )
    {
      showTeachErr( `太长：合计 ${ prompt.length + completion.length } 字符，上限 ${ UI_CONFIG.blockSize }（模型窗口就这么大）` );
      return;
    }
    mySamples.push( { prompt, completion } );
    saveJson( MY_KEY, mySamples );
    renderTeachList();
    teachIn.value = '';
    hideTeachErr();
  }

  async function runTeach ( steps: number ): Promise<void>
  {
    const t = await ensureTrainer();
    const opts = { lr: 0.01, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
    let first = 0;
    let last = 0;
    teachCurve.length = 0;
    const t0 = performance.now();
    await probeSnapshot( 0, 'teach' );

    for ( let s = 0; s < steps; s++ )
    {
      if ( stopRequested ) return;
      const idx = Array.from( { length: 8 }, ( _, i ) => ( s * 8 + i ) % mySamples.length );
      const batch = buildBatch( idx.map( ( i ) => mySamples[ i ] ), PAD_ID );
      t.forward( batch.tokens, batch.B, batch.T );
      last = await t.loss( batch.targets, batch.weights );
      if ( s === 0 ) first = last;
      t.backward( batch.targets, batch.weights );
      t.step( opts );
      teachCurve.push( { x: s + 1, y: last } );
      if ( s % 5 === 4 || s === steps - 1 )
      {
        setStatus( `教学 ${ s + 1 }/${ steps }，loss ${ last.toFixed( 3 ) }` );
        drawChart();
      }
    }
    metricCard( '教学', [
      `${ mySamples.length } 条样本 · ${ steps } 步 · ${ ( ( performance.now() - t0 ) / 1000 ).toFixed( 1 ) }s`,
      `loss ${ first.toFixed( 3 ) } → ${ last.toFixed( 3 ) } —— 去下面考它`,
    ] );
    await probeSnapshot( steps, 'teach' );
    await autoSave( 'teach' );
  }

  // ------------------------------------------------------------ 偏好（二选一 → DPO）

  async function runCompare (): Promise<void>
  {
    if ( !trainer || busy ) return;
    const prompt = chatInput.value.trim();
    if ( !prompt ) { setStatus( '先在左边输入考题，再点二选一' ); return; }
    const bad = vocabError( prompt );
    if ( bad ) { setStatus( `考题有问题：${ bad }` ); return; }
    chatInput.value = '';
    chatLog.append( el( 'div', 'tb-chat-user', `${ prompt }（对比模式）` ) );
    compareBox.innerHTML = '';
    compareBox.style.display = 'block';

    // A = 贪心（模型最有把握的回答）；B = 温度 0.8 采样（模型的另一种想法）
    const a = await generateFrom( prompt, 8, 0 );
    const b = await generateFrom( prompt, 8, 0.8 );
    if ( a.text === b.text )
    {
      compareBox.append( el( 'div', 'hint',
        `两个候选一样（"${ a.text || '（空）' }"）—— 模型对这道题非常确定。换个没教过的问题再试。` ) );
      chatLog.scrollTop = chatLog.scrollHeight;
      return;
    }

    const mk = ( label: string, text: string, steps: Array<{ ch: string; top: Array<{ ch: string; p: number }> }> ): HTMLElement =>
    {
      const card = el( 'div', 'tb-compare-card' );
      card.append( el( 'div', 'tb-compare-label', label ) );
      card.append( el( 'div', 'tb-compare-text', text === '' ? '（空回答）' : text ) );
      card.append( probStepsDom( steps ) );
      const btn = el( 'button', 'tb-btn', '✓ 这个更好' ) as HTMLButtonElement;
      btn.onclick = () =>
      {
        prefPairs.push( { prompt, chosen: text + '\n', rejected: ( text === a.text ? b : a ).text + '\n' } );
        saveJson( PREF_KEY, prefPairs );
        compareBox.innerHTML = '';
        compareBox.style.display = 'none';
        chatLog.append( el( 'div', 'tb-chat-meta', `已记录：在 "${ prompt }" 上你偏好 "${ text }"。攒到 ${ prefPairs.length } 组后可对齐。` ) );
        chatLog.scrollTop = chatLog.scrollHeight;
        btnAlign.style.display = 'inline-block';
        btnAlign.disabled = busy || trainer === null;
      };
      card.append( btn );
      return card;
    };
    compareBox.append( mk( '候选 A（贪心）', a.text, a.steps ), mk( '候选 B（采样）', b.text, b.steps ) );
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  async function runAlign (): Promise<void>
  {
    if ( prefPairs.length === 0 ) return;
    const pairs = prefPairs.map( ( p ) => ( {
      sample: { prompt: p.prompt, completion: p.chosen } as SftSample,
      rejected: p.rejected,
    } ) );
    const steps = isCpu ? 30 : 60;
    await runDpoOn( pairs, steps, `按你的偏好对齐（${ pairs.length } 组）` );
  }

  // ------------------------------------------------------------ 进化时间线

  async function probeSnapshot ( step: number, tag: string ): Promise<void>
  {
    if ( !trainer ) return;
    for ( const probe of EVOL_PROBES )
    {
      const col = evolCols.get( probe );
      if ( !col ) continue;
      let text = '';
      try
      {
        const r = await generateFrom( probe, 8, 0 );
        text = r.text;
      }
      catch { text = '（生成失败）'; }
      const item = el( 'div', 'tb-evol-item' );
      item.append(
        el( 'span', 'tb-evol-step', `${ tag }#${ step }` ),
        el( 'span', 'tb-evol-text', text === '' ? '（空）' : text ),
      );
      col.append( item );
    }
  }

  // ------------------------------------------------------------ 事件
  async function withBusy ( label: string, jobs: Array<() => Promise<void>> ): Promise<void>
  {
    if ( busy ) return;
    stopRequested = false;
    setBusy( true );
    try
    {
      for ( const job of jobs )
      {
        await job();
        if ( stopRequested ) { setStatus( '已停止' ); break; }
      }
      if ( !stopRequested ) setStatus( `${ label } 完成` );
    }
    catch ( err )
    {
      setStatus( `出错：${ ( err as Error ).message }` );
    }
    finally
    {
      setBusy( false );
      refreshButtons();
    }
  }

  btnPre.onclick = () => void withBusy( '预训练', [ runPretrain ] );
  btnSft.onclick = () => void withBusy( 'SFT', [ runSft ] );
  btnDpo.onclick = () => void withBusy( 'DPO', [ runDpo ] );
  btnAll.onclick = () => void withBusy( '完整管线（预训练 → SFT → DPO）', [ runPretrain, runSft, runDpo ] );
  btnStop.onclick = () => { stopRequested = true; setStatus( '停止中（当前步结束后退出）…' ); };
  btnSend.onclick = () => void send();
  chatInput.addEventListener( 'keydown', ( e ) => { if ( e.key === 'Enter' ) void send(); } );

  teachBtn.onclick = () =>
  {
    if ( mySamples.length === 0 ) { showTeachErr( '先在输入框写一条样本并回车添加，例如 abc -> cba' ); return; }
    const n = Math.max( 20, Math.min( 600, Number( teachStepsIn.value ) || 60 ) );
    teachStepsIn.value = String( n );
    void withBusy( `教学（${ mySamples.length } 条样本 × ${ n } 步）`, [ () => runTeach( n ) ] );
  };
  teachIn.addEventListener( 'keydown', ( e ) =>
  {
    if ( e.key === 'Enter' ) addSample();
  } );
  btnCompare.onclick = () => void runCompare().catch( ( err ) => setStatus( `出错：${ ( err as Error ).message }` ) );
  btnAlign.onclick = () => void withBusy( '按你的偏好对齐', [ runAlign ] );

  renderTeachList();
  if ( prefPairs.length > 0 ) btnAlign.style.display = 'inline-block';

  // 恢复已有 checkpoint 的提示
  void ( async () =>
  {
    try
    {
      await loadCheckpoint( CKPT_KEY );
      const resume = el( 'div', 'tb-resume' );
      const btnResume = el( 'button', 'tb-btn', '检测到已保存的模型 —— 点击加载' ) as HTMLButtonElement;
      btnResume.onclick = () => void ( async () =>
      {
        try
        {
          await loadFromIndexedDb();
          setStatus( '已从 checkpoint 恢复，可直接对话' );
          btnResume.remove();
        }
        catch ( err ) { setStatus( `恢复失败：${ ( err as Error ).message }` ); }
      } )();
      resume.append( btnResume );
      wrap.prepend( resume );
    }
    catch { /* 无 checkpoint，正常 */ }
  } )();

  drawChart();
}