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
import {
  encodeSft,
  makeSftSamples,
  makeDpoPairs,
  SFT_VOCAB,
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

interface CurvePoint { x: number; y: number; }

export function renderTrainApp ( gpu: GpuContext, root: HTMLElement ): void
{
  // ------------------------------------------------------------ 状态
  let trainer: Trainer | null = null;
  let stopRequested = false;
  let busy = false;

  const pretrainCurve: CurvePoint[] = [];
  const sftCurve: CurvePoint[] = [];
  const dpoCurve: CurvePoint[] = [];

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
  const status = el( 'div', 'tb-status', '空闲 —— 点击「跑完整管线」开始，或分步执行' );
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
    legendDot( '#d29922', 'DPO margin' ),
  );
  chartCard.append( legend );
  wrap.append( chartCard );

  // 指标卡
  const metrics = el( 'div', 'tb-metrics' );
  wrap.append( metrics );

  // 对话
  const chatCard = el( 'section', 'tb-card' );
  chatCard.append( el( 'h2', undefined, '指令对话（tiny-GPT · 贪心解码）' ) );
  const chatLog = el( 'div', 'tb-chatlog', '先完成 SFT（第 2 阶段），然后试试下面的指令。' );
  chatCard.append( chatLog );
  const chatRow = el( 'div', 'tb-chatrow' );
  const chatInput = el( 'input', 'tb-input' ) as HTMLInputElement;
  chatInput.placeholder = 'Reverse: abcd -> ';
  const btnSend = el( 'button', 'tb-btn', '生成' ) as HTMLButtonElement;
  btnSend.disabled = true;
  chatRow.append( chatInput, btnSend );
  chatCard.append( chatRow );
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
    for ( const b of [ btnAll, btnPre, btnSft, btnDpo, btnSend ] ) ( b as HTMLButtonElement ).disabled = value;
  }

  function refreshButtons (): void
  {
    btnPre.disabled = busy;
    btnSft.disabled = busy || trainer === null;
    btnDpo.disabled = busy || trainer === null;
    btnSend.disabled = busy || trainer === null;
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

  async function ensureTrainer (): Promise<Trainer>
  {
    if ( trainer ) return trainer;
    trainer = new Trainer( gpu, UI_CONFIG, initWeights( UI_CONFIG, 4242 ), 8 );
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
    const steps = 300;
    const opts = { lr: 3e-3, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
    const ids = corpus.ids;
    let firstLoss = 0;
    let last = 0;
    const t0 = performance.now();

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

  async function runSft ( steps = 1500 ): Promise<void>
  {
    const t = await ensureTrainer();
    const samples = sftSamples();
    const opts = { lr: 0.01, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
    let first = 0;
    let last = 0;
    const t0 = performance.now();

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
    const t = await ensureTrainer();
    const pairs = makeDpoPairs( 31337, 12 );
    const K = 4;
    const beta = 0.5;
    const opts = { lr: 1e-3, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.0 };
    const steps = 120;
    const V = UI_CONFIG.vocabSize;
    const T0 = performance.now();

    // reference：冻结当前权重的推理副本
    setStatus( 'DPO：冻结 reference 模型…' );
    const refWeights = await t.exportWeights();
    const { TinyGpt } = await import( '../model/tiny-gpt' );
    const ref = new TinyGpt( gpu, UI_CONFIG, refWeights, 8 );

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
      const refLogits = await ref.readTensor( ref.forward( batch.tokens, B, batch.T ) );

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
        setStatus( `DPO ${ s + 1 }/${ steps }，margin ${ last.toFixed(3) }` );
        drawChart();
      }
    }
    metricCard( 'DPO 对齐', [
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

  // ------------------------------------------------------------ 对话
  async function send (): Promise<void>
  {
    if ( !trainer || busy ) return;
    const prompt = chatInput.value.trim();
    if ( !prompt ) return;
    chatInput.value = '';
    const userLine = el( 'div', 'tb-chat-user', prompt );
    chatLog.append( userLine );
    const botLine = el( 'div', 'tb-chat-bot' );
    chatLog.append( botLine );
    chatLog.scrollTop = chatLog.scrollHeight;

    const started = performance.now();
    const ids = [ ...encodeSft( prompt ) ];
    let out = '';
    for ( let i = 0; i < 8; i++ )
    {
      const T = Math.min( ids.length, UI_CONFIG.blockSize );
      const window = Uint32Array.from( ids.slice( ids.length - T ) );
      trainer.forward( window, 1, T );
      const logits = await trainer.readLogits();
      const V = UI_CONFIG.vocabSize;
      const lastRow = logits.slice( ( T - 1 ) * V, T * V );
      let best = 0;
      for ( let v = 1; v < V; v++ ) if ( lastRow[ v ] > lastRow[ best ] ) best = v;
      const ch = SFT_VOCAB[ best ];
      if ( ch === '\n' ) break;
      out += ch;
      ids.push( best );
      botLine.textContent = out;
      chatLog.scrollTop = chatLog.scrollHeight;
    }
    const ms = performance.now() - started;
    const meta = el( 'div', 'tb-chat-meta', `${ out.length } chars · ${ ms.toFixed( 0 ) }ms` );
    chatLog.append( meta );
    chatLog.scrollTop = chatLog.scrollHeight;
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