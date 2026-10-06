import type { GpuContext } from './gpu/device';
import { runSelfTest } from './selftest';
import { renderTrainApp } from './ui/app';
import './ui/styles.css';

function requireElement<T extends Element>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) throw new Error(`页面元素不存在: ${selector}`);
  return node;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatLimits(gpu: GpuContext): string {
  const info = gpu.adapterInfo;
  const gib = (bytes: number) => (bytes / 1024 / 1024 / 1024).toFixed(2) + ' GiB';
  return [
    `vendor         : ${info.vendor ?? '(未知)'}`,
    `architecture   : ${info.architecture ?? '(未知)'}`,
    `device         : ${info.device ?? '(未知)'}`,
    `shader-f16     : ${gpu.hasF16 ? '支持（可走 fp16 推理路径）' : '不支持（只能用 fp32）'}`,
    `maxBufferSize  : ${gib(gpu.limits.maxBufferSize)}`,
    `maxStorageBufferBindingSize : ${gib(gpu.limits.maxStorageBufferBindingSize)}`,
    `maxBindGroups  : ${gpu.limits.maxBindGroups}`,
    `maxComputeWorkgroupStorageSize : ${gpu.limits.maxComputeWorkgroupStorageSize} B`,
    `maxComputeInvocationsPerWorkgroup : ${gpu.limits.maxComputeInvocationsPerWorkgroup}`,
  ].join('\n');
}

async function runSelfTestTab ( pane: HTMLElement, gpu: GpuContext ): Promise<void>
{
  pane.innerHTML = '';
  const envCard = el( 'section', 'card' );
  envCard.append( el( 'h2', undefined, '运行环境' ) );
  const envPre = el( 'pre', 'mono', '读取 GPU limits…' );
  envCard.append( envPre );
  pane.append( envCard );
  envPre.textContent = formatLimits( gpu );

  const testCard = el( 'section', 'card' );
  testCard.append( el( 'h2', undefined, '自检（GPU vs CPU 对拍）' ) );
  const list = el( 'ul', 'tests' );
  testCard.append( list );
  pane.append( testCard );

  await runSelfTest( gpu, ( result ) =>
  {
    const li = el( 'li', result.pass ? 'pass' : 'fail' );
    li.append( el( 'span', 'tag', result.pass ? 'PASS' : 'FAIL' ) );
    li.append( el( 'span', 'name', result.name ) );
    li.append( el( 'span', 'detail', result.detail ) );
    list.append( li );
  } );
}

async function boot(): Promise<void>
{
  const app = requireElement<HTMLDivElement>('#app');
  app.innerHTML = '';

  const header = el('header');
  header.append(el('h1', undefined, 'webllm · 自研浏览器 LLM 引擎'));
  header.append(el('p', 'sub', '训练台 · 预训练 / SFT / DPO / 对话 —— 全部在浏览器内完成'));
  app.append(header);

  // WebGPU 初始化（失败则整体报错退出）
  const gpuPre = el( 'pre', 'mono' );
  gpuPre.style.display = 'none';
  let gpu: GpuContext;
  try
  {
    gpu = await ( await import( './gpu/device' ) ).initGpu();
  }
  catch ( err )
  {
    gpuPre.style.display = 'block';
    gpuPre.className = 'mono bad';
    gpuPre.textContent = '初始化失败：' + ( err as Error ).message;
    app.append( gpuPre );
    return;
  }

  // Tab 切换
  const tabs = el( 'div', 'tabs' );
  const btnTrain = el( 'button', 'tab tab-active', '训练台' ) as HTMLButtonElement;
  const btnTests = el( 'button', 'tab', '自检（44 项）' ) as HTMLButtonElement;
  tabs.append( btnTrain, btnTests );
  app.append( tabs );

  const trainPane = el( 'div' );
  const testPane = el( 'div' );
  app.append( trainPane, testPane );

  function switchTab ( train: boolean ): void
  {
    btnTrain.className = train ? 'tab tab-active' : 'tab';
    btnTests.className = train ? 'tab' : 'tab tab-active';
    trainPane.style.display = train ? 'block' : 'none';
    testPane.style.display = train ? 'none' : 'block';
  }
  btnTrain.onclick = () => switchTab( true );
  btnTests.onclick = () => switchTab( false );

  renderTrainApp( gpu, trainPane );
  switchTab( true );

  // 自检在后台启动（切到自检 tab 时结果已在累积）
  void runSelfTestTab( testPane, gpu );
}

boot();