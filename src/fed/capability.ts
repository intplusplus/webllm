/**
 * 设备能力探测 + WebGPU 上下文缓存。
 *
 * 为什么要有这个模块：
 *   1. WebGPU 需要**安全上下文**。`http://192.168.x.x` 不是安全上下文，
 *      手机上 `navigator.gpu` 直接不存在 —— 这是「手机能不能用 WebGPU 训练」
 *      的第一个门槛，必须在界面上如实告诉用户怎么解决。
 *   2. `navigator.gpu` 存在 ≠ 能用：拿不到适配器（驱动/策略禁用）同样白搭。
 *   3. 引擎构造需要 GPU 上下文，而 `createEngine` 想保持同步（避免把整条
 *      编排链路改成 async），所以这里把上下文缓存在模块里：
 *      页面启动时先探测一次，之后同步取用。
 */
import { initGpu, type GpuContext } from '../gpu/device';

export interface Capability
{
  kind: string;
  cores: number;
  memoryGB: number;
  /** 页面是否处于安全上下文（https 或 localhost） */
  secureContext: boolean;
  /** navigator.gpu 是否存在 */
  hasWebGpuApi: boolean;
  /** 是否成功拿到适配器 */
  adapterOk: boolean;
  adapterError: string | null;
  vendor: string;
  architecture: string;
  device: string;
  hasF16: boolean;
  maxBufferSize: number;
  /** 面向用户的可执行建议 */
  advise: string[];
}

let ready: GpuContext | null = null;
let pending: Promise<GpuContext> | null = null;
let lastError: string | null = null;

/** 取得（必要时初始化）WebGPU 上下文。失败会抛，并记下原因。 */
export function ensureGpu (): Promise<GpuContext>
{
  if ( ready ) return Promise.resolve( ready );
  if ( !pending )
  {
    pending = initGpu()
      .then( ( g ) =>
      {
        ready = g;
        lastError = null;
        return g;
      } )
      .catch( ( err: unknown ) =>
      {
        pending = null;
        lastError = ( err as Error ).message;
        throw err;
      } );
  }
  return pending;
}

/** 同步取已就绪的上下文；没就绪返回 null（引擎工厂用）。 */
export function gpuIfReady (): GpuContext | null
{
  return ready;
}

export function gpuLastError (): string | null
{
  return lastError;
}

/** 根据 UA 粗判设备类别（仅用于展示）。 */
export function detectKind (): string
{
  const ua = typeof navigator === 'undefined' ? '' : navigator.userAgent;
  if ( /Android/i.test( ua ) ) return 'Android';
  if ( /iPhone|iPad|iPod/i.test( ua ) ) return 'iOS';
  if ( /Macintosh/i.test( ua ) ) return 'macOS';
  if ( /Windows/i.test( ua ) ) return 'Windows';
  if ( /Linux/i.test( ua ) ) return 'Linux';
  return '未知设备';
}

/**
 * 探测本机能力。会把 WebGPU 上下文一并初始化好（成功则后续同步可用）。
 * 无论成功失败都返回报告 —— 失败信息本身就是要展示给用户的内容。
 */
export async function probeCapability (): Promise<Capability>
{
  // deviceMemory 是非标准字段，只有部分浏览器有
  type NavEx = Navigator & { deviceMemory?: number };
  const nav = ( typeof navigator === 'undefined' ? {} : navigator ) as NavEx;

  const base: Capability = {
    kind: detectKind(),
    cores: nav.hardwareConcurrency ?? 0,
    memoryGB: nav.deviceMemory ?? 0,
    secureContext: typeof window !== 'undefined' ? window.isSecureContext : false,
    hasWebGpuApi: typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined',
    adapterOk: false,
    adapterError: null,
    vendor: '',
    architecture: '',
    device: '',
    hasF16: false,
    maxBufferSize: 0,
    advise: [],
  };

  try
  {
    const gpu = await ensureGpu();
    base.adapterOk = true;
    const info = gpu.adapterInfo ?? {};
    base.vendor = info.vendor ?? '';
    base.architecture = info.architecture ?? '';
    base.device = info.device ?? info.description ?? '';
    base.hasF16 = gpu.hasF16;
    base.maxBufferSize = gpu.limits.maxBufferSize;
  }
  catch ( err )
  {
    base.adapterError = ( err as Error ).message;
  }

  // ---- 给出可执行建议 ----
  if ( !base.secureContext && !base.hasWebGpuApi )
  {
    base.advise.push(
      '页面不是安全上下文，浏览器把 WebGPU 藏起来了。二选一：',
      '① 最快：手机 Chrome 打开 chrome://flags/#unsafely-treat-insecure-origin-as-secure，' +
      '把本页地址（形如 http://192.168.1.5:5173）填进去，选 Enabled，重启浏览器。',
      '② 更正规：用 npm run demo --https 起一个 https 服务，并按提示在手机上安装一次本地 CA 证书。',
    );
  }
  else if ( base.hasWebGpuApi && !base.adapterOk )
  {
    base.advise.push(
      `浏览器暴露了 navigator.gpu 但取不到适配器：${ base.adapterError ?? '未知原因' }`,
      '常见原因：显卡驱动过旧、浏览器把 WebGPU 关掉了（chrome://gpu 可看状态）、或虚拟机/远程桌面里没有可用 GPU。',
    );
  }
  else if ( base.adapterOk && !base.hasF16 )
  {
    base.advise.push( '适配器不支持 shader-f16：可以训练，但只能走 fp32 路径（更慢、更占显存）。' );
  }

  return base;
}
