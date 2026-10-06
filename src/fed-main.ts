/**
 * 联邦联训入口。
 *
 * 三个页面共用这一个入口，靠路径区分视图：
 *   /host.html —— 房主页：创建并设计房间（任务、模型、参数都由房主定义）
 *   /join.html —— 节点页：浏览开放房间、加入、只读地看进度与成果
 *   /fed.html  —— 着陆页：两块大按钮，指到上面两个页面
 *
 * 为什么要分开：让一个人在同一页里既当房主又当节点，结果就是谁都不知道
 * 自己现在该点哪个按钮。职责不同，页面就该不同。
 */
import { renderFedApp } from './ui/fed-app';

const app = document.querySelector<HTMLDivElement>( '#app' );
if ( !app ) throw new Error( '缺少 #app 容器' );

const path = location.pathname.split( '/' ).pop() ?? '';
// 单文件版（file://）没有路径可区分，由打包脚本注入 __WEBLLM_FED_VIEW__ 指定视图
const forced = ( window as unknown as { __WEBLLM_FED_VIEW__?: 'host' | 'join' } ).__WEBLLM_FED_VIEW__;
const view = forced ?? ( path === 'host.html' ? 'host' : path === 'join.html' ? 'join' : null );

app.innerHTML = '';

if ( view )
{
  renderFedApp( app, view );
}
else
{
  // 着陆页
  const head = document.createElement( 'div' );
  head.className = 'fed-head';
  const h = document.createElement( 'h1' );
  h.textContent = '公共训练网络 · 联邦联训';
  const p = document.createElement( 'p' );
  p.textContent = '选择你的角色。两页随时可以互相切换 —— 谁都可以当房主，也都可以只当训练节点。';
  head.append( h, p );
  app.append( head );

  const wrap = document.createElement( 'div' );
  wrap.style.display = 'grid';
  wrap.style.gap = '12px';

  const mk = ( href: string, title: string, desc: string ): void =>
  {
    const a = document.createElement( 'a' );
    a.href = href;
    a.style.cssText = 'display:block;padding:18px;border:1px solid #1e2635;border-radius:10px;' +
      'background:#121826;color:#e6edf3;text-decoration:none';
    const t = document.createElement( 'b' );
    t.style.fontSize = '16px';
    t.textContent = title;
    const d = document.createElement( 'div' );
    d.style.cssText = 'color:#8b98a9;font-size:13px;margin-top:4px';
    d.textContent = desc;
    a.append( t, d );
    wrap.append( a );
  };

  mk( '/host.html', '当房主（创建房间）', '设计任务、模型与参数，开训、聚合、下发全局权重' );
  mk( '/join.html', '当训练节点（加入房间）', '浏览开放房间点一下就进；数据不出本地，只交换权重' );
  app.append( wrap );
}
