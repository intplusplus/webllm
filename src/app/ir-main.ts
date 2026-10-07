/**
 * 模型编写台入口：只做两件事 —— 拿到 #app、把渲染交给 ir-app。
 *
 * 与 main.ts 一样，业务 DOM 全部由 TS 生成，页面 HTML 里不写业务结构；
 * 样式表在入口处 import，交给 Vite 打包（沿用现有页面的做法）。
 */
import { renderIrApp } from '../ui/ir-app';
import '../ui/styles.css';

const app = document.querySelector<HTMLDivElement>( '#app' );
if ( !app ) throw new Error( '缺少 #app 容器' );

renderIrApp( app );
