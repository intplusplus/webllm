/** 联邦联训 Demo 入口：/fed.html */
import { renderFedApp } from './ui/fed-app';

const app = document.querySelector<HTMLDivElement>( '#app' );
if ( !app ) throw new Error( '缺少 #app 容器' );
app.innerHTML = '';
renderFedApp( app );
