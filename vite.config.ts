import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    // 0.0.0.0 让同一 WiFi 下的手机也能访问（联邦联训 demo 需要）
    host: true,
    port: 5173,
  },
  build: {
    target: 'es2022',
    // 多页入口：index.html 是单机训练台，fed.html 是联邦联训 Demo。
    // 用相对路径（相对 vite root，默认即项目根目录），避免在 ESM 配置里引入 node 类型。
    rollupOptions: {
      input: {
        index: 'index.html',
        fed: 'fed.html',
      },
    },
  },
});