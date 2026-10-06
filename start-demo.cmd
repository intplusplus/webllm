@echo off
rem ============================================================
rem  公共训练网络 - 联邦联训 demo 一键启动
rem  双击本文件即可：会弹出一个窗口，打印手机要打开的局域网地址。
rem  关掉这个窗口 = 停止服务。
rem ============================================================
cd /d "%~dp0"

set NODE_EXE=C:\Users\intpp\.workbuddy\binaries\node\versions\22.22.2\node.exe
if not exist "%NODE_EXE%" set NODE_EXE=node

echo.
echo   正在启动信令服务器 + 开发服务器...
echo   （窗口保持打开就一直在跑；停止请直接关窗口）
echo.
"%NODE_EXE%" scripts\start-demo.mjs
echo.
echo   服务已退出。如果上面有报错，把内容发给助手。
pause
