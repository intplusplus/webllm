@echo off
rem 计划任务入口：每小时双远端同步（本地优先）。日志追加到 %USERPROFILE%\.webllm-sync.log
cd /d "%~dp0.."
node "%~dp0sync-remotes.mjs"
exit /b %ERRORLEVEL%
