@echo off
setlocal
title Qwen3.8-27B GSQ-RCO - NInfer
rem This terminal owns the server. Ctrl+C or closing it stops the server and releases GPU memory.
rem Arguments pass through to Start-Qwen38.ps1, for example: Start-Qwen38.cmd -Concurrency 4
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Qwen38.ps1" %*
set "ninfer_exit=%ERRORLEVEL%"
if not "%ninfer_exit%"=="0" (
    echo.
    echo Qwen3.8 could not start. Read the error above and the logs in %LOCALAPPDATA%\ninfer-qwen\logs.
    pause
)
endlocal & exit /b %ninfer_exit%
