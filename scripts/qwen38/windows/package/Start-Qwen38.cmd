@echo off
setlocal
title Qwen3.8-27B GSQ-RCO - NInfer
rem Starts the server in the ninfer-5060-qwen WSL distribution that Install.cmd set up, with the
rem launcher inside it (windows\Start-Qwen38.ps1 of the Linux release). This terminal owns the
rem server: Ctrl+C or closing it stops the server and releases GPU memory.
rem Arguments pass through, for example: Start-Qwen38.cmd -Concurrency 4
wsl.exe --distribution ninfer-5060-qwen --exec /bin/true >nul 2>&1
if not "%ERRORLEVEL%"=="0" (
    echo NInfer-Qwen is not installed. Run Install.cmd from the extracted Windows release first.
    pause
    exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "\\wsl$\ninfer-5060-qwen\opt\ninfer-5060-qwen\windows\Start-Qwen38.ps1" %*
set "ninfer_exit=%ERRORLEVEL%"
if not "%ninfer_exit%"=="0" (
    echo.
    echo Qwen3.8 could not start. Read the error above and the logs in %LOCALAPPDATA%\ninfer-5060-qwen\logs.
    pause
)
endlocal & exit /b %ninfer_exit%
