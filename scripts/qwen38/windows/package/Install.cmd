@echo off
setlocal
title NInfer-Qwen setup
rem Installs NInfer-Qwen as its own WSL2 distribution and downloads the model (scripts\Install.ps1).
rem Arguments pass through, for example: Install.cmd -InstallDir D:\NInfer-Qwen
if not exist "%~dp0scripts\Install.ps1" (
    echo Extract the whole zip first ^(right-click it, Extract All^), then run Install.cmd from the extracted folder.
    pause
    exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Install.ps1" %*
set "ninfer_exit=%ERRORLEVEL%"
echo.
if not "%ninfer_exit%"=="0" echo Setup stopped. Read the message above; running Install.cmd again continues where it stopped.
pause
endlocal & exit /b %ninfer_exit%
