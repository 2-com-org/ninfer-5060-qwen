@echo off
rem Removes NInfer-Qwen, its WSL distribution and the downloaded model (scripts\Uninstall.ps1 asks
rem first and pauses at the end). Arguments pass through, for example: Uninstall.cmd -Yes
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Uninstall.ps1" %*
