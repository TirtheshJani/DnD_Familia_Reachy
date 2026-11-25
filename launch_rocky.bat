@echo off
setlocal
powershell -NoProfile -NoExit -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1" -AutoSession %*
