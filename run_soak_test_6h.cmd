@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass ^
  -File "%~dp0scripts\start_soak_test.ps1"
