@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass ^
  -File "%~dp0scripts\stop_soak_test.ps1"
