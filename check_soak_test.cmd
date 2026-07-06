@echo off
cd /d "%~dp0"
.\.venv\Scripts\python.exe scripts\soak_test_status.py
