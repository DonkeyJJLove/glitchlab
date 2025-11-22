@echo off
setlocal
set "PY=C:\Python39\python.exe"
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "HOOK_DIR=%~dp0"
set "PYTHONPATH=%HOOK_DIR%;%PYTHONPATH%"
set "SCRIPT=%HOOK_DIR%prepare-commit-msg.py"
if not exist "%SCRIPT%" exit /b 0
"%PY%" "%SCRIPT%" %* 1>nul
exit /b %errorlevel%
