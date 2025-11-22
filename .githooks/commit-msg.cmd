@echo off
setlocal
set "PY=C:\Python39\python.exe"
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "HOOK_DIR=%~dp0"
set "PYTHONPATH=%HOOK_DIR%;%PYTHONPATH%"
set "SCRIPT=%HOOK_DIR%commit-msg.py"
if not exist "%SCRIPT%" exit /b 0
if not "%~1"=="" "%PY%" -c "import sys,os; p=sys.argv[1]; b=open(p,'rb').read(); b=b.lstrip(b'\xef\xbb\xbf'); open(p,'wb').write(b)" "%~1"
"%PY%" "%SCRIPT%" %*
exit /b %errorlevel%
