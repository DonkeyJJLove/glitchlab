@echo off
setlocal
set "PY=C:\Python39\python.exe"

rem UTF-8
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

set "HOOK_DIR=%~dp0"

rem Ustal root repo (git) z fallbackiem
set "REPO="
for /f "usebackq delims=" %%R in (`git rev-parse --show-toplevel 2^>nul`) do set "REPO=%%R"
if not defined REPO set "REPO=%CD%"

rem PYTHONPATH: repo, src/, glitchlab/, .githooks
set "PYTHONPATH=%HOOK_DIR%;%REPO%;%REPO%\src;%REPO%\glitchlab;%PYTHONPATH%"

rem Jeśli nie ma żadnego .env w repo, spróbuj skopiować z rodzica (tymczasowo)
set "HAD_ENV=" & set "HAD_ENV_LOCAL="
if exist "%REPO%\.env" set "HAD_ENV=1"
if exist "%REPO%\.env.local" set "HAD_ENV_LOCAL=1"
set "PARENT=%REPO%\.."
set "CREATED_ENV_COPY=" & set "CREATED_ENV_LOCAL_COPY="
if not defined HAD_ENV       if exist "%PARENT%\.env"       copy /Y "%PARENT%\.env"       "%REPO%\.env"       >nul && set "CREATED_ENV_COPY=1"
if not defined HAD_ENV_LOCAL if exist "%PARENT%\.env.local" copy /Y "%PARENT%\.env.local" "%REPO%\.env.local" >nul && set "CREATED_ENV_LOCAL_COPY=1"

rem Zapewnij GLX_ROOT w .env.local (tymczasowo)
set "LOCAL_CREATED_FOR_GLXROOT=" & set "LOCAL_BACKED_UP_FOR_GLXROOT="
if not exist "%REPO%\.env.local" (
  > "%REPO%\.env.local" echo GLX_ROOT=%REPO%
  set "LOCAL_CREATED_FOR_GLXROOT=1"
) else (
  findstr /R /C:"^GLX_ROOT=" "%REPO%\.env.local" >nul
  if errorlevel 1 (
    copy /Y "%REPO%\.env.local" "%REPO%\.env.local.__hook_bak" >nul
    >> "%REPO%\.env.local" echo GLX_ROOT=%REPO%
    set "LOCAL_BACKED_UP_FOR_GLXROOT=1"
  )
)

rem Jeśli nadal brak obu .env → NO-OP
if not exist "%REPO%\.env" if not exist "%REPO%\.env.local" goto :cleanup_success

rem >>> Jeżeli gateway NIE istnieje w repo (po ścieżkach), NO-OP <<<
set "HAS_GATEWAY="
if exist "%REPO%\glitchlab\analysis\autonomy\gateway.py" set "HAS_GATEWAY=1"
if exist "%REPO%\glitchlab\analysis\autonomy\gateway\__init__.py" set "HAS_GATEWAY=1"
if exist "%REPO%\analysis\autonomy\gateway.py" set "HAS_GATEWAY=1"
if exist "%REPO%\analysis\autonomy\gateway\__init__.py" set "HAS_GATEWAY=1"
if not defined HAS_GATEWAY goto :cleanup_success

rem Skoro moduł istnieje – odpal post-commit.py (jeśli jest)
set "SCRIPT=%HOOK_DIR%post-commit.py"
if not exist "%SCRIPT%" goto :cleanup_success

pushd "%REPO%" >nul
"%PY%" "%SCRIPT%"
set "RC=%ERRORLEVEL%"
popd >nul
goto :cleanup

:cleanup_success
set "RC=0"

:cleanup
rem Sprzątanie tylko tego, co utworzył wrapper
if defined CREATED_ENV_COPY       if exist "%REPO%\.env" del /Q "%REPO%\.env" >nul
if defined CREATED_ENV_LOCAL_COPY if exist "%REPO%\.env.local" (
  if not defined LOCAL_BACKED_UP_FOR_GLXROOT if not defined LOCAL_CREATED_FOR_GLXROOT del /Q "%REPO%\.env.local" >nul
)
if defined LOCAL_CREATED_FOR_GLXROOT if exist "%REPO%\.env.local" del /Q "%REPO%\.env.local" >nul
if defined LOCAL_BACKED_UP_FOR_GLXROOT if exist "%REPO%\.env.local.__hook_bak" move /Y "%REPO%\.env.local.__hook_bak" "%REPO%\.env.local" >nul

exit /b %RC%
