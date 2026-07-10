@echo off
REM ============================================================
REM  git_pull.bat - Fetch and merge latest changes from GitHub
REM ============================================================
setlocal
cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git is not installed or not on PATH.
    goto :end
)

if not exist ".git" (
    echo [ERROR] No git repository found. Run git_setup.bat first.
    goto :end
)

set "BRANCH="
for /f "delims=" %%b in ('git symbolic-ref --short HEAD 2^>nul') do set "BRANCH=%%b"
if "%BRANCH%"=="" set "BRANCH=main"
if /i "%BRANCH%"=="HEAD" set "BRANCH=main"

echo Pulling latest changes for branch "%BRANCH%" from origin...
git pull origin %BRANCH%
if errorlevel 1 (
    echo.
    echo [ERROR] Pull failed. Resolve any conflicts or check your connection.
) else (
    echo.
    echo Pull complete.
)

:end
echo.
pause
endlocal
