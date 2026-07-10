@echo off
REM ============================================================
REM  git_sync.bat - Commit + pull + push in one step
REM  Usage:
REM    git_sync.bat "Your commit message"
REM    git_sync.bat            (will prompt for a message)
REM ============================================================
setlocal enabledelayedexpansion
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

set "MSG=%~1"
if "%MSG%"=="" (
    set /p "MSG=Enter commit message: "
)
if "%MSG%"=="" (
    echo [ERROR] Commit message cannot be empty.
    goto :end
)

echo [1/3] Staging and committing...
git add -A
git commit -m "%MSG%"

REM Resolve branch AFTER committing (before the first commit it may be unborn)
set "BRANCH="
for /f "delims=" %%b in ('git symbolic-ref --short HEAD 2^>nul') do set "BRANCH=%%b"
if "%BRANCH%"=="" set "BRANCH=main"
if /i "%BRANCH%"=="HEAD" set "BRANCH=main"

echo.
echo [2/3] Pulling latest from origin/%BRANCH%...
git pull origin %BRANCH% --no-edit --allow-unrelated-histories
if errorlevel 1 (
    echo [ERROR] Pull failed - resolve conflicts before pushing.
    goto :end
)

echo.
echo [3/3] Pushing to origin/%BRANCH%...
git push -u origin %BRANCH%
if errorlevel 1 (
    echo [ERROR] Push failed.
) else (
    echo.
    echo Sync complete.
)

:end
echo.
pause
endlocal
