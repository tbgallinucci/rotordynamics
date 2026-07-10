@echo off
REM ============================================================
REM  git_push.bat - Push local commits to GitHub (origin/main)
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

REM Make sure at least one commit exists (otherwise there is nothing to push)
git rev-parse HEAD >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No commits yet - nothing to push.
    echo         Run git_commit.bat "your message" first to create your first commit.
    goto :end
)

REM Determine current branch (symbolic-ref works even before the first commit)
set "BRANCH="
for /f "delims=" %%b in ('git symbolic-ref --short HEAD 2^>nul') do set "BRANCH=%%b"
if "%BRANCH%"=="" set "BRANCH=main"
if /i "%BRANCH%"=="HEAD" set "BRANCH=main"

echo Pushing branch "%BRANCH%" to origin...
git push -u origin %BRANCH%
if errorlevel 1 (
    echo.
    echo [ERROR] Push failed. Check your network, credentials, or run git_pull.bat first.
) else (
    echo.
    echo Push complete.
)

:end
echo.
pause
endlocal
