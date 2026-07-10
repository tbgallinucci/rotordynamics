@echo off
REM ============================================================
REM  git_setup.bat - One-time initialization of the local repo
REM  Links this folder to https://github.com/tbgallinucci/rotordynamics
REM ============================================================
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Rotordynamic FEA - Git setup
echo ============================================================
echo.

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git is not installed or not on PATH.
    echo         Install it from https://git-scm.com/download/win
    goto :end
)

if exist ".git" (
    echo Repository already initialized in this folder.
) else (
    echo Initializing new git repository...
    git init
    git branch -M main
)

REM Configure the remote "origin" (add if missing, update if present)
git remote get-url origin >nul 2>nul
if errorlevel 1 (
    echo Adding remote origin...
    git remote add origin https://github.com/tbgallinucci/rotordynamics.git
) else (
    echo Updating remote origin URL...
    git remote set-url origin https://github.com/tbgallinucci/rotordynamics.git
)

echo.
echo Current remotes:
git remote -v
echo.
echo Setup complete. You can now use git_commit.bat, git_push.bat and git_pull.bat.

:end
echo.
pause
endlocal
