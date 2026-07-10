@echo off
REM ============================================================
REM  git_commit.bat - Stage all changes and create a commit
REM  Usage:
REM    git_commit.bat "Your commit message"
REM    git_commit.bat            (will prompt for a message)
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

REM Get commit message from argument, or prompt for one
set "MSG=%~1"
if "%MSG%"=="" (
    set /p "MSG=Enter commit message: "
)
if "%MSG%"=="" (
    echo [ERROR] Commit message cannot be empty.
    goto :end
)

echo.
echo Staging all changes...
git add -A

echo.
echo Status:
git status --short

echo.
echo Committing with message: "%MSG%"
git commit -m "%MSG%"
if errorlevel 1 (
    echo.
    echo [INFO] Nothing was committed (no changes, or commit failed).
)

:end
echo.
pause
endlocal
