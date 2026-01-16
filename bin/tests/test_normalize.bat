@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0..\.."

REM Test: file_resolver + preprocess + normalize

set "SCRIPT_PATH=scripts/tests/test_normalize.py"
set "LOG_DIR=data\logs"
set "TEST_NAME=test_normalize"
set "PAUSE_ON_EXIT=1"

call :timestamp

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG_FILE=%LOG_DIR%\%TEST_NAME%_%TIMESTAMP%.log"

echo ========================================
echo TEST: Normalize
echo ========================================
echo.
echo Running %SCRIPT_PATH%
echo.

uv run python "%SCRIPT_PATH%" > "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

type "%LOG_FILE%"

echo.
echo Log saved to: %LOG_FILE%
echo.

if %EXIT_CODE% NEQ 0 (
    echo ERROR: Test failed (exit code %EXIT_CODE%).
)

if /i "%PAUSE_ON_EXIT%"=="1" pause
popd
endlocal
exit /b %EXIT_CODE%

:timestamp
for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss" 2^>nul') do set "TIMESTAMP=%%i"
if not defined TIMESTAMP (
    set "TIMESTAMP=%date:~-4%-%date:~-10,2%-%date:~-7,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%"
    set "TIMESTAMP=%TIMESTAMP: =0%"
)
exit /b 0
