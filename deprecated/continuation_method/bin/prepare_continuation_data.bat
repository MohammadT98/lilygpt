@echo off
REM Generate continuation-style dataset with comprehensive logging

setlocal
pushd "%~dp0\.."

uv run python scripts/prepare_continuation_with_log.py

if errorlevel 1 (
    echo.
    echo Pipeline failed! Check the log file in data/logs/continuation_pipeline/
    pause
    exit /b 1
)

popd
endlocal
pause
