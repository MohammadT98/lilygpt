@echo off
REM Run the standard dataset processor from the repository root.
setlocal
pushd "%~dp0"
uv run python -m scripts.process_dataset --input "data/raw/Dataset" %*
popd
endlocal
echo.
pause
