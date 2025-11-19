@echo off
REM Run the dataset processor and keep only the first detected voice per file.
setlocal
pushd "%~dp0"
uv run python -m scripts.process_dataset --input "data/raw" --single-voice-only %*
popd
endlocal
echo.
pause
