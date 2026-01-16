@echo off
REM Run minimal processing: file_resolver + preprocess only

setlocal
pushd "%~dp0..\.."

echo === Running Minimal Processing (file_resolver + preprocess) ===
echo.

uv run python scripts/tests/test_preprocess.py

popd
endlocal

echo.
pause
