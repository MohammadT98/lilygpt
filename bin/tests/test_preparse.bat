@echo off
REM Run minimal processing: file_resolver + preparse only

setlocal
pushd "%~dp0..\.."

echo === Running Minimal Processing (file_resolver + preparse) ===
echo.

uv run python scripts/tests/test_preparse.py

popd
endlocal

echo.
pause
