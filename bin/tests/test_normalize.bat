@echo off
REM Test processing: file_resolver + preparse + normalize

setlocal
pushd "%~dp0..\.."

echo === Testing file_resolver + preparse + normalize ===
echo.

uv run python scripts/test_normalize.py

popd
endlocal

echo.
pause
