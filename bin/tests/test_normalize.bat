@echo off
REM Test processing: file_resolver + preparse + normalize

setlocal enabledelayedexpansion
pushd "%~dp0..\.."

REM Create timestamped log filename
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set datetime=%%i
set logfile=data\logs\test_normalize_%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%_%datetime:~8,2%-%datetime:~10,2%-%datetime:~12,2%.log

REM Ensure logs directory exists
if not exist "data\logs" mkdir "data\logs"

echo === Testing file_resolver + preparse + normalize ===
echo.

uv run python scripts/tests/test_normalize.py >> "!logfile!" 2>&1
type "!logfile!"

echo.
echo === Log saved to: !logfile! ===

popd
endlocal

echo.
pause
