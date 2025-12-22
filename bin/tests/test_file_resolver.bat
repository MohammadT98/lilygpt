@echo off
REM Run file_resolver test and save log

setlocal enabledelayedexpansion
pushd "%~dp0..\.."

echo === Running File Resolver Test ===
echo.

set "logfile=data\logs\test_file_resolver_%date:~-4%-%date:~-10,2%-%date:~-7,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%.log"
set "logfile=!logfile: =0!"

mkdir data\logs 2>nul

echo Test started at %date% %time% > "!logfile!"
echo. >> "!logfile!"

REM Run and capture to log, then display log to console
uv run python scripts/test_file_resolver.py >> "!logfile!" 2>&1
type "!logfile!"

echo.
echo === Log saved to: !logfile! ===
echo.

popd
endlocal

pause
