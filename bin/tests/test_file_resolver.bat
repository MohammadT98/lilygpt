@echo off
REM Debug preprocessing stages for all files - saves output after each stage

setlocal enabledelayedexpansion
pushd "%~dp0..\.."

echo === Running Debug Stages on All Raw Files ===
echo.

set count=0
for /r "data\raw" %%f in (*.ly) do (
    set /a count+=1
    echo [!count!] Processing: %%f
    uv run python scripts/test_file_resolver.py "%%f"
    echo.
)

echo.
echo === Processed !count! files ===
echo Output saved to data/test_file_resolver/

popd
endlocal

echo.
pause
