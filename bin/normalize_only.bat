@echo off
REM Normalize raw LilyPond files (cleanup only, no tokenization)

setlocal
pushd "%~dp0\.."

echo ========================================
echo STEP 1: Normalize Raw Files
echo ========================================
echo.
echo This will clean up raw .ly files by:
echo   - Removing engraving commands
echo   - Fixing syntax issues
echo   - Preserving musical content
echo.
echo Input:  data/raw
echo Output: data/normalized_dataset
echo.

uv run python scripts/process_dataset.py --input "data/raw" --normalized-out "data/normalized_dataset" --skip-tokenize

if errorlevel 1 (
    echo.
    echo ERROR: Normalization failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS!
echo ========================================
echo.
echo Normalized files saved to: data/normalized_dataset
echo.
echo Next step: Run bin\prepare_continuation_data.bat
echo.

popd
endlocal
pause
