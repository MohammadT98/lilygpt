@echo off
REM EXPERIMENT 5 PIPELINE: Normalize → Full Assignments → Train/Val/Test
REM This creates complete structural examples (1 per assignment) for exp5

setlocal
pushd "%~dp0\.."

echo ========================================
echo STEP 1: Normalize Raw Files
echo ========================================
echo.
uv run python -m lilynorm.cli --input "data/raw" --normalized-out "data/normalized_dataset"

if errorlevel 1 (
    echo ERROR: Normalization failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo STEP 2: Generate Full Assignment Examples
echo ========================================
echo.
uv run python scripts/prepare_full_assignment_dataset.py

if errorlevel 1 (
    echo ERROR: Full assignment dataset preparation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo STEP 3: Split into Train/Val/Test
echo ========================================
echo.

uv run python src/lilynorm/stages/splitting/build_splits.py --input-jsonl "data/full_assignment_dataset/all_examples.jsonl" --output-dir "data/splits_full"

if errorlevel 1 (
    echo ERROR: Splitting failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! EXP5 DATASET READY
echo ========================================
echo.
echo Full assignment dataset ready at:
echo   - data/splits_full/train.jsonl
echo   - data/splits_full/val.jsonl
echo   - data/splits_full/test.jsonl
echo.

popd
endlocal
pause
