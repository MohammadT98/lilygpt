@echo off
REM NEW PIPELINE: Normalize → Continuation Examples → Train/Val/Test
REM This creates 3x more training data using continuation-style training

setlocal
pushd "%~dp0\.."

echo ========================================
echo STEP 1: Normalize Raw Files
echo ========================================
echo.
uv run python -m lilynorm.cli --input "data/raw" --normalized-out "data/normalized_dataset" --skip-tokenize

if errorlevel 1 (
    echo ERROR: Normalization failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo STEP 2: Generate Continuation Examples
echo ========================================
echo.
uv run python scripts/prepare_continuation_dataset.py --input "data/normalized_dataset" --output "data/continuation_dataset" --splits-per-piece 3

if errorlevel 1 (
    echo ERROR: Dataset preparation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo STEP 3: Split into Train/Val/Test
echo ========================================
echo.

uv run python src/lilynorm/stages/splitting/build_splits.py --input-jsonl "data/continuation_dataset/all_examples.jsonl" --output-dir "data/splits"

if errorlevel 1 (
    echo ERROR: Splitting failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS!
echo ========================================
echo.
echo Dataset ready at:
echo   - data/splits/train.jsonl
echo   - data/splits/val.jsonl
echo   - data/splits/test.jsonl
echo.

popd
endlocal
pause
