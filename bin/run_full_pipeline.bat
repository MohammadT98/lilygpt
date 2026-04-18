@echo off
setlocal EnableExtensions
pushd "%~dp0\.."

REM Pipeline: normalize LilyPond files -> build full-assignment dataset -> split train/val/test
REM Outputs:
REM   - data/normalized_dataset
REM   - data/assignment_dataset/all_examples.jsonl
REM   - data/splits_full/{train,val,test}.jsonl

set "INPUT_DIR=data/raw"
set "NORMALIZED_DIR=data/normalized_dataset"
set "FULL_DATASET_DIR=data/assignment_dataset"
set "SPLIT_DIR=data/splits_full"
set "PAUSE_ON_EXIT=1"

echo ========================================
echo STEP 1: Normalize Raw Files
echo ========================================
echo.
uv run python -m lilybench.cli --input "%INPUT_DIR%" --normalized-out "%NORMALIZED_DIR%"
if errorlevel 1 (
    echo ERROR: Normalization failed.
    set "EXIT_CODE=1"
    goto :cleanup
)

echo.
echo ========================================
echo STEP 2: Build Full Assignment Dataset
echo ========================================
echo.
uv run python -m lilybench.stages.dataset.build_assignment_dataset
if errorlevel 1 (
    echo ERROR: Full assignment dataset preparation failed.
    set "EXIT_CODE=1"
    goto :cleanup
)

echo.
echo ========================================
echo STEP 3: Split Train/Val/Test
echo ========================================
echo.
uv run python src/lilybench/stages/splitting/build_splits.py --input-jsonl "%FULL_DATASET_DIR%/all_examples.jsonl" --output-dir "%SPLIT_DIR%"
if errorlevel 1 (
    echo ERROR: Splitting failed.
    set "EXIT_CODE=1"
    goto :cleanup
)

echo.
echo ========================================
echo SUCCESS: DATASET READY
echo ========================================
echo.
echo Outputs:
echo   - %SPLIT_DIR%\train.jsonl
echo   - %SPLIT_DIR%\val.jsonl
echo   - %SPLIT_DIR%\test.jsonl
echo.

set "EXIT_CODE=0"

:cleanup
if /i "%PAUSE_ON_EXIT%"=="1" pause
popd
endlocal
exit /b %EXIT_CODE%
