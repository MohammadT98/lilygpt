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

uv run python -c "import json; import random; from pathlib import Path; print('Loading examples...'); examples = [json.loads(line) for line in open('data/continuation_dataset/all_examples.jsonl', encoding='utf-8') if line.strip()]; print(f'Total examples: {len(examples)}'); random.seed(42); random.shuffle(examples); n = len(examples); train_size = int(0.8 * n); val_size = int(0.1 * n); train = examples[:train_size]; val = examples[train_size:train_size + val_size]; test = examples[train_size + val_size:]; print(f'Train: {len(train)} examples'); print(f'Val:   {len(val)} examples'); print(f'Test:  {len(test)} examples'); Path('data/splits').mkdir(exist_ok=True); [open(f'data/splits/{split_name}.jsonl', 'w', encoding='utf-8').writelines(json.dumps(ex, ensure_ascii=False) + '\n' for ex in split_data) for split_name, split_data in [('train', train), ('val', val), ('test', test)]]; print('Done!')"

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
