@echo off
REM Generate continuation-style dataset from normalized files

setlocal
pushd "%~dp0\.."

echo ========================================
echo STEP 1: Generate Continuation Examples
echo ========================================
echo.
echo This will process all normalized .ly files
echo and create 3 training examples per piece.
echo.
echo Input:  data/normalized_dataset
echo Output: data/continuation_dataset/all_examples.jsonl
echo.

uv run python scripts/prepare_continuation_dataset.py ^
    --input "data/normalized_dataset" ^
    --output "data/continuation_dataset" ^
    --splits-per-piece 3

if errorlevel 1 (
    echo.
    echo ERROR: Dataset preparation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo STEP 2: Split into Train/Val/Test
echo ========================================
echo.

uv run python -c "
import json
import random
from pathlib import Path

print('Loading examples...')
with open('data/continuation_dataset/all_examples.jsonl') as f:
    examples = [json.loads(line) for line in f if line.strip()]

print(f'Total examples: {len(examples)}')

# Shuffle
random.seed(42)
random.shuffle(examples)

# Split: 80%% train, 10%% val, 10%% test
n = len(examples)
train_size = int(0.8 * n)
val_size = int(0.1 * n)

train = examples[:train_size]
val = examples[train_size:train_size + val_size]
test = examples[train_size + val_size:]

print(f'Train: {len(train)} examples')
print(f'Val:   {len(val)} examples')
print(f'Test:  {len(test)} examples')

# Save
Path('data/continuation_splits').mkdir(exist_ok=True)

for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
    output_path = f'data/continuation_splits/{split_name}.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in split_data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f'Saved {split_name}.jsonl')

print('')
print('Done!')
"

if errorlevel 1 (
    echo.
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
echo   - data/continuation_splits/train.jsonl
echo   - data/continuation_splits/val.jsonl
echo   - data/continuation_splits/test.jsonl
echo.
echo Next step: Update training script to use continuation dataset
echo See: CONTINUATION_TRAINING_GUIDE.md
echo.

popd
endlocal
pause
