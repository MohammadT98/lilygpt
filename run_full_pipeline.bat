@echo off
REM Run normalization + tokenization + splitting

setlocal
pushd "%~dp0"

echo === Normalizing + Tokenizing Dataset ===
uv run python -m scripts.process_dataset --input "data/raw"

echo.
echo === Building Train/Val/Test Splits ===
uv run python -m lilynorm.stages.splitting.build_splits --tokenized-root "data/tokenized_dataset" --output-dir "data/splits"

popd
endlocal

echo.
echo === ALL DONE ===
pause
