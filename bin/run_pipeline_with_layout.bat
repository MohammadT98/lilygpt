@echo off
REM Run normalization + tokenization + splitting
REM Keeps layout/midi/score blocks for PDF compilation

setlocal
pushd "%~dp0\.."

REM Set environment variable to keep layout blocks
set LILYNORM_KEEP_LAYOUT=1

echo === Normalizing + Tokenizing Dataset (keeping layout for PDF) ===
uv run python -m scripts.process_dataset --input "data/raw"

echo.
echo === Building Train/Val/Test Splits ===
uv run python -m lilynorm.stages.splitting.build_splits --tokenized-root "data/tokenized_dataset" --output-dir "data/splits"

popd
endlocal

echo.
echo === ALL DONE ===
pause
