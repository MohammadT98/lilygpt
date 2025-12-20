@echo off
REM Quick launcher for GPT-OSS-20B LoRA training

setlocal
pushd "%~dp0\.."

echo === Training GPT-OSS-20B with LoRA ===
uv run python -m lilynorm.stages.training.train ^
    --train "data/splits/train.jsonl" ^
    --val "data/splits/val.jsonl" ^
    --batch-size 4 ^
    --gradient-accumulation-steps 4 ^
    --learning-rate 2e-4 ^
    --epochs 3 ^
    --lora-r 8 ^
    --lora-alpha 32 ^
    --bf16 ^
    --save-steps 500 ^
    --eval-steps 500

popd
endlocal

pause
