#!/bin/bash
#SBATCH --job-name=lily-20b-lora
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/20b-train-%j.out
#SBATCH --error=logs/20b-train-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

source "$HOME/lilygpt/.venv/bin/activate"

# Helps CUDA fragmentation on long jobs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m lilynorm.stages.training.train \
  --train "data/splits/train.jsonl" \
  --val "data/splits/val.jsonl" \
  --model-name "openai/gpt-oss-20b" \
  --output-dir "runs/20b_lora_a40_2g" \
  --epochs 3 \
  --max-length 1024 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --lora-r 8 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --bf16 \
  --save-steps 100 \
  --eval-steps 100 \
  --logging-steps 10 \
  --warmup-steps 100