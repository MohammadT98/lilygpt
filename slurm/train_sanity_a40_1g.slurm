#!/bin/bash
#SBATCH --job-name=lily-sanity
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/sanity-%j.out
#SBATCH --error=logs/sanity-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs
source "$HOME/lilygpt/.venv/bin/activate"

python -m lilynorm.stages.training.train \
  --train "data/splits/train.jsonl" \
  --val   "data/splits/val.jsonl" \
  --model-name "gpt2" \
  --output-dir "runs/sanity_gpt2_a40_1g" \
  --epochs 1 \
  --max-length 512 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --lora-r 4 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --bf16 \
  --save-steps 50 \
  --eval-steps 50 \
  --logging-steps 10
