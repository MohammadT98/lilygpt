#!/bin/bash
#SBATCH --job-name=lilygpt-exp7-weight5-keys
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/exp7-weight5-keys-%j.out
#SBATCH --error=logs/exp7-weight5-keys-%j.err
#SBATCH --qos=allgroups_qos

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

# Activate venv (DO NOT load python modules)
source "$HOME/lilygpt/.venv/bin/activate"

echo "================================================================================"
echo "EXPERIMENT 7 (weight 5) + cleaned keys"
echo "================================================================================"
echo "Same setup as exp6b (5x brace weight) but using the cleaned dataset with key/tempo"
echo "and scheme-fragment removals. Objective: keep brace emphasis while training on the"
echo "latest normalized data (no \key noise, no measureLength, no #' fragments)."
echo "================================================================================"
echo ""

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

echo "=== Dataset ==="
echo "Train/Val splits: data/splits_full"
echo "Using cleaned normalized_dataset (keys/time/tempo cleaned, scheme fragments removed)."
echo ""

python -m lilynorm.stages.training.train_weighted \
    --train "data/splits_full/train.jsonl" \
    --val "data/splits_full/val.jsonl" \
    --output-dir "runs/lora_exp7_weight5_keys" \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --learning-rate 5e-5 \
    --epochs 3 \
    --max-length 2048 \
    --lora-r 8 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --structural-weight 5.0 \
    --bf16 \
    --save-steps 250 \
    --eval-steps 250 \
    --logging-steps 10 \
    --warmup-steps 100

echo ""
echo "End time: $(date)"
echo "=== Job Complete ==="
