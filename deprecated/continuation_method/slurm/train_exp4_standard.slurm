#!/bin/bash
#SBATCH --job-name=lilygpt-exp4-standard
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/exp4-standard-%j.out
#SBATCH --error=logs/exp4-standard-%j.err
#SBATCH --qos=allgroups_qos

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

# Activate venv (DO NOT load python modules)
source "$HOME/lilygpt/.venv/bin/activate"

echo "================================================================================"
echo "EXPERIMENT 4: STANDARD TRAINING (No Loss Masking)"
echo "================================================================================"
echo "This experiment uses the STANDARD domain adaptation approach:"
echo "  - Train on FULL sequences (all tokens contribute to loss)"
echo "  - No artificial input/output masking with -100"
echo "  - Same hyperparameters as exp3 for fair comparison"
echo ""
echo "Difference from exp3:"
echo "  exp3: labels = [-100] * input_len + output_tokens (masked continuation)"
echo "  exp4: labels = input_ids (standard full-sequence training)"
echo "================================================================================"
echo ""

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

python -m lilynorm.stages.training.train_standard \
    --train "data/splits/train.jsonl" \
    --val "data/splits/val.jsonl" \
    --output-dir "runs/lora_exp4_standard" \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --learning-rate 5e-5 \
    --epochs 3 \
    --max-length 2048 \
    --lora-r 8 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --bf16 \
    --save-steps 250 \
    --eval-steps 250 \
    --logging-steps 10 \
    --warmup-steps 100

echo ""
echo "End time: $(date)"
echo "=== Job Complete ==="
