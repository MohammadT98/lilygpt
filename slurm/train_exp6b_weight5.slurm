#!/bin/bash
#SBATCH --job-name=lilygpt-exp6b-weight5
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/exp6b-weight5-%j.out
#SBATCH --error=logs/exp6b-weight5-%j.err
#SBATCH --qos=allgroups_qos

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

# Activate venv (DO NOT load python modules)
source "$HOME/lilygpt/.venv/bin/activate"

echo "================================================================================"
echo "EXPERIMENT 6b: WEIGHTED LOSS (5x Weight - Reduced from 20x)"
echo "================================================================================"
echo "After exp6 (20x weight) generated too many closing braces, we reduce to 5x:"
echo "  - Same dataset as exp5 and exp6 (full assignment)"
echo "  - Same hyperparameters as exp5 and exp6"
echo "  - ONLY CHANGE: Closing brace } has 5x higher loss weight (instead of 20x)"
echo "  - Final newline before } has 1.25x higher loss weight"
echo ""
echo "Hypothesis being tested:"
echo "  5x weight is the sweet spot - model learns to generate } without overdoing it"
echo ""
echo "Comparison:"
echo "  exp5:  No weighting - never generates }"
echo "  exp6:  20x weight - generates } everywhere (12 times!)"
echo "  exp6b: 5x weight - hopefully generates } once at the right place"
echo "================================================================================"
echo ""

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

echo "=== Dataset Statistics ==="
echo "Train: 1195 examples from 41 works"
echo "Val: 107 examples from 5 works"
echo "Test: 201 examples from 6 works"
echo "Total: 1503 full assignments (same as exp5 and exp6)"
echo ""

python -m lilynorm.stages.training.train_weighted \
    --train "data/splits_full/train.jsonl" \
    --val "data/splits_full/val.jsonl" \
    --output-dir "runs/lora_exp6b_weight5" \
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
