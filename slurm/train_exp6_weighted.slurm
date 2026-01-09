#!/bin/bash
#SBATCH --job-name=lilygpt-exp6-weighted
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/exp6-weighted-%j.out
#SBATCH --error=logs/exp6-weighted-%j.err
#SBATCH --qos=allgroups_qos

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

# Activate venv (DO NOT load python modules)
source "$HOME/lilygpt/.venv/bin/activate"

echo "================================================================================"
echo "EXPERIMENT 6: WEIGHTED LOSS ON STRUCTURAL TOKENS"
echo "================================================================================"
echo "This experiment tests if weighted loss solves the closing brace problem:"
echo "  - Same dataset as exp5 (full assignment, 100% structurally complete)"
echo "  - Same hyperparameters as exp5 for fair comparison"
echo "  - ONLY CHANGE: Closing brace } has 20x higher loss weight"
echo "  - Final newline before } has 5x higher loss weight"
echo ""
echo "Hypothesis being tested:"
echo "  If model is penalized heavily for missing }, it will learn to generate it"
echo ""
echo "Comparison with exp5:"
echo "  exp5: All tokens have equal weight in loss"
echo "  exp6: Structural tokens (}) have 20x weight"
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
echo "Total: 1503 full assignments (same as exp5)"
echo ""

python -m lilynorm.stages.training.train_weighted \
    --train "data/splits_full/train.jsonl" \
    --val "data/splits_full/val.jsonl" \
    --output-dir "runs/lora_exp6_weighted" \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --learning-rate 5e-5 \
    --epochs 3 \
    --max-length 2048 \
    --lora-r 8 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --structural-weight 20.0 \
    --bf16 \
    --save-steps 250 \
    --eval-steps 250 \
    --logging-steps 10 \
    --warmup-steps 100

echo ""
echo "End time: $(date)"
echo "=== Job Complete ==="
