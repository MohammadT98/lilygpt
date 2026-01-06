#!/bin/bash
#SBATCH --job-name=lilygpt-exp5-full
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/exp5-full-%j.out
#SBATCH --error=logs/exp5-full-%j.err
#SBATCH --qos=allgroups_qos

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

# Activate venv (DO NOT load python modules)
source "$HOME/lilygpt/.venv/bin/activate"

echo "================================================================================"
echo "EXPERIMENT 5: FULL ASSIGNMENT TRAINING (No Continuation Splits)"
echo "================================================================================"
echo "This experiment tests if continuation splitting is the root cause:"
echo "  - Train on COMPLETE assignments (1 example per piece)"
echo "  - No artificial start/middle/near-end fragmentation"
echo "  - Every example shows proper structure with closure"
echo "  - Same hyperparameters as exp4 for fair comparison"
echo ""
echo "Dataset differences:"
echo "  exp4: 3 examples per piece (start/middle/near-end) with input/output splits"
echo "  exp5: 1 complete example per piece (full assignment)"
echo ""
echo "Hypothesis being tested:"
echo "  Fragmented training (66% incomplete closures) prevents structure learning"
echo "  Full assignments should teach the model proper syntax boundaries"
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
echo "Total: 1503 full assignments (vs 8852 continuation fragments in exp4)"
echo ""

python -m lilynorm.stages.training.train_standard \
    --train "data/splits_full/train.jsonl" \
    --val "data/splits_full/val.jsonl" \
    --output-dir "runs/lora_exp5_full" \
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
