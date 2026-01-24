#!/bin/bash
#SBATCH --job-name=lilygpt-exp12-voice-tokens-eff4
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/exp12-voice-tokens-eff4-%j.out
#SBATCH --error=logs/exp12-voice-tokens-eff4-%j.err
#SBATCH --qos=allgroups_qos

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

# Redirect caches to home directory (avoid /tmp filling up on compute node)
export TMPDIR="$HOME/.cache/tmp"
export TORCH_HOME="$HOME/.cache/torch"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export HF_HOME="$HOME/.cache/huggingface"
mkdir -p "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$HF_HOME"

# Debug flags (optional)
export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1

# Activate venv (DO NOT load python modules)
source "$HOME/lilygpt/.venv/bin/activate"

echo "================================================================================"
echo "EXPERIMENT 12: VOICE LABELS + STRUCTURAL TOKENS (EFFECTIVE BATCH 4)"
echo "================================================================================"
echo "Unweighted training (standard loss). Dataset uses EOS appended per sample."
echo "NO INPUT MASKING - model trains on full prompt + completion context."
echo "Voice labels injected at training time (e.g., <VOICE:IvlIn>)."
echo "Per-GPU batch: 1 | Grad accum: 4 | Effective batch: 4"
echo "================================================================================"
echo ""

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "Start time: $(date)"
echo ""

echo "=== Dataset ==="
echo "Train/Val splits: data/splits_full"
echo ""

python -m lilynorm.stages.training.train_lora \
    --train "data/splits_full/train.jsonl" \
    --val "data/splits_full/val.jsonl" \
    --output-dir "runs/lora_exp12_voice_tokens_eff4" \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
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
