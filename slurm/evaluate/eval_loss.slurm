#!/bin/bash
#SBATCH --job-name=lilybench-eval-loss
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-loss-%x-%j.out
#SBATCH --error=logs/eval-loss-%x-%j.err

# Thin wrapper around python -m lilybench.evaluate.loss.
# Required env vars:
#   MODEL_ID       registry id
#   LORA_PATH      path to the LoRA adapter folder (runs/.../final)
#   LOSS_OUT       output json path
# Optional:
#   TEST_JSONL     default: data/splits_full/test.jsonl
#   MAX_LENGTH     default 2048

set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required}"
: "${LORA_PATH:?LORA_PATH is required}"
: "${LOSS_OUT:?LOSS_OUT is required}"

TEST_JSONL="${TEST_JSONL:-data/splits_full/test.jsonl}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$HOME/lilybench/.venv/bin/activate"

SCRATCH_ROOT="${TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_ROOT/$USER-$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH/tmp"
export TORCH_HOME="$SCRATCH/torch"
export TRITON_CACHE_DIR="$SCRATCH/triton"
mkdir -p "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR"

export HF_HOME="/nfsd/voce/machine_learning/experiments/lilybench/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

if [ -f "$SLURM_SUBMIT_DIR/HF_API" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$SLURM_SUBMIT_DIR/HF_API")"
fi

python -m lilybench.evaluate.loss \
  "model=$MODEL_ID" \
  "lora_path=$LORA_PATH" \
  "data=$TEST_JSONL" \
  "max_length=$MAX_LENGTH" \
  "out=$LOSS_OUT" \
  "bf16=true"
