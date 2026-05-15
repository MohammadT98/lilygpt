#!/bin/bash
#SBATCH --job-name=lilybench-infer-und
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/infer-und-%x-%j.out
#SBATCH --error=logs/infer-und-%x-%j.err

# Zero-shot music-understanding inference. One job per model; emits one
# predictions JSONL per task.
#
# Required env vars:
#   MODEL_ID       registry id (phi4 | qwen-coder | deepseek-coder | codestral)
# Optional:
#   BENCH_PATH     default data/understanding/bench.jsonl
#   OUTPUT_DIR     default data/understanding/predictions
#   TASKS          "all" (default) or comma list e.g. "bar_count,composer_recognition"
#   LIMIT          cap per-task records (smoke runs)
#   MAX_NEW_TOKENS default 20
#
# Example:
#   MODEL_ID=phi4 \
#     sbatch --job-name=infer-und-phi4 slurm/infer/infer_understanding.slurm

set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required}"

export MODEL_ID
export BENCH_PATH="${BENCH_PATH:-data/understanding/bench.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-data/understanding/predictions}"
export TASKS="${TASKS:-all}"
export LIMIT="${LIMIT:-}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

SCRATCH_ROOT="${TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_ROOT/$USER-$SLURM_JOB_ID"
mkdir -p "$SCRATCH"

export TMPDIR="$SCRATCH/tmp"
export TORCH_HOME="$SCRATCH/torch"
export TRITON_CACHE_DIR="$SCRATCH/triton"

export HF_HOME="/nfsd/voce/machine_learning/experiments/lilybench/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

mkdir -p "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

if [ -f "$SLURM_SUBMIT_DIR/HF_API" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$SLURM_SUBMIT_DIR/HF_API")"
fi

INFER_OVERRIDES=(
  "model=$MODEL_ID"
  "bench_path=$BENCH_PATH"
  "output_dir=$OUTPUT_DIR"
  "tasks=$TASKS"
  "max_new_tokens=$MAX_NEW_TOKENS"
)

if [ -n "$LIMIT" ]; then
  INFER_OVERRIDES+=("limit=$LIMIT")
fi

python -m lilybench.infer_understanding "${INFER_OVERRIDES[@]}"
