#!/bin/bash
#SBATCH --job-name=lilybench-eval-fmd
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval-fmd-%x-%j.out
#SBATCH --error=logs/eval-fmd-%x-%j.err

# Thin wrapper around python -m lilybench.evaluate.fmd.
# Required env vars:
#   GENERATIONS_DIR        directory of sample_*.ly from a single infer run
#   REFERENCE_KIND         test | mutopia
#   REFERENCE_PATH         test.jsonl (for kind=test) or mutopia manifest json (kind=mutopia)
#   EMBEDDER_CHECKPOINT    e.g. CSCPadova/lilybert
#   FMD_OUT                output json path (e.g. .../fmd_test.json)
# Optional:
#   REFERENCE_EMBEDDINGS_PATH   precomputed .npz of reference embeddings (skip re-embed)
#   EMBED_LAYER                  default 6
#   BATCH_SIZE                   default 16
#   MAX_LENGTH                   default 512

set -euo pipefail

: "${GENERATIONS_DIR:?GENERATIONS_DIR is required}"
: "${REFERENCE_KIND:?REFERENCE_KIND is required (test|mutopia)}"
: "${REFERENCE_PATH:?REFERENCE_PATH is required}"
: "${EMBEDDER_CHECKPOINT:?EMBEDDER_CHECKPOINT is required}"
: "${FMD_OUT:?FMD_OUT is required}"

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

FMD_OVERRIDES=(
  "generations_dir=$GENERATIONS_DIR"
  "reference_kind=$REFERENCE_KIND"
  "reference_path=$REFERENCE_PATH"
  "embedder_checkpoint=$EMBEDDER_CHECKPOINT"
  "out=$FMD_OUT"
  "embed_layer=${EMBED_LAYER:-6}"
  "batch_size=${BATCH_SIZE:-16}"
  "max_length=${MAX_LENGTH:-512}"
)

if [ -n "${REFERENCE_EMBEDDINGS_PATH:-}" ]; then
  FMD_OVERRIDES+=("reference_embeddings_path=$REFERENCE_EMBEDDINGS_PATH")
fi

python -m lilybench.evaluate.fmd "${FMD_OVERRIDES[@]}"
