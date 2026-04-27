#!/bin/bash
#SBATCH --job-name=lilybench-eval-fmd-cpu
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval-fmd-cpu-%x-%j.out
#SBATCH --error=logs/eval-fmd-cpu-%x-%j.err

# CPU-only FMD wrapper. Same env-var contract as eval_fmd.slurm but runs the
# embedder on CPU; LilyBERT is 12M params so this finishes in ~5 min and lets
# us bypass L40S queue contention.
#
# Required env vars:
#   GENERATIONS_DIR        directory of sample_*.ly from a single infer run
#   REFERENCE_KIND         test | mutopia
#   REFERENCE_PATH         test.jsonl (for kind=test) or mutopia manifest json
#   EMBEDDER_CHECKPOINT    e.g. csc-unipd/lilybert
#   FMD_OUT                output json path
# Optional:
#   REFERENCE_EMBEDDINGS_PATH   precomputed .npz of reference embeddings
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

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

SCRATCH_ROOT="${TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_ROOT/$USER-$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH/tmp"
export TORCH_HOME="$SCRATCH/torch"
mkdir -p "$TMPDIR" "$TORCH_HOME"

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
  "device=cpu"
)

if [ -n "${REFERENCE_EMBEDDINGS_PATH:-}" ]; then
  FMD_OVERRIDES+=("reference_embeddings_path=$REFERENCE_EMBEDDINGS_PATH")
fi

python -m lilybench.evaluate.fmd "${FMD_OVERRIDES[@]}"
