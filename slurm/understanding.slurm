#!/bin/bash
#SBATCH --job-name=lilybench-und
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/und-%x-%j.out
#SBATCH --error=logs/und-%x-%j.err

# Run the understanding benchmark for one model.
#
# Required env vars:
#   MODEL_ID    phi4 | qwen-coder | deepseek-coder | codestral
#   BENCH       path to the bench JSONL
#   OUTPUT_DIR  predictions root (predictions go under $OUTPUT_DIR/$MODEL_ID/)
# Optional:
#   TASKS         "all" or comma-separated names
#   QUANTIZATION  int8 | int4

set -euo pipefail
: "${MODEL_ID:?MODEL_ID is required}"
: "${BENCH:?BENCH is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

EXTRA_ARGS=()
[[ -n "${TASKS:-}" ]]        && EXTRA_ARGS+=(--tasks "$TASKS")
[[ -n "${QUANTIZATION:-}" ]] && EXTRA_ARGS+=(--quantization "$QUANTIZATION")

lilybench understanding run \
    --model "$MODEL_ID" \
    --bench "$BENCH" \
    --out   "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
