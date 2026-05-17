#!/bin/bash
#SBATCH --job-name=lilybench-gen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gen-%x-%j.out
#SBATCH --error=logs/gen-%x-%j.err

# Run one (model, regime) generation cell.
#
# Required env vars:
#   MODEL_ID    phi4 | qwen-coder | deepseek-coder | codestral
#   REGIME      zero | few | few_ablation (or any registered regime)
#   PROMPTS     path to the prompt bank JSONL
#   OUTPUT_DIR  output root (sample dir lives at $OUTPUT_DIR/samples)
# Optional:
#   FEWSHOT_FILE   demonstrations file (required for --regime=few)
#   QUANTIZATION   int8 | int4 (bitsandbytes)

set -euo pipefail
: "${MODEL_ID:?MODEL_ID is required}"
: "${REGIME:?REGIME is required}"
: "${PROMPTS:?PROMPTS is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

EXTRA_ARGS=()
[[ -n "${FEWSHOT_FILE:-}" ]]   && EXTRA_ARGS+=(--fewshot-file "$FEWSHOT_FILE")
[[ -n "${QUANTIZATION:-}" ]]   && EXTRA_ARGS+=(--quantization "$QUANTIZATION")

lilybench generation run \
    --model    "$MODEL_ID" \
    --regime   "$REGIME" \
    --prompts  "$PROMPTS" \
    --out      "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
