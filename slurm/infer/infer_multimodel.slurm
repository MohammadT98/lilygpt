#!/bin/bash
#SBATCH --job-name=lilybench-infer
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/infer-%x-%j.out
#SBATCH --error=logs/infer-%x-%j.err

# Parametrized multi-model, multi-regime inference.
# Required env vars:
#   MODEL_ID       registry id (gpt-oss | phi4 | qwen-coder | deepseek-coder | codestral)
#   REGIME         zero | few | lora
# Required for REGIME=lora:
#   LORA_PATH      path to LoRA adapter folder (e.g. runs/.../final)
# Required for REGIME=few:
#   FEWSHOT_FILE   text file with three curated demonstrations separated by a blank line
# Optional:
#   PROMPT_FILE    path to a prompt text file (user instruction). If absent, a default
#                  monophonic prompt is used.
#   NUM_SAMPLES    default 100
#   TEMPERATURE    default 0.7
#   TOP_P          default 0.9
#   MAX_NEW_TOKENS default 3000
#
# Example:
#   MODEL_ID=qwen-coder REGIME=zero NUM_SAMPLES=50 \
#     sbatch --job-name=qwen-zero slurm/infer/infer_multimodel.slurm

set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required}"
: "${REGIME:?REGIME is required (zero|few|lora)}"

export MODEL_ID REGIME
export LORA_PATH="${LORA_PATH:-}"
export FEWSHOT_FILE="${FEWSHOT_FILE:-}"
export PROMPT_FILE="${PROMPT_FILE:-}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.9}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3000}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$HOME/lilybench/.venv/bin/activate"

SCRATCH_ROOT="${TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_ROOT/$USER-$SLURM_JOB_ID"
mkdir -p "$SCRATCH"

export TMPDIR="$SCRATCH/tmp"
export TORCH_HOME="$SCRATCH/torch"
export TRITON_CACHE_DIR="$SCRATCH/triton"

export HF_HOME="$SCRATCH/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

mkdir -p "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

OUTPUT_DIR="${OUTPUT_DIR:-data/inference/outputs/${MODEL_ID}_${REGIME}}"

INFER_OVERRIDES=(
  "model=$MODEL_ID"
  "regime=$REGIME"
  "num_samples=$NUM_SAMPLES"
  "temperature=$TEMPERATURE"
  "top_p=$TOP_P"
  "max_new_tokens=$MAX_NEW_TOKENS"
  "output_dir=$OUTPUT_DIR"
)

if [ "$REGIME" = "lora" ]; then
  : "${LORA_PATH:?REGIME=lora requires LORA_PATH}"
  INFER_OVERRIDES+=("regime.path=$LORA_PATH")
fi
if [ "$REGIME" = "few" ]; then
  : "${FEWSHOT_FILE:?REGIME=few requires FEWSHOT_FILE}"
  INFER_OVERRIDES+=("regime.fewshot_file=$FEWSHOT_FILE")
fi
if [ -n "$PROMPT_FILE" ]; then
  INFER_OVERRIDES+=("prompt_file=$PROMPT_FILE")
fi

python -m lilybench.infer "${INFER_OVERRIDES[@]}"
