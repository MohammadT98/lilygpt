#!/bin/bash
#SBATCH --job-name=lilybench-train
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train-%x-%j.out
#SBATCH --error=logs/train-%x-%j.err

# Parametrized multi-model LoRA training.
# Required env vars (export before sbatch or use --export):
#   MODEL_ID       registry id (gpt-oss | phi4 | qwen-coder | deepseek-coder | codestral)
#   TRAIN_JSONL    path to train.jsonl
#   VAL_JSONL      path to val.jsonl
#   OUTPUT_DIR     output directory for the run (adapters saved to $OUTPUT_DIR/final)
# Optional:
#   EFF_BATCH      effective batch size (default: 16) via gradient accumulation
#   EPOCHS         default 3
#   LR             default 5e-5
#   LORA_R         default 8
#   LORA_ALPHA     default 32
#   MAX_LENGTH     default 2048
#
# Usage example:
#   MODEL_ID=phi4 \
#   TRAIN_JSONL=data/splits_full/train.jsonl \
#   VAL_JSONL=data/splits_full/val.jsonl \
#   OUTPUT_DIR=runs/phi4_lora \
#   sbatch --job-name=phi4-lora slurm/train/train_multimodel.slurm

set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required (e.g. gpt-oss, phi4, qwen-coder, deepseek-coder, codestral)}"
: "${TRAIN_JSONL:?TRAIN_JSONL is required}"
: "${VAL_JSONL:?VAL_JSONL is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

EFF_BATCH="${EFF_BATCH:-16}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
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

export HF_HOME="$SCRATCH/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

mkdir -p "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

echo "[slurm] MODEL_ID=$MODEL_ID EFF_BATCH=$EFF_BATCH EPOCHS=$EPOCHS LR=$LR"
echo "[slurm] OUTPUT_DIR=$OUTPUT_DIR"

python -m lilybench.train \
  "model=$MODEL_ID" \
  "data.train=$TRAIN_JSONL" \
  "data.val=$VAL_JSONL" \
  "output_dir=$OUTPUT_DIR" \
  "batch_size=1" \
  "gradient_accumulation_steps=$EFF_BATCH" \
  "learning_rate=$LR" \
  "epochs=$EPOCHS" \
  "lora.r=$LORA_R" \
  "lora.alpha=$LORA_ALPHA" \
  "max_length=$MAX_LENGTH" \
  "bf16=true"
