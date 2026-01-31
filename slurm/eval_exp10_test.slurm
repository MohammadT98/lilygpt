#!/bin/bash
#SBATCH --job-name=lilygpt-eval-test
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval-test-%j.out
#SBATCH --error=logs/eval-test-%j.err
#SBATCH --qos=allgroups_qos

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

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

export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

source "$HOME/lilygpt/.venv/bin/activate"

# Usage:
# sbatch slurm/eval_exp_test.slurm --export=LORA_PATH=runs/lora_exp10_voice_tokens/final
# Defaults to exp10 if LORA_PATH is not provided.
LORA_PATH="${LORA_PATH:-runs/lora_exp10_voice_tokens/final}"
DATA_PATH="${DATA_PATH:-data/splits_full/test.jsonl}"

python -m lilynorm.stages.training.eval_lora \
    --data "$DATA_PATH" \
    --lora-path "$LORA_PATH" \
    --model-name "openai/gpt-oss-20b" \
    --batch-size 1 \
    --max-length 2048 \
    --bf16

rm -rf "$SCRATCH" || true
