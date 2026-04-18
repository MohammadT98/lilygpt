#!/bin/bash
#SBATCH --job-name=lilybench-exp12-voice-tokens-eff4
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/tmp/exp12-voice-tokens-eff4-%j.out
#SBATCH --error=/tmp/exp12-voice-tokens-eff4-%j.err
#SBATCH --qos=allgroups_qos

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

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

export WANDB_DIR="$SCRATCH/wandb"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
         "$WANDB_DIR"

RUN_NAME="lora_exp12_voice_tokens_eff4"
RUN_SCRATCH_DIR="$SCRATCH/runs/$RUN_NAME"
RUN_HOME_DIR="$SLURM_SUBMIT_DIR/runs/$RUN_NAME"
mkdir -p "$RUN_SCRATCH_DIR"

source "$HOME/lilybench/.venv/bin/activate"

echo "=== Storage Info ==="
echo "SCRATCH: $SCRATCH"
df -h "$SCRATCH_ROOT" || true
df -h /tmp || true
df -h "$HOME" || true
echo ""

sync_back() {
    echo "=== Syncing results back to home ==="
    mkdir -p "$RUN_HOME_DIR"
    rsync -av --partial "$RUN_SCRATCH_DIR/" "$RUN_HOME_DIR/" || true
    # Copy logs from /tmp back to home
    cp "/tmp/exp12-voice-tokens-eff4-$SLURM_JOB_ID.out" "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true
    cp "/tmp/exp12-voice-tokens-eff4-$SLURM_JOB_ID.err" "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true
}
cleanup() {
    sync_back
    rm -rf "$SCRATCH" || true
    rm -f "/tmp/exp12-voice-tokens-eff4-$SLURM_JOB_ID.out" "/tmp/exp12-voice-tokens-eff4-$SLURM_JOB_ID.err" || true
}
trap cleanup EXIT

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "Start time: $(date)"
echo ""

python -m lilybench.stages.training.train_lora \
    --train "data/splits_full/train.jsonl" \
    --val "data/splits_full/val.jsonl" \
    --output-dir "$RUN_SCRATCH_DIR" \
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

echo "End time: $(date)"