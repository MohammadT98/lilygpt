#!/bin/bash
#SBATCH --job-name=lilygpt-multi
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos

set -euo pipefail

# Multi-GPU on a single node. Adjust gres/time/mem if needed.

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

# Activate local venv (already contains torch/peft/accelerate)
source "$HOME/lilygpt/.venv/bin/activate"

echo "=== Multi-GPU Training Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "GPUs per node: $(nvidia-smi --list-gpus 2>/dev/null | wc -l)" || true
echo "Total GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo ""

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_GPUS_ON_NODE

# Ensure accelerate uses the GPUs allocated by SLURM
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

accelerate launch \
    --multi_gpu \
    --num_processes=$WORLD_SIZE \
    --num_machines=1 \
    --mixed_precision=bf16 \
    -m lilynorm.stages.training.train \
    --train "data/splits/train.jsonl" \
    --val "data/splits/val.jsonl" \
    --output-dir "runs/lora_multigpu" \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-4 \
    --epochs 3 \
    --max-length 1024 \
    --lora-r 8 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --save-steps 250 \
    --eval-steps 250 \
    --logging-steps 10 \
    --warmup-steps 100

echo ""
echo "End time: $(date)"
echo "=== Multi-GPU Job Complete ==="
