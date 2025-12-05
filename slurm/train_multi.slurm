#!/bin/bash
#SBATCH --job-name=lilygpt-multi
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Multi-GPU training with DeepSpeed or DDP
# Adjust resource requests based on your cluster

module purge
module load python/3.10
module load cuda/11.8  # Adjust CUDA version as needed
module load nccl      # For multi-GPU communication

echo "=== Multi-GPU Training Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "GPUs per node: $(nvidia-smi --list-gpus | wc -l)"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
echo "Start time: $(date)"
echo ""

cd $SLURM_SUBMIT_DIR || exit 1
mkdir -p logs

# Multi-node training with torchrun/accelerate
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))

# Using accelerate for multi-GPU (recommended)
accelerate launch \
    --multi_gpu \
    --num_processes=$WORLD_SIZE \
    --num_machines=$SLURM_NNODES \
    --machine_rank=$SLURM_NODEID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    -m lilynorm.stages.training.train \
    --train "data/splits/train.jsonl" \
    --val "data/splits/val.jsonl" \
    --output-dir "data/fine_tuning/checkpoints_multigpu" \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-4 \
    --epochs 3 \
    --bf16 \
    --save-steps 250 \
    --eval-steps 250

echo ""
echo "End time: $(date)"
echo "=== Multi-GPU Job Complete ==="
