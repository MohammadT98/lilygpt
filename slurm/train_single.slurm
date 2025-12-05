#!/bin/bash
#SBATCH --job-name=lilygpt-lora
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Adjust these based on your cluster configuration:
# --gres=gpu:a100:1  # for specific GPU type
# --partition=YOUR_PARTITION_NAME
# --account=YOUR_ACCOUNT

module purge
module load python/3.10  # Adjust to your cluster's Python module

# Activate virtual environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

# Or use uv if available on the cluster
# export PATH="$HOME/.cargo/bin:$PATH"

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR || exit 1
mkdir -p logs

# Run training
python -m lilynorm.stages.training.train \
    --train "data/splits/train.jsonl" \
    --val "data/splits/val.jsonl" \
    --output-dir "data/fine_tuning/checkpoints" \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --epochs 3 \
    --max-length 1024 \
    --lora-r 8 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --bf16 \
    --save-steps 500 \
    --eval-steps 500 \
    --logging-steps 10 \
    --warmup-steps 100

echo ""
echo "End time: $(date)"
echo "=== Job Complete ==="
