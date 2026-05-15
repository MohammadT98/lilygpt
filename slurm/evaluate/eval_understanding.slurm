#!/bin/bash
#SBATCH --job-name=lilybench-eval-und
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval-und-%x-%j.out
#SBATCH --error=logs/eval-und-%x-%j.err

# Music-understanding scoring. CPU-only — reads predictions JSONL + bench
# JSONL, emits one summary.json per model.
#
# Required env vars:
#   MODEL_ID            registry id (phi4 | qwen-coder | deepseek-coder | codestral)
# Optional:
#   PREDICTIONS_DIR     default data/understanding/predictions/${MODEL_ID}
#   BENCH_PATH          default data/understanding/bench.jsonl
#   OUT                 default data/understanding/eval/${MODEL_ID}/summary.json

set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required}"

export MODEL_ID
export PREDICTIONS_DIR="${PREDICTIONS_DIR:-data/understanding/predictions/${MODEL_ID}}"
export BENCH_PATH="${BENCH_PATH:-data/understanding/bench.jsonl}"
export OUT="${OUT:-data/understanding/eval/${MODEL_ID}/summary.json}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

EVAL_OVERRIDES=(
  "model_id=$MODEL_ID"
  "predictions_dir=$PREDICTIONS_DIR"
  "bench_path=$BENCH_PATH"
  "out=$OUT"
)

python -m lilybench.evaluate.understanding "${EVAL_OVERRIDES[@]}"
