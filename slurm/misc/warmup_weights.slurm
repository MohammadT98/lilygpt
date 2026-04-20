#!/bin/bash
#SBATCH --job-name=lilybench-warmup
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/warmup-%x-%j.out
#SBATCH --error=logs/warmup-%x-%j.err

# Pre-populate the shared HuggingFace cache with the weights needed by the
# experiment. One job per HF ID. Purely CPU — we just call from_pretrained
# with device_map='cpu' so the weights materialize on disk.
#
# Required env vars:
#   HF_ID          full HF repo id (e.g. microsoft/phi-4, csc-unipd/lilybert)
#   MODE           llm | embedder     (embedder uses AutoModel; llm uses AutoModelForCausalLM)
# Optional:
#   TRUST_REMOTE_CODE   0 | 1  default 0
#
# Usage:
#   HF_ID=microsoft/phi-4 MODE=llm sbatch --job-name=warmup-phi4 slurm/misc/warmup_weights.slurm

set -euo pipefail

: "${HF_ID:?HF_ID is required}"
: "${MODE:?MODE is required (llm|embedder)}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

SCRATCH_ROOT="${TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_ROOT/$USER-$SLURM_JOB_ID"
mkdir -p "$SCRATCH"

export TMPDIR="$SCRATCH/tmp"
export TORCH_HOME="$SCRATCH/torch"
export TRITON_CACHE_DIR="$SCRATCH/triton"
mkdir -p "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR"

# Persistent shared HF cache for the whole sweep.
export HF_HOME="/nfsd/voce/machine_learning/experiments/lilybench/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

if [ -f "$SLURM_SUBMIT_DIR/HF_API" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$SLURM_SUBMIT_DIR/HF_API")"
fi

echo "[slurm] HF_ID=$HF_ID MODE=$MODE TRUST_REMOTE_CODE=$TRUST_REMOTE_CODE"
echo "[slurm] HF_HOME=$HF_HOME"

python - <<PY
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
hf_id = os.environ["HF_ID"]
mode = os.environ["MODE"]
trust = os.environ.get("TRUST_REMOTE_CODE", "0") == "1"
tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust)
if mode == "llm":
    mdl = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype="auto", low_cpu_mem_usage=True, trust_remote_code=trust)
elif mode == "embedder":
    mdl = AutoModel.from_pretrained(hf_id, trust_remote_code=trust)
else:
    raise SystemExit(f"unknown MODE={mode}")
print(f"[warmup] cached {hf_id} ({mode})")
PY
