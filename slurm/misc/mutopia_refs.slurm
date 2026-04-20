#!/bin/bash
#SBATCH --job-name=lilybench-mutopia-refs
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mutopia-refs-%x-%j.out
#SBATCH --error=logs/mutopia-refs-%x-%j.err

# Pre-compute FMD reference embeddings with LilyBERT (layer 6) for both
# reference sets used across the eval sweep:
#   * in-domain:  data/splits_full/test.jsonl full_text
#   * out-of-domain: /nfsd/voce/machine_learning/datasets/mutopia + its manifest
# Saves embeddings under /nfsd/.../lilybench/fmd_refs/ so every later eval_fmd
# job can skip re-embedding.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$HOME/lilybench/.venv/bin/activate"

SCRATCH_ROOT="${TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_ROOT/$USER-$SLURM_JOB_ID"
mkdir -p "$SCRATCH"

export TMPDIR="$SCRATCH/tmp"
export TORCH_HOME="$SCRATCH/torch"
export TRITON_CACHE_DIR="$SCRATCH/triton"
mkdir -p "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR"

export HF_HOME="/nfsd/voce/machine_learning/experiments/lilybench/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

if [ -f "$SLURM_SUBMIT_DIR/HF_API" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$SLURM_SUBMIT_DIR/HF_API")"
fi

export EMBEDDER="${EMBEDDER:-CSCPadova/lilybert}"
export EMBED_LAYER="${EMBED_LAYER:-6}"
export REFS_DIR="/nfsd/voce/machine_learning/experiments/lilybench/fmd_refs"
export TEST_JSONL="${TEST_JSONL:-/nfsd/voce/machine_learning/experiments/lilybench/data/splits_full/test.jsonl}"
export MUTOPIA_MANIFEST="${MUTOPIA_MANIFEST:-/nfsd/voce/machine_learning/datasets/mutopia/dataset_mutopia.json}"

mkdir -p "$REFS_DIR"

# fmd.py embeds both generations and reference in one shot; here we only need
# reference-side embeddings, so we call the internal helpers directly.
python - <<PY
import json, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from lilybench.evaluate.fmd import _load_reference, _embed_corpus

embedder = os.environ.get("EMBEDDER", "CSCPadova/lilybert")
layer = int(os.environ.get("EMBED_LAYER", "6"))
refs_dir = Path(os.environ["REFS_DIR"])
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[refs] loading embedder {embedder} on {device}")
tok = AutoTokenizer.from_pretrained(embedder, use_fast=True)
mdl = AutoModel.from_pretrained(embedder).to(device).eval()

for kind, path_env, out_name in [
    ("test", "TEST_JSONL", "test_lilybert_L6.npz"),
    ("mutopia", "MUTOPIA_MANIFEST", "mutopia_lilybert_L6.npz"),
]:
    src = Path(os.environ[path_env])
    out = refs_dir / out_name
    if not src.exists():
        print(f"[refs] skipping {kind}: {src} not found")
        continue
    docs = _load_reference(kind, src, min_chars=40)
    print(f"[refs] {kind}: {len(docs)} docs from {src}")
    if len(docs) < 2:
        print(f"[refs] WARNING {kind} has <2 docs; skipping")
        continue
    y = _embed_corpus(docs, tok, mdl, device, batch_size=16, max_length=512, embed_layer=layer)
    np.savez(out, embeddings=y, n=np.asarray(len(docs), dtype=np.int64),
             embed_layer=np.asarray(layer, dtype=np.int64))
    print(f"[refs] wrote {out}  shape={y.shape}")
PY

export REFS_DIR TEST_JSONL MUTOPIA_MANIFEST EMBEDDER EMBED_LAYER
