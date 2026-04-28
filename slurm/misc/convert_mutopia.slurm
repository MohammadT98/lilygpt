#!/bin/bash
#SBATCH --job-name=lilybench-convert-mutopia
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/convert-mutopia-%x-%j.out
#SBATCH --error=logs/convert-mutopia-%x-%j.err

# Run convert-ly on the entire Mutopia corpus to upgrade old LilyPond syntax
# (e.g., \include "deutsch.ly" → \language "deutsch") so more files compile to
# MIDI under LilyPond 2.24.4. Writes a parallel tree + a new manifest.
#
# Required env vars:
#   SOURCE          source dataset_mutopia.json
#   OUT_DIR         root dir for upgraded .ly files
#   MANIFEST_OUT    path for the new manifest (with convert_ly_path field)
# Optional:
#   TO_VERSION      target LilyPond version (default 2.24.4)

set -euo pipefail

: "${SOURCE:?SOURCE is required}"
: "${OUT_DIR:?OUT_DIR is required}"
: "${MANIFEST_OUT:?MANIFEST_OUT is required}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

export LILYPOND_BIN="${LILYPOND_BIN:-/home/spanio/lilypond-2.24.4/bin/lilypond}"
CONVERT_LY="$(dirname "$LILYPOND_BIN")/convert-ly"

SCRATCH_ROOT="${TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_ROOT/$USER-$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$TMPDIR"

mkdir -p "$OUT_DIR" "$(dirname "$MANIFEST_OUT")"

python scripts/convert_mutopia.py \
  --source "$SOURCE" \
  --out-dir "$OUT_DIR" \
  --manifest-out "$MANIFEST_OUT" \
  --convert-ly "$CONVERT_LY" \
  --to "${TO_VERSION:-2.24.4}" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
