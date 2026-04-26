#!/bin/bash
#SBATCH --job-name=lilybench-js-ref
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/js-ref-%x-%j.out
#SBATCH --error=logs/js-ref-%x-%j.err

# Build a JS Divergence Similarity reference aggregate by rendering a corpus
# of LilyPond files to MIDI and aggregating muspy metrics.
#
# Required env vars:
#   SOURCE          test.jsonl path (kind=test) or mutopia manifest json (kind=mutopia)
#   KIND            test | mutopia
#   OUT             output JSON cache path
# Optional:
#   WORKDIR         scratch dir for .ly + .mid files (default: $TMPDIR/js_ref)
#   LIMIT           cap on number of files (default: all)

set -euo pipefail

: "${SOURCE:?SOURCE is required}"
: "${KIND:?KIND is required (test|mutopia)}"
: "${OUT:?OUT is required}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

export LILYPOND_BIN="${LILYPOND_BIN:-/home/spanio/lilypond-2.24.4/bin/lilypond}"
export PATH="$(dirname "$LILYPOND_BIN"):$PATH"

SCRATCH_ROOT="${TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_ROOT/$USER-$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$TMPDIR"

WORKDIR="${WORKDIR:-$SCRATCH/js_ref}"
mkdir -p "$WORKDIR"

EXTRA=()
if [ -n "${LIMIT:-}" ]; then
  EXTRA+=("--limit" "$LIMIT")
fi

python scripts/build_js_reference.py \
  --source "$SOURCE" \
  --kind "$KIND" \
  --out "$OUT" \
  --workdir "$WORKDIR" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  "${EXTRA[@]}"
