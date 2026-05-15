#!/bin/bash
#SBATCH --job-name=lilybench-prep-emopia
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prep-emopia-%j.out
#SBATCH --error=logs/prep-emopia-%j.err

# Download EMOPIA, run midi2ly on every clip in parallel, emit the manifest.
#
# Optional env vars:
#   EMOPIA_URL   default https://zenodo.org/records/5257995/files/EMOPIA_2.2.zip
#   WORKDIR      default /nfsd/voce/machine_learning/datasets/emopia/_workdir
#   OUT_ROOT     default /nfsd/voce/machine_learning/datasets/emopia
#   MIDI2LY      default /home/spanio/lilypond-2.24.4/bin/midi2ly
#   JOBS         default 16
#   LIMIT        cap MIDI count (for smoke runs); empty = process all

set -euo pipefail

export EMOPIA_URL="${EMOPIA_URL:-https://zenodo.org/records/5257995/files/EMOPIA_2.2.zip}"
export WORKDIR="${WORKDIR:-/nfsd/voce/machine_learning/datasets/emopia/_workdir}"
export OUT_ROOT="${OUT_ROOT:-/nfsd/voce/machine_learning/datasets/emopia}"
export MIDI2LY="${MIDI2LY:-/home/spanio/lilypond-2.24.4/bin/midi2ly}"
export JOBS="${JOBS:-16}"
export LIMIT="${LIMIT:-}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# midi2ly needs python on PATH (it's a Python script that imports lilypond
# internals); the LilyPond install bundles its own Python.
export PATH="$(dirname "$MIDI2LY"):$PATH"

ARGS=(
  "--zip-url=$EMOPIA_URL"
  "--workdir=$WORKDIR"
  "--out-root=$OUT_ROOT"
  "--midi2ly=$MIDI2LY"
  "--jobs=$JOBS"
)
if [ -n "$LIMIT" ]; then
  ARGS+=("--limit=$LIMIT")
fi

python scripts/prepare_emopia.py "${ARGS[@]}"
