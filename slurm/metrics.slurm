#!/bin/bash
#SBATCH --job-name=lilybench-metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/metrics-%x-%j.out
#SBATCH --error=logs/metrics-%x-%j.err

# Compile rate + JS-similarity + LilyBERT-FMD on a generation run.
#
# Required env vars:
#   SAMPLES         directory of generated .ly files
#   OUT             output JSON path
# Optional:
#   REFERENCE_TEST           BMdataset test split JSONL (FMD in-domain)
#   REFERENCE_MUTOPIA        Mutopia manifest JSON  (FMD out-of-domain)
#   REFERENCE_TEST_MIDI_DIR  BMdataset test MIDIs   (JS in-domain)
#   REFERENCE_MUTOPIA_MIDI_DIR Mutopia MIDIs        (JS out-of-domain)
#   LILYBERT                LilyBERT checkpoint path
#   LILYBERT_LAYER          default 6 (paper)
#   MIDI_DIR                where to drop compiled MIDIs (default: $OUT/midi)

set -euo pipefail
: "${SAMPLES:?SAMPLES is required}"
: "${OUT:?OUT is required}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

EXTRA_ARGS=()
[[ -n "${REFERENCE_TEST:-}" ]]              && EXTRA_ARGS+=(--reference-test "$REFERENCE_TEST")
[[ -n "${REFERENCE_MUTOPIA:-}" ]]           && EXTRA_ARGS+=(--reference-mutopia "$REFERENCE_MUTOPIA")
[[ -n "${REFERENCE_TEST_MIDI_DIR:-}" ]]     && EXTRA_ARGS+=(--reference-test-midi-dir "$REFERENCE_TEST_MIDI_DIR")
[[ -n "${REFERENCE_MUTOPIA_MIDI_DIR:-}" ]]  && EXTRA_ARGS+=(--reference-mutopia-midi-dir "$REFERENCE_MUTOPIA_MIDI_DIR")
[[ -n "${LILYBERT:-}" ]]                    && EXTRA_ARGS+=(--lilybert "$LILYBERT")
[[ -n "${LILYBERT_LAYER:-}" ]]              && EXTRA_ARGS+=(--lilybert-layer "$LILYBERT_LAYER")
[[ -n "${MIDI_DIR:-}" ]]                    && EXTRA_ARGS+=(--midi-dir "$MIDI_DIR")

lilybench metrics generation \
    --samples "$SAMPLES" \
    --out     "$OUT" \
    "${EXTRA_ARGS[@]}"
