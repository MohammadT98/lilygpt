#!/bin/bash
#SBATCH --job-name=lilybench-eval-text-midi
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-text-midi-%x-%j.out
#SBATCH --error=logs/eval-text-midi-%x-%j.err

# Thin wrapper around python -m lilybench.evaluate.text_midi.
# Required env vars:
#   INPUT_DIR        directory of sample_*.ly files produced by infer
#   EVAL_OUT_DIR     directory for eval.jsonl / summary.json / midi/
# Optional:
#   EXPECTED_NOTATION         default "relative"
#   REQUIRE_LOWERCASE         default true
#   ALLOW_ACCIDENTALS         default true
#   FORCE_MIDI                default false
#   REFERENCE_MIDI_DIR        dir of reference MIDIs (for JS-divergence similarity)
#   REFERENCE_AGGREGATE_PATH  JSON cache of reference aggregate stats (load-or-compute)
#
# LilyPond must be on PATH (or LILYPOND_BIN set) — text_midi shells out to it.

set -euo pipefail

: "${INPUT_DIR:?INPUT_DIR is required}"
: "${EVAL_OUT_DIR:?EVAL_OUT_DIR is required}"

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

mkdir -p "$EVAL_OUT_DIR/midi"

EXTRA_OVERRIDES=()
if [ -n "${REFERENCE_MIDI_DIR:-}" ]; then
  EXTRA_OVERRIDES+=("reference_midi_dir=$REFERENCE_MIDI_DIR")
fi
if [ -n "${REFERENCE_AGGREGATE_PATH:-}" ]; then
  EXTRA_OVERRIDES+=("reference_aggregate_path=$REFERENCE_AGGREGATE_PATH")
fi
# Multi-reference JS-sim: pass each label and path via REFERENCE_AGGREGATE_PATH_<LABEL>.
# Example: REFERENCE_AGGREGATE_PATH_TEST=/.../test.json REFERENCE_AGGREGATE_PATH_MUTOPIA=/.../mut.json
PAIRS=""
for VAR in $(env | awk -F= '/^REFERENCE_AGGREGATE_PATH_/ {print $1}'); do
  LABEL=$(echo "${VAR#REFERENCE_AGGREGATE_PATH_}" | tr 'A-Z' 'a-z')
  VAL="${!VAR}"
  [ -z "$VAL" ] && continue
  PAIRS="$PAIRS$LABEL: $VAL, "
done
if [ -n "$PAIRS" ]; then
  EXTRA_OVERRIDES+=("reference_aggregate_paths={${PAIRS%, }}")
fi

python -m lilybench.evaluate.text_midi \
  "input_dir=$INPUT_DIR" \
  "out=$EVAL_OUT_DIR/eval.jsonl" \
  "summary=$EVAL_OUT_DIR/summary.json" \
  "midi_dir=$EVAL_OUT_DIR/midi" \
  "output_dir=$EVAL_OUT_DIR" \
  "expected_notation=${EXPECTED_NOTATION:-relative}" \
  "require_lowercase=${REQUIRE_LOWERCASE:-true}" \
  "allow_accidentals=${ALLOW_ACCIDENTALS:-true}" \
  "force_midi=${FORCE_MIDI:-false}" \
  "${EXTRA_OVERRIDES[@]}"
