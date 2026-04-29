#!/bin/bash
#SBATCH --job-name=rescue-and-eval-lora
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rescue-%x-%j.out
#SBATCH --error=logs/rescue-%x-%j.err

# Run scripts/rescue_lora_outputs.py on a finished LoRA inference dir, then
# sbatch text_midi + 2 CPU FMD jobs against the rescued samples.
#
# Required env:
#   KEY     e.g. phi4_lora — selects inference/$KEY → inference/${KEY}_rescued_v2
# Optional:
#   MIN_BARS         default 8
#   RATIO_THRESHOLD  default 0.5

set -euo pipefail
: "${KEY:?KEY is required (e.g. phi4_lora)}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

export LILYPOND_BIN="${LILYPOND_BIN:-/home/spanio/lilypond-2.24.4/bin/lilypond}"

INFER_ROOT=/nfsd/voce/machine_learning/experiments/lilybench/inference
EVAL_ROOT=/nfsd/voce/machine_learning/experiments/lilybench/eval
JS_TEST=/nfsd/voce/machine_learning/experiments/lilybench/js_refs/bmdataset_muspy_agg.json
JS_MUTOPIA=/nfsd/voce/machine_learning/experiments/lilybench/js_refs/mutopia_muspy_agg.json
TEST_JSONL=/nfsd/voce/machine_learning/experiments/lilybench/data/splits_full/test.jsonl
MUTOPIA_NEW=/nfsd/voce/machine_learning/experiments/lilybench/data/mutopia/dataset_mutopia_converted.json
FMD_TEST_REF=/nfsd/voce/machine_learning/experiments/lilybench/fmd_refs/test_lilybert_L6.npz
FMD_MUTOPIA_REF=/nfsd/voce/machine_learning/experiments/lilybench/fmd_refs/mutopia_lilybert_L6.npz
LILYBERT=csc-unipd/lilybert

RAW="$INFER_ROOT/$KEY/samples"
RESCUED_KEY="${KEY}_rescued_v2"
RESCUED_DIR="$INFER_ROOT/$RESCUED_KEY/samples"
EOUT="$EVAL_ROOT/$RESCUED_KEY"
mkdir -p "$RESCUED_DIR" "$EOUT"

echo "[rescue-and-eval] rescuing $RAW -> $RESCUED_DIR"
python scripts/rescue_lora_outputs.py \
  --input-dir "$RAW" \
  --output-dir "$RESCUED_DIR" \
  --min-bars "${MIN_BARS:-8}" \
  --ratio-threshold "${RATIO_THRESHOLD:-0.5}" \
  --stats-out "$EOUT/rescue_stats.json"

# Bail out if no samples survived the rescue.
N_SURVIVING=$(ls "$RESCUED_DIR"/*.ly 2>/dev/null | wc -l)
echo "[rescue-and-eval] surviving samples: $N_SURVIVING"
if [ "$N_SURVIVING" -lt 2 ]; then
  echo "[rescue-and-eval] not enough samples; skipping eval"
  exit 0
fi

echo "[rescue-and-eval] queuing eval chain for $RESCUED_KEY"
INPUT_DIR="$RESCUED_DIR" EVAL_OUT_DIR="$EOUT" \
  REFERENCE_AGGREGATE_PATH_TEST="$JS_TEST" \
  REFERENCE_AGGREGATE_PATH_MUTOPIA="$JS_MUTOPIA" \
  sbatch --job-name="eval-tm-$RESCUED_KEY" slurm/evaluate/eval_text_midi.slurm

GENERATIONS_DIR="$RESCUED_DIR" REFERENCE_KIND=test REFERENCE_PATH="$TEST_JSONL" \
  EMBEDDER_CHECKPOINT="$LILYBERT" FMD_OUT="$EOUT/fmd_test.json" \
  REFERENCE_EMBEDDINGS_PATH="$FMD_TEST_REF" \
  sbatch --job-name="eval-fmd-test-$RESCUED_KEY-cpu" slurm/evaluate/eval_fmd_cpu.slurm

GENERATIONS_DIR="$RESCUED_DIR" REFERENCE_KIND=mutopia REFERENCE_PATH="$MUTOPIA_NEW" \
  EMBEDDER_CHECKPOINT="$LILYBERT" FMD_OUT="$EOUT/fmd_mutopia.json" \
  REFERENCE_EMBEDDINGS_PATH="$FMD_MUTOPIA_REF" \
  sbatch --job-name="eval-fmd-mut-$RESCUED_KEY-cpu" slurm/evaluate/eval_fmd_cpu.slurm

echo "[rescue-and-eval] done"
