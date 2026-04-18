# lilybench

LilyPond code-generation benchmark: dataset builder, LoRA fine-tuning, and evaluation.

## Overview

`lilybench` turns the `data/bmdataset/preprocessed/` LilyPond corpus into a full-file
training dataset with anti-memorization augmentations and a structured metadata
header, trains LoRA adapters over several code/general LLMs, and evaluates the
generated `.ly` samples with text checks, MIDI analysis, and Fréchet Music Distance.

The design rationale — dataset source, chunking policy, augmentations, metadata
conditioning, loss masking, schema — lives in [notelog.md](notelog.md).

## Installation

```bash
uv sync --all-groups          # runtime + train + eval + dev (pytest)
# or, without uv:
pip install -e ".[train,eval]"
```

Evaluation shells out to the LilyPond binary. If `lilypond` isn't on `PATH`, set
`LILYPOND_BIN` to its full path.

## Usage

Build the full-file training JSONL:

```bash
lilybench build-dataset \
  --input-dir data/bmdataset/preprocessed \
  --metadata data/bmdataset/metadata.json \
  --output data/fullfile_dataset/all_examples.jsonl
```

Split into train/val/test by source work:

```bash
lilybench build-splits \
  --input-jsonl data/fullfile_dataset/all_examples.jsonl \
  --output-dir data/splits_full
```

## Training

LoRA training (see `lilybench/models/registry.py` for known model ids):

```bash
python -m lilybench.stages.training.train_lora \
  --model-id phi4 \
  --train data/splits_full/train.jsonl \
  --val data/splits_full/val.jsonl \
  --output-dir runs/phi4_lora \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate 5e-5 \
  --bf16
```

Known model ids: `gpt-oss`, `phi4`, `qwen-coder`, `deepseek-coder`, `codestral`.
All models are trained and evaluated on raw LilyPond text only — no special
structural tokens are added to any tokenizer vocabulary. Key/time/tempo/voice
information is expressed via the native LilyPond directives (`\key`, `\time`,
`\tempo`, `\new Voice …`).

SLURM templates in `slurm/train/` and `slurm/infer/` are env-var driven:

```bash
MODEL_ID=qwen-coder \
TRAIN_JSONL=data/splits_full/train.jsonl \
VAL_JSONL=data/splits_full/val.jsonl \
OUTPUT_DIR=runs/qwen_coder_lora \
sbatch slurm/train/train_multimodel.slurm

MODEL_ID=qwen-coder REGIME=zero NUM_SAMPLES=100 \
  sbatch slurm/infer/infer_multimodel.slurm
```

> Codestral-22B is a **gated** model on HuggingFace. Accept the license and
> export `HF_TOKEN` (or run `huggingface-cli login`) before the first load.

## Post-Inference

Extract generated LilyPond samples from SLURM inference logs:

```bash
python scripts/extract_detokenized.py \
  --input-dir data/inference/outputs \
  --output-dir data/inference/samples
```

## Evaluation

Text and MIDI analysis:

```bash
python scripts/eval_extracted_ly.py data/inference/samples \
  --out data/inference/sample_eval/eval.jsonl \
  --summary data/inference/sample_eval/summary.json \
  --midi-dir data/inference/sample_eval/midi
```

### Fréchet Music Distance (FMD)

LilyBench reports FMD as a primary distributional quality metric, with
LilyBERT as the symbolic-music embedder applied directly to LilyPond source.
Report FMD against two reference sets:

- **in-domain:** held-out test split (`data/splits_full/test.jsonl`)
- **out-of-domain:** Mutopia LilyPond corpus

```bash
pip install -e ".[eval]"

python scripts/eval_fmd.py \
  --generations-dir data/inference/samples/phi4_zero \
  --reference-kind test \
  --reference-path data/splits_full/test.jsonl \
  --embedder-checkpoint /path/to/lilybert \
  --out data/inference/sample_eval/fmd_phi4_zero_test.json
```

LilyBERT checkpoint: https://github.com/CSCPadova/lilybert. FMD is computed
as in Retkowski et al. 2024; a self-vs-self FMD on the reference set is ≈ 0.

## Development

Tests are written in pytest (with pytest-xdist for parallel runs):

```bash
uv run pytest                  # sequential
uv run pytest -n auto          # parallel across CPU cores
uv run pytest -k augmentations # filter by substring
```

Pytest config is in `[tool.pytest.ini_options]` in `pyproject.toml`; fixtures
are in `tests/conftest.py`.

## Reproducibility

### Run order (end-to-end)

1. Build the full-file dataset: `lilybench build-dataset …`
2. Split: `lilybench build-splits …`
3. Train: `MODEL_ID=… TRAIN_JSONL=… VAL_JSONL=… OUTPUT_DIR=… sbatch slurm/train/train_multimodel.slurm`
4. Infer: `MODEL_ID=… REGIME={zero,few,lora} [LORA_PATH=…] sbatch slurm/infer/infer_multimodel.slurm`
5. Extract detokenized `.ly`: `python scripts/extract_detokenized.py …`
6. Evaluate: `python scripts/eval_extracted_ly.py …` and `python scripts/eval_fmd.py …`

### Expected artifacts

- Trained adapters/checkpoints under `runs/.../final`
- Inference logs under `logs/` (SLURM `%j` job-id naming)
- Extracted LilyPond samples under `data/inference/samples/exp*/sample_*.ly`
- Evaluation outputs under `data/inference/sample_eval/`

### Determinism notes

- Dataset build is deterministic: seeds are `file_seed ^ variant.seed_salt` where `file_seed` is a hash of the filename stem.
- Inference scripts set deterministic seeds per sample (`1234 + i`).
- Generation is still stochastic (`do_sample=True`), so outputs can vary across runs/hardware. Compare trends across matched settings, not exact text identity.

## Project structure

```
lilybench/
  cli.py                          - CLI (build-dataset, build-splits)
  models/registry.py              - Model registry (HF id, dtype, chat template, LoRA targets)
  stages/
    dataset/
      build_fullfile_dataset.py   - Full-file JSONL builder
      prelude.py                  - Prelude boundary detection
      augmentations.py            - shuffle / drop / inline + brace-balance gate
      metadata_header.py          - Metadata resolution and %% === METADATA === block
      training_dataset.py         - Tokenizing dataset loader with char-range loss masking
    splitting/build_splits.py     - Train/val/test split by base work
    training/
      train_lora.py               - LoRA training
      eval_lora.py                - LoRA evaluation
scripts/
  extract_detokenized.py          - Extract .ly files from inference outputs
  eval_extracted_ly.py            - Text + MIDI evaluation
  eval_fmd.py                     - Fréchet Music Distance
  translate_preprocessed_to_nederlands.py - one-shot corpus fixer (notelog §8.1)
tests/                            - pytest suite
slurm/
  train/                          - Training job templates
  infer/                          - Inference job templates
```

## Data structure

```
data/
  bmdataset/
    preprocessed/                 - Input .ly files (include-resolved, nederlands-pitched)
    metadata.json                 - Per-piece metadata (composer, period, form, instruments)
  fullfile_dataset/               - Built JSONL (all_examples.jsonl)
  splits_full/                    - train.jsonl / val.jsonl / test.jsonl
  inference/
    outputs/                      - Raw SLURM inference .out files
    samples/                      - Extracted .ly files
    sample_eval/                  - eval.jsonl, summary.json, midi/
```

## License

MIT
