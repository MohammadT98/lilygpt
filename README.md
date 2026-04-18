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

LoRA training is driven by Hydra (see `lilybench/models/registry.py` for known
model ids, `configs/train.yaml` for tunables):

```bash
python -m lilybench.train \
  model=phi4 \
  data.train=data/splits_full/train.jsonl \
  data.val=data/splits_full/val.jsonl \
  output_dir=runs/phi4_lora \
  epochs=3 batch_size=1 gradient_accumulation_steps=32 \
  learning_rate=5e-5 bf16=true

# Multi-run sweep to SLURM via submitit
python -m lilybench.train --multirun model=phi4,qwen-coder hydra/launcher=slurm_train
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

## Inference

Generation is a Hydra entry point; regime picks zero/few/lora:

```bash
python -m lilybench.infer model=phi4 regime=zero num_samples=100
python -m lilybench.infer model=phi4 regime=lora regime.path=runs/phi4_lora/final
python -m lilybench.infer model=phi4 regime=few regime.fewshot_file=configs/fewshot/phi4.txt

# Multi-run sweep to SLURM
python -m lilybench.infer --multirun model=phi4,qwen-coder regime=zero,lora \
  hydra/launcher=slurm_infer
```

Samples are written to `${output_dir}/samples/sample_####.ly`. If you still need
to parse older SLURM log files, `lilybench/evaluate/extract_detokenized.py`
remains available as an argparse helper.

## Evaluation

Three Hydra entry points. Text and MIDI analysis:

```bash
python -m lilybench.evaluate.text_midi \
  input_dir=data/inference/samples \
  out=data/inference/sample_eval/eval.jsonl \
  summary=data/inference/sample_eval/summary.json \
  midi_dir=data/inference/sample_eval/midi
```

Held-out loss (requires a trained LoRA adapter):

```bash
python -m lilybench.evaluate.loss \
  model=phi4 \
  lora_path=runs/phi4_lora/final \
  data=data/splits_full/test.jsonl
```

### Fréchet Music Distance (FMD)

LilyBench reports FMD as a primary distributional quality metric, with
LilyBERT as the symbolic-music embedder applied directly to LilyPond source.
Report FMD against two reference sets:

- **in-domain:** held-out test split (`data/splits_full/test.jsonl`)
- **out-of-domain:** Mutopia LilyPond corpus

```bash
pip install -e ".[eval]"

python -m lilybench.evaluate.fmd \
  generations_dir=data/inference/samples/phi4_zero \
  reference_kind=test \
  reference_path=data/splits_full/test.jsonl \
  embedder_checkpoint=/path/to/lilybert \
  out=data/inference/sample_eval/fmd_phi4_zero_test.json
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
3. Train: `python -m lilybench.train model=… data.train=… data.val=… output_dir=…` (or `sbatch slurm/train/train_multimodel.slurm` with the legacy env-var contract)
4. Infer: `python -m lilybench.infer model=… regime={zero,few,lora} [regime.path=…]`
5. Evaluate: `python -m lilybench.evaluate.text_midi …` / `python -m lilybench.evaluate.loss …` / `python -m lilybench.evaluate.fmd …`

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
  cli.py                          - argparse CLI (build-dataset, build-splits)
  train.py                        - Hydra entry point: LoRA training
  infer.py                        - Hydra entry point: inference (zero/few/lora)
  models/registry.py              - Model registry (HF id, dtype, chat template, LoRA targets)
  preprocess/
    build_dataset.py              - Full-file JSONL builder
    build_splits.py               - Train/val/test split by base work
    prelude.py                    - Prelude boundary detection
    augmentations.py              - shuffle / drop / inline + brace-balance gate
    metadata_header.py            - Metadata resolution and %% === METADATA === block
  data/
    training_dataset.py           - Tokenizing dataset loader with char-range loss masking
  evaluate/
    loss.py                       - Hydra entry point: held-out loss of a LoRA adapter
    text_midi.py                  - Hydra entry point: text + MIDI analysis
    fmd.py                        - Hydra entry point: Fréchet Music Distance
    extract_detokenized.py        - argparse helper: extract .ly files from log files
configs/                          - Hydra config tree (train, infer, evaluate, model, regime, launcher)
scripts/
  translate_preprocessed_to_nederlands.py - one-shot corpus fixer (notelog §8.1)
tests/                            - pytest suite
slurm/
  train/                          - Training job templates (thin Hydra wrappers)
  infer/                          - Inference job templates (thin Hydra wrappers)
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
