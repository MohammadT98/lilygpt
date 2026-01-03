# lilynorm

Pipeline for normalizing LilyPond music scores and preparing datasets for GPT fine-tuning.

## What it does

Takes LilyPond source files and converts them to a normalized form suitable for machine learning:

1. Resolves `\include` directives and splits on `\forma` blocks
2. Removes comments and cleans whitespace
3. Expands `\relative`, `\transpose`, `\repeat` to absolute notation
4. Strips engraving directives (`\override`, `\markup`, dynamics, etc.)
5. Fixes common malformed patterns from real-world scores
6. Tokenizes for GPT training (standard or continuation-style)
7. Splits into train/val sets

## Installation

```bash
cd lilynorm
pip install -e .
```

For training:
```bash
pip install -e ".[train]"
```

## Usage

Normalize LilyPond files:
```bash
python -m lilynorm.cli \
  --input data/raw \
  --normalized-out data/normalized \
  --skip-tokenize
```

Full pipeline (normalize + tokenize):
```bash
python -m lilynorm.cli \
  --input data/raw \
  --normalized-out data/normalized \
  --tokenized-out data/tokenized
```

Generate continuation-style dataset:
```bash
python scripts/prepare_continuation_dataset.py \
  --input data/normalized \
  --output data/continuation \
  --splits-per-piece 3
```

Build train/val/test splits:
```bash
python -m lilynorm.stages.splitting.build_splits \
  --tokenized-root data/tokenized \
  --output-dir data/splits
```

## Batch scripts

Windows batch scripts in `bin/`:

- `normalize_only.bat` - Just normalization
- `run_dataset_single_voice.bat` - Filter single-voice pieces
- `run_full_pipeline.bat` - Complete pipeline
- `prepare_continuation_data.bat` - Generate continuation dataset

## Training

Standard full-sequence training:
```bash
python -m lilynorm.stages.training.train_standard \
  --train data/splits/train.jsonl \
  --val data/splits/val.jsonl \
  --output-dir runs/experiment \
  --epochs 3 \
  --batch-size 1 \
  --learning-rate 5e-5
```

Continuation-style (masked loss):
```bash
python -m lilynorm.stages.training.train \
  --train data/splits/train.jsonl \
  --val data/splits/val.jsonl \
  --output-dir runs/experiment \
  --epochs 3
```

SLURM job templates in `slurm/`.

## Project structure

```
src/lilynorm/
  cli.py                  - Command-line interface
  normalize.py            - Pipeline orchestration
  stages/
    normalization/        - Text normalization stages
      file_resolver.py    - Resolve includes, split on \forma
      preparse.py         - Remove comments, clean whitespace
      expand.py           - Expand relative/transpose/repeat
      engrave_strip.py    - Strip engraving directives
      postprocessing.py   - Fix malformed patterns
      utils/
        voice_extraction.py - Voice detection heuristics
    tokenization/         - GPT tokenization
      tokenize_gpt.py     - Basic tokenization
      dataset_standard.py - Full-sequence dataset
      dataset_continuation.py - Continuation dataset
    splitting/            - Train/val split
      build_splits.py
    training/             - LoRA fine-tuning
      train.py            - Continuation-style training
      train_standard.py   - Standard training
  utils/
    options.py            - Configuration
    formatting.py         - Output formatting
```

## Configuration

Normalization profiles in `configs/profiles/`:
- `strict_strip.yaml` - Aggressive stripping
- `keep_engraving.yaml` - Preserve more directives

Default settings in `configs/defaults.yaml`.

## License

MIT
