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
7. Splits into train/val/test sets

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

Generate full assignment dataset:
```bash
python scripts/prepare_full_assignment_dataset.py
```

Build train/val/test splits:
```bash
python src/lilynorm/stages/splitting/build_splits.py \
  --input-jsonl data/full_assignment_dataset/all_examples.jsonl \
  --output-dir data/splits_full
```

## Batch scripts

Windows batch scripts in `bin/`:

- `normalize_only.bat` - Just normalization
- `run_dataset_single_voice.bat` - Filter single-voice pieces
- `run_full_pipeline.bat` - Complete pipeline

## Training

Standard full-sequence training:
```bash
python -m lilynorm.stages.training.train_standard \
  --train data/splits_full/train.jsonl \
  --val data/splits_full/val.jsonl \
  --output-dir runs/experiment \
  --epochs 3 \
  --batch-size 1 \
  --learning-rate 5e-5
```

Weighted loss training (for structural tokens):
```bash
python -m lilynorm.stages.training.train_weighted \
  --train data/splits_full/train.jsonl \
  --val data/splits_full/val.jsonl \
  --output-dir runs/experiment \
  --structural-weight 5.0 \
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
    splitting/            - Train/val split
      build_splits.py
    training/             - LoRA fine-tuning
      train_standard.py   - Standard training
      train_weighted.py   - Weighted loss training
  utils/
    options.py            - Configuration
    formatting.py         - Output formatting
```

## Configuration

Normalization profiles in `configs/profiles/`:
- `strict_strip.yaml` - Aggressive stripping
- `keep_engraving.yaml` - Preserve more directives

Default settings in `configs/defaults.yaml`.

## Deprecated Methods

Some experimental approaches that didn't work are preserved in `deprecated/` for research documentation:

- **Continuation method** (`deprecated/continuation_method/`) - Split each piece into 3 continuation examples
  - **Status**: ❌ Failed - Generated invalid syntax
  - **Experiments**: Exp 1-4
  - **Result**: Only 34% structurally complete, model learned garbage output
  - **Replaced by**: Full assignment method (Exp 5+)

See `deprecated/continuation_method/README.md` for detailed analysis of why it failed.

**Important**: Do not use deprecated methods for new experiments. They are kept for thesis documentation only.

## License

MIT
