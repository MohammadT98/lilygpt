# lilygpt

Pipeline for normalizing LilyPond music notation and preparing training datasets for large language model fine-tuning.

## Overview

This tool processes raw LilyPond source files through a multi-stage normalization pipeline, producing clean, structurally consistent training data:

1. **File resolution** - Merges `\include` directives, splits multi-movement scores
2. **Preprocessing** - Removes comments, normalizes whitespace
3. **Expansion** - Converts relative notation to absolute, unfolds repeats and tuplets
4. **Stripping** - Removes engraving directives (layout, dynamics, articulations)
5. **Postprocessing** - Fixes malformed syntax patterns
6. **Dataset generation** - Creates train/validation/test splits with balanced work distribution

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
  --normalized-out data/normalized
```

Generate full assignment dataset:
```bash
python -m lilynorm.stages.dataset.build_full_assignment_dataset
```

Build train/val/test splits:
```bash
python src/lilynorm/stages/splitting/build_splits.py \
  --input-jsonl data/full_assignment_dataset/all_examples.jsonl \
  --output-dir data/splits_full
```

## Batch scripts

Convenience scripts for Windows (in `bin/`):

- `run_full_pipeline.bat` - End-to-end pipeline

## Training

Standard full-sequence training:
```bash
python -m lilynorm.stages.training.train_lora \
  --train data/splits_full/train.jsonl \
  --val data/splits_full/val.jsonl \
  --output-dir runs/experiment \
  --epochs 3 \
  --batch-size 1 \
  --learning-rate 5e-5
```

SLURM job templates in `slurm/`.

## Notebooks

The `notebooks/` directory contains standalone prompt engineering experiments (zero-shot, few-shot, prompt configuration tests) for evaluating LLM generation capabilities. These are separate from the main normalization and training pipeline.

## Project structure

```
src/lilynorm/
  cli.py                  - Command-line interface
  normalize.py            - Pipeline orchestration
  stages/
    normalization/        - Text normalization stages
      file_resolver.py    - Resolve includes, split on \forma
      preprocess.py       - Remove comments, clean whitespace
      normalize_syntax.py - Expand relative/transpose/repeat
      forma.py            - Prepend structure and inline \forma
      engrave_strip.py    - Strip engraving directives
      postprocessing.py   - Fix malformed patterns
    tokenization/         - Tokenizer utilities
      special_tokens.py   - LilyPond special tokens
    dataset/              - Dataset loading for training
      training_dataset.py - Full-sequence dataset
    splitting/            - Train/val split
      build_splits.py
    training/             - LoRA fine-tuning
      train_lora.py       - LoRA training
      train_weighted.py   - Weighted loss training
  utils/
    options.py            - Configuration
```

## License

MIT
