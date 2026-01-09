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

Convenience scripts for Windows (in `bin/`):

- `normalize_only.bat` - Normalization only
- `run_dataset_single_voice.bat` - Single-voice filtering
- `run_full_pipeline.bat` - End-to-end pipeline

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

Failed experimental approaches are preserved in `deprecated/` for research documentation:

**Continuation method** (`deprecated/continuation_method/`) - Experiments 1-4
- Approach: Three-way fragmentation of musical assignments (start/middle/near-end splits)
- Outcome: Structural incompleteness (66% unclosed examples) resulted in invalid syntax generation
- Status: Superseded by full assignment method (Experiments 5+)

Refer to `deprecated/continuation_method/README.md` for detailed failure analysis.

These implementations are maintained solely for thesis documentation and should not be used in production.

## License

MIT
