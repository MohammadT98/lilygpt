# lilygpt

Pipeline for normalizing LilyPond music notation and preparing training datasets for large language model fine-tuning.

## Overview

This tool processes raw LilyPond source files through a multi-stage normalization pipeline, producing clean, structurally consistent training data:

1. **File resolution** - Merges `\include` directives, splits multi-movement scores
2. **Preprocessing** - Removes comments, normalizes whitespace
3. **Syntax normalization** - Expands music functions/transposes, unfolds repeats, normalizes tuplets
4. **Forma handling** - Prepends structure and inlines `\forma`
5. **Stripping** - Removes engraving directives (layout, dynamics, articulations)
6. **Postprocessing** - Fixes malformed syntax patterns
7. **Dataset generation** - Builds datasets and creates train/val/test splits

## Installation

```bash
cd lilygpt
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
  --normalized-out data/normalized_dataset
```
Note: Some normalization steps call the LilyPond binary. If it is not on PATH, set
`LILYPOND_BIN` to its full path (e.g., `C:\lilypond-2.24.4-mingw-x86_64\lilypond-2.24.4\bin\lilypond.exe`).

Generate full assignment dataset:
```bash
python -m lilynorm.stages.dataset.build_assignment_dataset
```

Build train/val/test splits:
```bash
python src/lilynorm/stages/splitting/build_splits.py \
  --input-jsonl data/assignment_dataset/all_examples.jsonl \
  --output-dir data/splits_full
```

## Batch scripts

Convenience scripts for Windows (in `bin/`):

- `run_full_pipeline.bat` - End-to-end pipeline

## Training

LoRA training:
```bash
python -m lilynorm.stages.training.train_lora \
  --train data/splits_full/train.jsonl \
  --val data/splits_full/val.jsonl \
  --output-dir runs/experiment \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate 5e-5
```
(Effective batch size = 1 × 32 = 32)

SLURM job templates in `slurm/`.

## Post-Inference

Extract generated LilyPond samples from inference output files:
```bash
python scripts/extract_detokenized.py \
  --input-dir data/inference/outputs \
  --output-dir data/inference/samples
```

This extracts all "Detokenized Output" sections and saves them as `.ly` files organized by experiment.

## Evaluation

Evaluate extracted LilyPond files (text checks + MIDI analysis):
```bash
python scripts/eval_extracted_ly.py data/inference/samples \
  --out data/inference/sample_eval/eval.jsonl \
  --summary data/inference/sample_eval/summary.json \
  --midi-dir data/inference/sample_eval/midi
```

Outputs:
- `data/inference/sample_eval/eval.jsonl` — per-sample metrics
- `data/inference/sample_eval/summary.json` — aggregate summary
- `data/inference/sample_eval/midi/` — rendered MIDI files (when LilyPond is available)

Note: This evaluation uses the LilyPond binary for rendering (if available) and `music21` for MIDI analysis.

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
      utils/              - Normalization helpers
        brackets.py       - Balanced bracket parsing
    tokenization/         - Tokenizer utilities
      special_tokens.py   - LilyPond special tokens
    dataset/              - Dataset loading for training
      training_dataset.py - Full-sequence dataset
      build_assignment_dataset.py - Build full assignment dataset
    splitting/            - Train/val split
      build_splits.py
    training/             - LoRA fine-tuning
      train_lora.py       - LoRA training
      eval_lora.py        - LoRA evaluation
  utils/
    options.py            - Configuration
scripts/
  extract_detokenized.py  - Extract .ly files from inference outputs
  eval_extracted_ly.py    - Evaluate extracted LilyPond files
  tests/                  - Unit tests
    test_file_resolver.py
    test_normalize_syntax.py
    test_preprocess.py
bin/
  run_full_pipeline.bat   - End-to-end pipeline
  tests/                  - Test batch scripts
slurm/
  train/                  - Training job templates
  infer/                  - Inference job templates
  test/                   - Test set evaluation job templates
```

## Data structure

```
data/
  raw/                    - Raw input LilyPond files
  normalized_dataset/     - Normalized output
  assignment_dataset/     - Built assignment dataset
  splits_full/            - Train/val/test splits
  logs/                   - Processing logs
  inference/
    outputs/              - Raw SLURM inference .out files
    samples/              - Extracted .ly files
    sample_eval/          - Evaluation results
      eval.jsonl          - Per-sample metrics
      summary.json        - Aggregate summary
      midi/               - Rendered MIDI files
```

## License

MIT
