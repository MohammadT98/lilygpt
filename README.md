# lilybench

Pipeline for normalizing LilyPond music notation and preparing training datasets for large language model fine-tuning.

## Overview

This tool processes raw LilyPond source files through a multi-stage normalization pipeline, producing clean, structurally consistent training data:

1. **File resolution** - Resolves `\include` directives, splits multi-forma scores
2. **Preprocessing** - Removes comments, normalizes whitespace
3. **Syntax normalization** - Expands music functions/transposes, unfolds repeats, normalizes tuplets
4. **Forma handling** - Prepends structure and inlines `\forma`
5. **Stripping** - Removes engraving directives (layout, dynamics, articulations)
6. **Postprocessing** - Fixes malformed syntax patterns
7. **Dataset generation** - Builds datasets and creates train/val/test splits

## Installation

```bash
cd lilybench
pip install -e .
```

For training:
```bash
pip install -e ".[train]"
```

## Usage

Normalize LilyPond files:
```bash
python -m lilybench.cli \
  --input data/raw \
  --normalized-out data/normalized_dataset
```
Note: Some normalization steps call the LilyPond binary. If it is not on PATH, set
`LILYPOND_BIN` to its full path (e.g., `C:\lilypond-2.24.4-mingw-x86_64\lilypond-2.24.4\bin\lilypond.exe`).

Generate full assignment dataset:
```bash
python -m lilybench.stages.dataset.build_assignment_dataset
```

Build train/val/test splits:
```bash
python -m lilybench.stages.splitting.build_splits \
  --input-jsonl data/assignment_dataset/all_examples.jsonl \
  --output-dir data/splits_full
```

## Batch scripts

Convenience scripts for Windows (in `bin/`):

- `run_full_pipeline.bat` - End-to-end pipeline

## Training

LoRA training:
```bash
python -m lilybench.stages.training.train_lora \
  --model-id gpt-oss \
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

### Multi-model LoRA

Model selection is driven by a registry at `src/lilybench/models/registry.py`.
Known ids: `gpt-oss`, `phi4`, `qwen-coder`, `deepseek-coder`, `codestral`.
The registry resolves the HuggingFace id, chat-template kind, dtype, LoRA
target modules, and whether the tokenizer should accept the LilyPond
structural tokens `<KEY:...>`, `<TIME:...>`, `<TEMPO:...>`, `<VOICE>` (only
`gpt-oss` does by default; other models fall back to inline text).

Train any registered model by passing `--model-id`:
```bash
python -m lilybench.stages.training.train_lora \
  --model-id phi4 \
  --train data/splits_full/train.jsonl \
  --val data/splits_full/val.jsonl \
  --output-dir runs/phi4_lora \
  --bf16
```

SLURM wrappers for the full matrix live in
`slurm/train/train_multimodel.slurm` and
`slurm/infer/infer_multimodel.slurm`; both are env-var driven:
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

### Fréchet Music Distance (FMD)

LilyBench reports FMD as a primary distributional quality metric, with
LilyBERT as the symbolic-music embedder applied directly to LilyPond source.
Report FMD against two reference sets:

- **in-domain:** held-out test split (`data/splits_full/test.jsonl`)
- **out-of-domain:** Mutopia LilyPond corpus

```bash
pip install -e ".[eval]"

# In-domain reference
python scripts/eval_fmd.py \
  --generations-dir data/inference/samples/phi4_zero \
  --reference-kind test \
  --reference-path data/splits_full/test.jsonl \
  --embedder-checkpoint /path/to/lilybert \
  --out data/inference/sample_eval/fmd_phi4_zero_test.json

# Out-of-domain reference
python scripts/eval_fmd.py \
  --generations-dir data/inference/samples/phi4_zero \
  --reference-kind mutopia \
  --reference-path data/mutopia_ly \
  --embedder-checkpoint /path/to/lilybert \
  --out data/inference/sample_eval/fmd_phi4_zero_mutopia.json
```

LilyBERT checkpoint: https://github.com/CSCPadova/lilybert (CodeBERT-based,
125M params, MLM-pretrained on LilyPond). FMD is computed as in Retkowski et
al. 2024; a self-vs-self FMD on the reference set should be ≈ 0.

## Reproducibility

### Environment

- Python `>=3.10` (see `pyproject.toml`)
- Core deps: `pandas`, `music21`
- Training deps: `transformers`, `torch`, `peft`, `accelerate`, `tensorboard`
- External tools: LilyPond (for compile/render checks), `music21` (MIDI analysis)

Install:

```bash
pip install -e ".[train]"
```

### Run order (end-to-end)

1. Normalize raw LilyPond files:

```bash
python -m lilybench.cli \
  --input data/raw \
  --normalized-out data/normalized_dataset
```

2. Build assignment dataset:

```bash
python -m lilybench.stages.dataset.build_assignment_dataset
```

3. Build train/val/test splits:

```bash
python -m lilybench.stages.splitting.build_splits \
  --input-jsonl data/assignment_dataset/all_examples.jsonl \
  --output-dir data/splits_full
```

4. Train (examples):

```bash
sbatch slurm/train/train_exp10_voice_tokens.slurm
sbatch slurm/train/train_exp11_voice_tokens_eff16.slurm
sbatch slurm/train/train_exp12_voice_tokens_eff4.slurm
```

5. Inference (examples):

```bash
sbatch slurm/infer/infer_exp10_voice_tokens.slurm
sbatch slurm/infer/infer_exp11_voice_tokens.slurm
sbatch slurm/infer/infer_exp12_voice_tokens.slurm
```

6. Extract detokenized `.ly` outputs:

```bash
python scripts/extract_detokenized.py \
  --input-dir data/inference/outputs \
  --output-dir data/inference/samples
```

7. Evaluate generated samples:

```bash
python scripts/eval_extracted_ly.py data/inference/samples \
  --out data/inference/sample_eval/eval.jsonl \
  --summary data/inference/sample_eval/summary.json \
  --midi-dir data/inference/sample_eval/midi
```

### Expected artifacts

- Trained adapters/checkpoints under `runs/.../final` (training scripts)
- Inference logs under `logs/` (SLURM `%j` job-id naming)
- Extracted LilyPond samples under `data/inference/samples/exp*/sample_*.ly`
- Evaluation outputs:
  - `data/inference/sample_eval/eval.jsonl`
  - `data/inference/sample_eval/summary.json`
  - `data/inference/sample_eval/midi/` (grouped by experiment where available)

### Determinism notes

- Inference scripts set deterministic seeds per sample (`1234 + i`).
- Generation is still stochastic (`do_sample=True`), so outputs can vary across runs/hardware.
- Small metric fluctuations are expected; compare trends across matched settings, not exact text identity.

## Project structure

```
src/lilybench/
  cli.py                  - Command-line interface
  normalize.py            - Pipeline orchestration
  stages/
    normalization/        - Text normalization stages
      file_resolver.py    - Resolve includes, split on \forma
      preprocess.py       - Remove comments, clean whitespace
      normalize_syntax.py - Resolve transpose, unfold repeats, normalize tuplets
      forma.py            - Prepend structure and inline \forma
      engrave_strip.py    - Strip engraving directives
      postprocessing.py   - Fix malformed patterns
      utils/              - Normalization helpers
        brackets.py       - Balanced bracket parsing
    dataset/              - Dataset loading for training
      training_dataset.py - Full-sequence dataset
      build_assignment_dataset.py - Build full assignment dataset
    splitting/            - Train/val/test split
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
