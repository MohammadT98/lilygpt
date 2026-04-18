# data/

This directory is git-ignored. The pipeline generates the following structure:

```
data/
  raw/                    Raw input LilyPond files (place your dataset here)
  normalized_dataset/     Output of the normalization pipeline
  assignment_dataset/     Full assignment dataset (all_examples.jsonl)
  splits_full/            Train/val/test JSONL splits
  logs/                   Processing logs
  inference/
    outputs/              Raw SLURM inference .out files
    samples/              Extracted .ly files (per experiment)
    sample_eval/          Evaluation results
      eval.jsonl          Per-sample metrics
      summary.json        Aggregate summary
      midi/               Rendered MIDI files
```

## Generating the data

1. Place raw `.ly` files under `data/raw/`.
2. Run the pipeline (see root README for details):

```bash
python -m lilybench.cli --input data/raw --normalized-out data/normalized_dataset
python -m lilybench.stages.dataset.build_assignment_dataset
python -m lilybench.stages.splitting.build_splits \
  --input-jsonl data/assignment_dataset/all_examples.jsonl \
  --output-dir data/splits_full
```
