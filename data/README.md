# data/

This directory is git-ignored. The pipeline expects and generates the following
structure:

```
data/
  bmdataset/
    preprocessed/         Input .ly files (include-resolved, nederlands-pitched)
    metadata.json         Per-piece metadata (composer, period, form, instruments)
  fullfile_dataset/
    all_examples.jsonl    Output of `lilybench build-dataset`
  splits_full/            train.jsonl / val.jsonl / test.jsonl
  inference/
    outputs/              Raw SLURM inference .out files
    samples/              Extracted .ly files (per experiment)
    sample_eval/          Evaluation results
      eval.jsonl          Per-sample metrics
      summary.json        Aggregate summary
      midi/               Rendered MIDI files
```

## Generating the data

1. Obtain `bmdataset/preprocessed/` and `bmdataset/metadata.json`.
2. Run the pipeline (see root README for details):

```bash
lilybench build-dataset \
  --input-dir data/bmdataset/preprocessed \
  --metadata data/bmdataset/metadata.json \
  --output data/fullfile_dataset/all_examples.jsonl

lilybench build-splits \
  --input-jsonl data/fullfile_dataset/all_examples.jsonl \
  --output-dir data/splits_full
```
