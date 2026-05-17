# LilyBench

LilyBench is an evaluation framework for large language models on
[LilyPond](https://lilypond.org/), a code-like textual score format.
It accompanies the Ital-IA 2026 paper

> **Can LLMs understand LilyPond? A benchmark for symbolic music
> generation and understanding.**
> *Matteo Spanio, Mohammad Torabi, Andrea Poltronieri, Antonio Rodà.*
> Ital-IA 2026 — 6th National Conference on Artificial Intelligence
> (CINI), Rome, Italy.

The framework pairs

* a **generation benchmark** — a fixed metadata-conditioned prompt bank
  evaluated with compile rate, Jensen–Shannon similarity over three MusPy
  descriptors, and a LilyBERT-based Fréchet Music Distance, and
* an **understanding benchmark** — ten ABC-Eval-adapted tasks
  (bar count, metadata QA, bar sequencing, next-bar prediction,
  metadata prediction, music captioning, composer recognition, genre
  recognition, emotion recognition, error detection) scored with
  accuracy, exact match, penalised Kendall-τ, and macro-F1.

Both benchmarks are modular and extendable: new generation regimes plug
in through `lilybench.generation.regimes.Regime`, new understanding
tasks through the `@register_task` decorator. The four backbones used in
the paper — `phi4`, `qwen-coder`, `deepseek-coder`, `codestral` — are
registered out of the box; adding a new HuggingFace model is one
`register_model(ModelSpec(...))` call.

## Installation

```bash
git clone https://github.com/CSCPadova/lilybench.git
cd lilybench
python -m venv .venv && source .venv/bin/activate
pip install -e .
# Optional: bitsandbytes int4 / int8 decoding
pip install -e ".[quant]"
```

LilyBench shells out to the LilyPond binary for the compile-rate metric.
Install LilyPond 2.24.4 from the [official site](https://lilypond.org/) or
your distribution; set `LILYPOND_BIN` if it is not on `PATH`.

## Datasets

The companion archive on Zenodo (DOI to be assigned) bundles:

| Corpus     | Use case                              | Source            |
|------------|---------------------------------------|-------------------|
| BMdataset  | Generation in-domain; understanding   | Spanio et al. 2026 |
| Mutopia    | Generation out-of-domain; understanding | Mutopia Project  |
| EMOPIA     | Emotion-recognition task              | Hung et al. 2021  |

Unpack the archive under `data/` (see [`data/README.md`](data/README.md)
for the expected layout). Two helper scripts are shipped only for users
who want to *rebuild* the archive from upstream sources:
[`scripts/convert_mutopia.py`](scripts/convert_mutopia.py) and
[`scripts/prepare_emopia.py`](scripts/prepare_emopia.py). The
LilyBERT checkpoint used for FMD lives at
<https://github.com/CSCPadova/lilybert>.

## Quickstart

LilyBench exposes a single CLI with four verbs. Every paper experiment
fits in roughly the following pipeline.

### 1. Build the prompt bank

```bash
lilybench prompt-bank build \
    --bmdataset-dir data/bmdataset/preprocessed \
    --bmdataset-metadata data/bmdataset/metadata.json \
    --n 200 --seed 1234 \
    --out data/prompt_bank.jsonl
```

The bank is reused byte-for-byte across every `(model, regime)` cell so
comparisons are fair. The 200-prompt size matches the paper; pass
`--n` to evaluate at a different budget.

### 2. Generate

```bash
# Zero-shot
lilybench generation run \
    --model phi4 --regime zero \
    --prompts data/prompt_bank.jsonl \
    --out runs/phi4_zero

# Few-shot from the training distribution
lilybench generation run \
    --model phi4 --regime few \
    --fewshot-file configs/fewshot_train_distribution.txt \
    --prompts data/prompt_bank.jsonl \
    --out runs/phi4_few

# Few-shot ablation (hand-written A-minor demos, §3.2 of the paper)
lilybench generation run \
    --model phi4 --regime few \
    --fewshot-file configs/fewshot_ablation_amin.txt \
    --prompts data/prompt_bank.jsonl \
    --out runs/phi4_few_ablation
```

Generated `.ly` files land under `runs/<cell>/samples/sample_####.ly`,
with the raw decoded text alongside in `raw_####.txt` for debugging.

### 3. Build the understanding bench

```bash
# Mutopia tasks (eight of the ten)
lilybench understanding build \
    --corpus mutopia \
    --mutopia-manifest data/mutopia/dataset_mutopia.json \
    --tasks all --seed 1234 \
    --out data/understanding/mutopia.jsonl

# EMOPIA emotion task
lilybench understanding build \
    --corpus emopia \
    --emopia-manifest data/emopia/manifest.csv \
    --emopia-ly-root  data/emopia/ly \
    --tasks emotion_recognition --seed 1234 \
    --out data/understanding/emopia.jsonl
```

### 4. Run understanding inference (greedy)

```bash
lilybench understanding run \
    --model phi4 \
    --bench data/understanding/mutopia.jsonl \
    --out runs/understanding
```

One JSONL of predictions per task is written under
`runs/understanding/phi4/<task>.jsonl`.

### 5. Score generations

```bash
lilybench metrics generation \
    --samples runs/phi4_zero/samples \
    --reference-test          data/splits/test.jsonl \
    --reference-mutopia       data/mutopia/dataset_mutopia.json \
    --reference-test-midi-dir data/splits/test_midi \
    --reference-mutopia-midi-dir data/mutopia/midi \
    --lilybert /path/to/lilybert \
    --out runs/phi4_zero/summary.json
```

The output JSON contains `compile_rate`, `js_similarity` (in-domain and
out-of-domain), and `fmd` (in-domain and out-of-domain), matching the
columns of Table 1 in the paper. Provide only the metrics you care about
— each `--reference-*` is optional.

### 6. Score understanding

```bash
lilybench metrics understanding \
    --bench data/understanding/mutopia.jsonl \
    --predictions runs/understanding/phi4 \
    --model-id phi4 \
    --out runs/understanding/phi4/summary.json
```

Per-task metrics plus a macro / weighted aggregate are written to
`summary.json`.

## Architecture

The package is intentionally small and orthogonal:

```
lilybench/
  cli.py                 unified argparse CLI (prompt-bank, generation,
                         understanding, metrics)
  models.py              backbone registry + HF loader (+ DeepSeek shims)
  utils.py               JSONL I/O, HF env helpers, LilyPond binary lookup
  data/
    bmdataset.py         BMdataset preprocessed/*.ly loader
    mutopia.py           Mutopia manifest loader
    emopia.py            EMOPIA manifest-CSV loader (midi2ly outputs)
    splits.py            deterministic work-level splitter
    types.py             CorpusEntry dataclass (uniform input type)
  generation/
    prompt_bank.py       stratified prompt-bank builder (paper: 200 prompts)
    regimes.py           ZeroShot, FewShot, plus a `register_regime` hook
    runner.py            backbone + regime + bank -> sample directory
    metadata_block.py    %% === METADATA === block renderer
  understanding/
    base.py              UnderstandingTask abstract base class
    registry.py          @register_task decorator + lookup
    runner.py            greedy-decoding inference loop
    tasks/               ten registered tasks, one module each
    bar_utils.py         |-delimited bar splitting / counting
    corruptor.py         five error-injection categories
    score_metadata.py    extract/mask key / meter / note-length
    title_parser.py      extract title from \header blocks
    midi_to_lily.py      midi2ly subprocess wrapper (EMOPIA prep)
  metrics/
    compile_rate.py      LilyPond compile rate
    muspy_descriptors.py the three MusPy descriptors used by JS-similarity
    js_similarity.py     JS-similarity over Gaussian-fit descriptors
    fmd.py               LilyBERT-based Fréchet Music Distance
    understanding.py     accuracy / Kendall-τ / F1 / bar-count tolerance
```

### Adding a new generation regime

```python
from lilybench.generation.regimes import Regime, register_regime

class ChainOfThought(Regime):
    name = "cot"
    def build_prompt(self, prompt, *, tokenizer):
        ...  # build any chat-template string here

register_regime(ChainOfThought)
```

Then `lilybench generation run --regime cot ...` works without any
other edits.

### Adding a new understanding task

```python
from lilybench.understanding import UnderstandingTask, register_task

@register_task
class HarmonicFunctionQA(UnderstandingTask):
    name = "harmonic_function"
    template_kind = "multiple_choice"
    task_instruction = "Pick the most plausible Roman-numeral function..."
    default_n = 60

    def build(self, corpus, *, n, seed): ...
    def score(self, bench, predictions): ...
```

The task is picked up automatically by `lilybench understanding build`
(`--tasks all`) and `lilybench metrics understanding`.

### Registering a new backbone

```python
from lilybench.models import ModelSpec, register_model

register_model(ModelSpec(
    model_id="my-model",
    hf_id="org/my-model",
    dtype="bf16",
    family="general",
))
```

## Reproducing the paper

The paper reports four backbones × three generation regimes (zero,
few-train, few-ablation) and four backbones × ten understanding tasks.
With the Zenodo archive unpacked, the full sweep is twelve generation
runs and four understanding runs. The `slurm/` directory contains three
thin wrappers (`generation.slurm`, `understanding.slurm`,
`metrics.slurm`) showing how to drive the CLI from a SLURM cluster.

Determinism notes:

* The prompt bank is built with `seed=1234`; per-prompt inference seeds
  are `seed_base + i`. Generation is `do_sample=True`, so numeric
  metrics drift slightly between runs — compare trends, not exact text.
* Understanding decoding is greedy (`do_sample=False`, `T=0`,
  `max_new_tokens=20`) per ABC-Eval, so understanding predictions are
  deterministic on identical hardware/library versions.
* Splits are computed at the work level (no part-of-the-same-piece can
  cross the split boundary). The Zenodo archive ships the splits used
  in the paper.

## Testing

```bash
pytest                  # sequential
pytest -n auto          # parallel via pytest-xdist
```

## License

Code: MIT (see [LICENSE](LICENSE)).
Datasets retain their upstream licenses; see the corresponding entries on
Zenodo for redistribution terms.

## Citation

If you use LilyBench in your research, please cite:

```bibtex
@inproceedings{spanio2026lilybench,
  title  = {Can LLMs understand LilyPond? A benchmark for symbolic music
            generation and understanding},
  author = {Spanio, Matteo and Torabi, Mohammad and Poltronieri, Andrea
            and Rod{\`a}, Antonio},
  booktitle = {Proceedings of Ital-IA 2026},
  year   = {2026},
}
```
