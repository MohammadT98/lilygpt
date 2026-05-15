# LilyBench results — out-of-the-box benchmark

**Date:** 2026-05-15 (article pivot)

**Scope.** Four open-weight code/general LLMs evaluated out of the box (no fine-tuning) on two complementary benchmarks: (1) LilyPond *generation* under zero-shot and few-shot prompting, and (2) LilyPond *understanding* on the Mutopia corpus, adapted from arXiv-2509.23350v1 ("ABC-Eval").

| short id | HF id | family | params |
|---|---|---|---|
| phi4 | `microsoft/phi-4` | general | 14B |
| qwen-coder | `Qwen/Qwen2.5-Coder-14B-Instruct` | code | 14B |
| codestral | `mistralai/Codestral-22B-v0.1` | code | 22B |
| deepseek-coder | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | code | 16B MoE |

Removed from the paper: LoRA fine-tuning (archived numbers in [RESULTS_lora_archive.md](RESULTS_lora_archive.md), training code remains in the repo) and Gemma-2-27B (could not finish 1000-sample inference within the 24h cluster wall).

---

## 1. Generation benchmark

Each model is prompted with the same 1000-record metadata-conditioned prompt bank in two regimes:

- **zero**: chat-templated system + user prompt, no demonstrations.
- **few**: three demonstrations sampled deterministically from the bmdataset train split (Vivaldi concerto + Corrette + Bach).

We also keep a "simple-demo" ablation row to show what happens when the few-shot examples are hand-written A-minor / 4-bar stubs that don't match the corpus shape.

### 1.1 Metrics

| metric | range | direction | notes |
|---|---|---|---|
| `compiles` | [0, 1] | ↑ | fraction of generations that LilyPond renders to a valid `.mid` |
| `JS/test`, `JS/muto` | [25, 100] | ↑ | `100·exp(-2·mean_JS)` over `polyphony_rate`, `groove_consistency`, `scale_consistency`; references = bmdataset / Mutopia |
| `FMD/test`, `FMD/muto` | ≥ 0 | ↓ | Fréchet distance over LilyBERT layer-6 [CLS] embeddings |

`JS-sim` looks at the music *after* MIDI rendering; `FMD` looks at the LilyPond *text* before rendering. Used together they catch failure modes either misses.

### 1.2 Results

| cell | n | compiles | JS/test ↑ | JS/muto ↑ | FMD/test ↓ | FMD/muto ↓ |
|---|---|---|---|---|---|---|
| **Zero-shot** | | | | | | |
| phi4_zero | 1000 | 71.1% | **83.27** | **81.24** | 0.933 | 1.419 |
| qwen-coder_zero | 1000 | 69.0% | **84.85** | 73.80 | 1.139 | 1.681 |
| codestral_zero | 1000 | **79.3%** | 75.78 | 65.56 | 0.960 | 1.722 |
| deepseek-coder_zero (partial) | 693 | 48.6% | 55.39 | 58.07 | 0.887 | 1.578 |
| **Few-shot — train.jsonl demos** (Vivaldi+Corrette+Bach) | | | | | | |
| phi4_few | 1000 | 35.1% | 74.80 | 67.58 | 0.701 | 1.278 |
| qwen-coder_few | 1000 | 19.9% | 63.43 | 65.69 | 0.742 | 1.414 |
| codestral_few | 1000 | 45.2% | 67.55 | 59.18 | **0.696** | 1.407 |
| deepseek-coder_few (partial) | 666 | 26.3% | 57.04 | 60.23 | 0.714 | 1.428 |
| **Few-shot — simple demos** (A-min, 4-bar) *ablation* | | | | | | |
| phi4_few_simple | 1000 | **99.6%** | 71.13 | 69.50 | 1.874 | 2.683 |
| qwen-coder_few_simple | 1000 | **98.9%** | 63.09 | 55.93 | 1.980 | 2.796 |
| codestral_few_simple | 1000 | **97.1%** | **89.44** | **76.95** | 1.754 | 2.535 |
| deepseek-coder_few_simple | 1000 | **99.9%** | 53.44 | 46.96 | 1.960 | 2.773 |

`deepseek-coder` did not complete the full 1000 samples in either zero or few-shot regimes (12h wall TIMEOUT after ~666–693 generations); the row above is the partial subset.

### 1.3 Observations

1. **codestral wins compile rate zero-shot** (79.3%), but `phi4` is closest to the corpus by both JS-sim metrics (83.27 / 81.24). The bigger code-tuned model produces more confident syntax, the smaller general-purpose model produces music whose statistics sit nearer to the Baroque references.

2. **Realistic few-shot demos halve compile rate but halve FMD too.** Switching from train-derived demos (Vivaldi+Corrette+Bach) to hand-written A-minor stubs lifts compile to ≥97% but pushes FMD/test from ~0.7 to ~1.9 — the model learns to produce *narrow safe* outputs that pass parsing at the cost of distribution match. The simple-demo row is therefore *not* a representative score, it is a distribution-collapse benchmark.

3. **`codestral_few_simple` is the table-wide outlier.** Highest JS/test (89.44) and JS/muto (76.95) in the entire generation table, yet FMD/test = 1.754. The 22B code-tuned model under constrained demos collapses to a polyphony/groove distribution that happens to align with the corpus mean, while the generated `.ly` text remains lexically distant from training data.

4. **deepseek-coder is consistently the weakest at JS-sim across regimes** despite competitive FMD. The 16B MoE writes Baroque-flavored prose (low FMD) but produces MIDIs whose three muspy features drift from the corpus more than the other three models.

---

## 2. Music-understanding benchmark

Eight LilyPond-input tasks adapted from arXiv-2509.23350v1, executed zero-shot on the 4 surviving models. Each task uses the paper's prompt scaffolding (multiple-choice or structured-output template) with temperature = 0 / greedy decoding.

### 2.1 Tasks

| # | task | n | metric |
|---|---|---|---|
| 1 | `bar_count` | 100 | exact-match accuracy (integer output) |
| 2 | `metadata_qa` | 60 | 4-way MC accuracy (key / meter / note_length, visible) |
| 3 | `bar_sequencing` | 119 | penalised Kendall-τ ∈ [0, 1] |
| 4 | `next_bar_prediction` | 119 | 4-way MC accuracy |
| 5 | `metadata_prediction` | 60 | 4-way MC accuracy (queried field masked from input) |
| 6 | `music_captioning` | 60 | 4-way MC accuracy on the score's true title |
| 7 | `composer_recognition` | 96 | 4-way MC accuracy |
| 8 | `genre_recognition` | 132 | 4-way MC accuracy |

Dropped from the paper's task list because Mutopia has no labels:
- *Emotion recognition* — Russell's quadrants require EMOPIA / ADL-piano which we don't carry.
- *Error detection* — no annotated error catalogue exists for Mutopia scores.

### 2.2 Methodology differences vs ABC-Eval

- **Input format.** ABC-Eval feeds ABC notation; we feed raw LilyPond text. The paper's "note length" field (ABC's `L:1/N`) becomes the denominator of the `\time` declaration in our setup, since LilyPond does not carry an explicit unit-length field.
- **Title source.** Mutopia's `dataset_mutopia.json` has no `title` field; titles are extracted from `\header { title = "..." }` blocks in the score body. Pieces without parseable titles are excluded from the captioning task.
- **Distractor sampling.** Implemented from the paper's textual description (we did *not* copy from the anonymous reference repo). Same-pool distractors (composer, genre, title) draw 3 distinct labels from the full corpus value set; within-score distractors (next_bar) draw 3 random later bars from the same piece; within-field distractors (metadata_qa, metadata_prediction) draw 3 distinct values from the field's corpus value set.
- **Reproducibility.** Bench is generated with a fixed seed via [scripts/build_understanding_bench.py](scripts/build_understanding_bench.py); the same JSONL feeds every model so comparisons are byte-identical.

### 2.3 Results

_Pending — inference + eval jobs not yet submitted. Numbers land in [REPORT_understanding.md](REPORT_understanding.md) and a per-task accuracy table here once the sweep completes._

---

## 3. Files

- Generation per-cell JSON: `/nfsd/voce/machine_learning/experiments/lilybench/eval/<model>_<regime>/{summary,fmd_test,fmd_mutopia}.json` (mirrored locally at `data/eval/`)
- Generation rendered MIDI per sample: `<eval-cell>/midi/`
- Understanding bench (single file, byte-stable per seed): `data/understanding/bench.jsonl`
- Understanding predictions per (model, task): `data/understanding/predictions/<model>/<task>.jsonl`
- Understanding eval summaries: `data/understanding/eval/<model>/summary.json`
- LilyBERT embeddings reference: `fmd_refs/{test,mutopia}_lilybert_L6.npz`
- muspy reference aggregates: `js_refs/{bmdataset,mutopia}_muspy_agg.json`
- Mutopia corpus + manifest: `/nfsd/voce/machine_learning/datasets/mutopia/{dataset_mutopia.json,stripped/*.ly}`
- LoRA narrative archive (not paper-facing): [RESULTS_lora_archive.md](RESULTS_lora_archive.md)
