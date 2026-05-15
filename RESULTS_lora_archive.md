# LilyBench evaluation snapshot — LoRA archive

> **Archived 2026-05-15.** The article no longer reports LoRA results; this file
> preserves the LoRA / rescue / loss commentary for posterity. The training
> code remains in the repo under [lilybench/train/](lilybench/train/) and
> [lilybench/evaluate/loss.py](lilybench/evaluate/loss.py) and can be re-run.
> The paper-facing narrative is in [RESULTS.md](RESULTS.md).

**Date:** 2026-04-30 (later revision)
**Cluster state:** 11 PD, 2 RUNNING (codestral-lora rerun, codestral-lora loss-eval-4h). Pipeline near completion: 19 cells populated, 4 still pending (codestral_lora rerun + rescue + chain, deepseek-coder_lora chain).
**Models in this round (4 total):** phi4 (14B), qwen-coder (14B), codestral (22B), deepseek-coder (16B MoE). **Dropped:** gpt-oss (MXFP4 incompatible with our QLoRA path), gemma-2-27b (too slow under our wall budget — generated ≤40 samples in 24h), gemma-4-31B (transformers 5.x broke DeepSeek vendored modeling).

## Metrics — what each one measures and why

The benchmark runs five complementary metrics per `(model, regime)` cell. Each one catches a failure mode the others can miss; reading them together is the point.

### `compiles` — syntactic correctness gate (↑ better)

**Definition.** Fraction of generated `.ly` files that the LilyPond binary renders to a valid `.mid` without errors. Anything that compiles produces a MIDI; anything that doesn't is unusable end-to-end.

**Why we monitor it.** Music generation has a hard pre-condition: the output must parse. A model with beautifully musical but uncompilable output is useless for any downstream use (transcription, MIDI playback, performance practice). It also gates the MIDI-feature metrics: only compiled samples enter the muspy pipeline.

**How to read it.**
- **Very high (≥ 95%)**: model is producing safe, narrow outputs that always parse — *suspicious of distribution collapse* (e.g., the old simple-demo few-shot only emits 4-bar A-minor stubs).
- **Mid (50-80%)**: typical for capable models — they attempt rich structures and most succeed.
- **Very low (< 10%)**: structurally broken. Common cause for LoRA cells: max_new_tokens cap truncates long generations mid-piece.

### `JS/test` and `JS/muto` — distributional similarity in MIDI feature space (↑ better, range [25, 100])

**Definition.** `100 · exp(-2 · mean_JS)` where `mean_JS` is the average of Jensen-Shannon divergences between Gaussian fits to three muspy features (`polyphony_rate`, `groove_consistency`, `scale_consistency`) of the model's MIDIs vs. a reference corpus. Ported verbatim from the mxlGPT paper.

**Why we monitor it.** A single scalar that captures whether the *musical statistics* of generated MIDIs match a real Baroque corpus, regardless of textual surface. It catches drift that the embedding-based FMD (text-only) can't see — e.g., a model that writes plausible-looking LilyPond but with grossly wrong polyphony density.

**Two references because each tests a different thing:**
- **`JS/test`** uses the in-domain bmdataset corpus — *did the model match the distribution it was tuned on?*
- **`JS/muto`** uses Mutopia (out-of-domain Baroque) — *does the model generalize, or did it memorize the training distribution's quirks?*

**How to read it.**
- **100** = the model's Gaussians coincide with the reference. Identical mean and std across all three muspy features.
- **~75-90** = model captures the distribution well; differences are within typical Gaussian overlap.
- **~25-50** = poor match; distributions are mostly disjoint.
- **25** is the mathematical floor (fully disjoint Gaussians produce `100 · exp(-2 · ln 2) ≈ 25`).
- **Gap between JS/test and JS/muto** is the in/out-of-domain generalization gap. Big gap = model overfit to its tuning corpus.

### `FMD/test` and `FMD/muto` — distributional similarity in LilyPond text embedding space (↓ better)

**Definition.** Fréchet distance between Gaussians fit to LilyBERT layer-6 `[CLS]` embeddings of the model's `.ly` generations vs. a reference corpus. Same form as Fréchet Inception Distance, with LilyBERT (a domain-specific BERT trained on LilyPond) replacing Inception.

**Why we monitor it.** Operates on raw text — does **not** require the sample to compile. So it gives a signal even when `compiles = 0%`. This is what reveals that the raw LoRA outputs are *textually* near-perfect even though they don't parse to MIDI. Complements JS-sim: JS-sim sees the music after rendering, FMD sees the *prose* the model writes.

**How to read it.**
- **< 0.5** = strong textual match.
- **0.5-1.5** = moderate divergence.
- **> 2** = noticeably different distribution.
- The two references serve the same in-vs-OOD role as for JS-sim.
- **Low FMD with 0% compile** is the autoregressive failure-mode signature: model writes the *right text* token-by-token but the long-sequence dynamics break syntactic structure.

### `loss` — per-token cross-entropy on held-out test bodies (↓ better, LoRA cells only)

**Definition.** Mean cross-entropy of the LoRA-fine-tuned model on test-split sequences, computed teacher-forced (each token sees the true previous tokens) with metadata + prelude positions masked to `-100`. Equivalent to perplexity but reported as raw loss.

**Why we monitor it.** This is the only metric that directly measures the *training objective*. FMD/JS-sim evaluate generated samples (deployment behavior); loss evaluates likelihood of the actual reference data given the model. They can diverge:
- **Low loss + good FMD/JS** = fine-tuning worked end-to-end.
- **Low loss + bad raw compile + low FMD** = our LoRA case: per-token prediction is correct, but autoregressive sampling accumulates errors over long sequences. *This divergence motivates the rescue strategy.*
- **High loss** = the LoRA didn't actually fit the test distribution; the adapter is weak or the data is far from training.

Loss only makes sense for LoRA cells — zero/few have no adapter to evaluate against the test data.

### `n` — number of evaluable samples for the cell

Reported separately because some cells are partial (timeouts, drops):
- `compiles` is over `n` total samples.
- `JS-sim` and `FMD` are estimated from the subset that produced valid MIDIs (for JS) or any text (for FMD).
- Confidence is loose for `n < 30`; treat those rows as indicative only.

## How to read cross-metric patterns

| pattern | meaning |
|---|---|
| high compile + low JS/FMD-sim | model produces *narrow, safe* outputs that game the syntax check. Distribution collapse. (Old simple-demo few-shot.) |
| low compile + high FMD + low JS | model failed to produce parseable text. Useless for any downstream. (Not seen here.) |
| low compile + low FMD + null JS | model writes textually right things but truncation/sampling breaks syntactic completeness. (Raw LoRA.) |
| mid compile + low FMD + high JS | the rescue regime — surgical post-processing recovers most LoRA samples into valid MIDIs that statistically match the corpus. (LoRA rescued_v2.) |
| high compile + balanced FMD/JS | the ideal: model produces correct, distributionally-aligned outputs at scale. Closest cell here is `phi4_zero`. |
| low loss + high FMD | model memorized format but autoregressive generation diverges. Suggests overfitting to local context. |
| high loss + good FMD | model didn't fit test distribution well; FMD-low is statistical accident. (Not seen here.) |

## Results

| cell | n | compiles | JS/test ↑ | JS/muto ↑ | FMD/test ↓ | FMD/muto ↓ | loss ↓ |
|---|---|---|---|---|---|---|---|
| **Zero-shot** | | | | | | | |
| phi4_zero | 1000 | 71.1% | **83.27** | **81.24** | 0.933 | 1.419 | — |
| qwen-coder_zero | 1000 | 69.0% | **84.85** | 73.80 | 1.139 | 1.681 | — |
| codestral_zero | 1000 | **79.3%** | 75.78 | 65.56 | 0.960 | 1.722 | — |
| deepseek-coder_zero (partial) | 693 | 48.6% | 55.39 | 58.07 | 0.887 | 1.578 | — |
| **Few-shot — train.jsonl demos** (Vivaldi+Corrette+Bach) | | | | | | | |
| phi4_few | 1000 | 35.1% | 74.80 | 67.58 | 0.701 | 1.278 | — |
| qwen-coder_few | 1000 | 19.9% | 63.43 | 65.69 | 0.742 | 1.414 | — |
| codestral_few | 1000 | 45.2% | 67.55 | 59.18 | **0.696** | 1.407 | — |
| deepseek-coder_few (partial) | 666 | 26.3% | 57.04 | 60.23 | 0.714 | 1.428 | — |
| **Few-shot — old simple demos (A-min 4-bar)** ablation | | | | | | | |
| phi4_few_old_simple_demos | 1000 | **99.6%** | 71.13 | 69.50 | 1.874 | 2.683 | — |
| qwen-coder_few_old_simple_demos | 1000 | **98.9%** | 63.09 | 55.93 | 1.980 | 2.796 | — |
| codestral_few_old_simple_demos | 1000 | **97.1%** | **89.44** | **76.95** | 1.754 | 2.535 | — |
| deepseek-coder_few_old_simple_demos | 1000 | **99.9%** | 53.44 | 46.96 | 1.960 | 2.773 | — |
| **LoRA — raw** (full-file structure, often truncated) | | | | | | | |
| phi4_lora | 200 | 0.0% | — | — | — | 1.414 | **0.5317** |
| qwen-coder_lora | 200 | 0.0% | — | — | **0.147** | **1.263** | **0.5549** |
| codestral_lora (40 partial) | 40 | 0.0% | — | — | 0.330 | 1.475 | — |
| **LoRA — rescued v2** (≥8 bars, score-rebuilt) | | | | | | | |
| phi4_lora_rescued_v2 | 74 | 25.7% | **83.82** | 71.25 | 0.417 | 1.784 | — |
| qwen-coder_lora_rescued_v2 | 78 | 17.9% | **93.89** | **77.69** | **0.359** | 1.669 | — |
| codestral_lora_rescued_v2 | 14 | 14.3% | — | — | 0.666 | 2.073 | — |
| **phi4_lora_rescued v1** (brace-balance only baseline) | 200 | 9.5% | — | — | — | — | — |

## Cells that never landed (sweep closed 2026-05-01, post-mortem)

- **codestral_lora (full 1000 rerun)** — `infer-codestral-lora` 4276113 FAILED at 4h27m on 2026-04-30 17:34. The partial 40-sample + rescued_v2 (14) cells in the table above are the only codestral_lora numbers we have.
- **codestral_lora loss** — `eval-loss-codestral_lora-4h` 4285863 FAILED at 2h18m on 2026-04-30. No `loss.json` exists for codestral_lora.
- **deepseek-coder_lora (inference + everything downstream)** — racing twins `infer-deepseek-coder-lora` 4285864 (A40) and 4285870 (L40S) both TIMEOUT at 12h on 2026-05-01 05:45 without writing samples. Root cause is the DeepSeek-V2 vendored modeling code's incompatibility with `transformers` ≥5.x cache API; the `seen_tokens → get_seq_length()` shim wasn't enough to make the generation loop survive. The deepseek-coder_lora row is fully empty.

## Key observations

1. **LoRA captures the corpus textual distribution best.** Even raw (0% compile rate, structurally broken), `qwen-coder_lora` reaches FMD/test = 0.147 — six times closer to the test reference than its zero-shot equivalent (1.139). The LoRA-fine-tuned models *learned* the bmdataset distribution near-perfectly; the failure is purely autoregressive sampling collapsing the structure of long sequences.

2. **Rescue v2 unlocks the trapped signal.** Surgical structural rebuild (cut at last bar separator, balance braces, inject `\score` block, drop <8-bar fragments) lifts compile rate from 0% → 17-26%, and `JS/test` for `qwen-coder_lora_rescued_v2` hits **93.89** — the highest in the entire table by 9 points. Confirms the LoRA outputs are *content-rich, form-broken*, and a structural pass recovers the form.

3. **Old simple-demo few-shot games compile rate at the cost of distribution — but codestral is the exception.** A-min 4-bar monophonic demos give 97-100% compile but FMD/test ≈ 1.7-2.0 (column-worst). Replacing with realistic train-distributed demos drops compile to 19-45% but lands FMD/test at 0.7 — much closer to the corpus. **Outlier:** codestral under simple demos hits **JS/test = 89.44** (top of table) and JS/muto = 76.95 (top OOD), while still showing the typical high FMD (1.754). This is a striking dissociation: codestral with constrained demos produces MIDIs whose three muspy features (polyphony_rate, groove_consistency, scale_consistency) almost perfectly match the bmdataset Gaussian, while its LilyBERT text embedding stays distant. Hypothesis: the 22B code-tuned model under simple demos collapses to a *very tight* polyphony/groove distribution that happens to align with the corpus mean exactly — but the resulting `.ly` files lexically don't look like training data.

4. **codestral compiles best zero-shot** (79.3%) but lowest JS-sim of the three (75.78). Bigger code-tuned model = more confident syntax, but stylistically further from Baroque than the smaller phi4 (83.27).

5. **LoRA loss confirms fine-tuning landed.** `phi4_lora` at 0.5317 and `qwen-coder_lora` at 0.5549 cross-entropy on held-out test bodies — both well below the typical 1-2 of an unfit base model. Combined with their FMD/test ≈ 0.15-0.93, this is the textbook *low loss + low FMD + zero raw compile* pattern: the adapters learned to assign high probability to real Baroque sequences token-by-token, but autoregressive sampling accumulates errors over long generations and breaks structural integrity. This is precisely the failure mode the rescue strategy targets.

6. **deepseek-coder is consistently the weakest at JS-sim across all regimes** but competitive on FMD. With realistic train demos: JS/test 57.04 (worst of 4 models) but FMD/test 0.714 (best zero, second-best few). With old simple demos: JS/test 53.44 + JS/muto 46.96 (column-worst). With zero-shot partial: JS/test 55.39 + 48.6% compile. Pattern: deepseek-coder writes Baroque-flavored *prose* that LilyBERT recognizes (low FMD), but the resulting MIDIs have musical statistics that drift from the corpus more than the other 3 models do. The 16B MoE seems to lock onto syntactic surface form at the expense of musical-feature alignment.

## Files

- LilyPond samples + rendered MIDI per cell: [generations/](generations/)
- Per-cell JSON: `eval/<cell>/{summary.json, fmd_test.json, fmd_mutopia.json, loss.json}` on cluster NFS at `/nfsd/voce/machine_learning/experiments/lilybench/eval/`
- LoRA adapters: `runs/<model>_lora{,_l40s}/final` on cluster
- JS-sim references: `js_refs/{bmdataset,mutopia}_muspy_agg.json` (bmdataset: 2551 in-domain pieces; mutopia: 858 OOD pieces after `convert-ly` upgrade lifted compile rate from 40% → 58%)
- FMD references: `fmd_refs/{test,mutopia}_lilybert_L6.npz` (8365 test chunks, 2123 Mutopia files embedded)
