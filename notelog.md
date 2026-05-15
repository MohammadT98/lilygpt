# notelog.md — Design journal

This file is the running record of every design decision taken while building the LilyPond code-generation benchmark on top of `bmdataset`. Each entry states the decision, the rationale, the alternatives considered, and — when applicable — the empirical signal we expect to collect. It is written as paper-source material: future "why did we do X?" questions should be answerable from this file alone.

**Update discipline:** append or edit an entry whenever a design choice is taken, revised, or overturned by an experimental result. Do not silently change behavior — record the revision with a date and the new rationale.

---

## 1. Dataset source

### 1.1 Source = `data/bmdataset/preprocessed/` only

**Decision.** Train and evaluate exclusively on the 2,645 flat `.ly` files under `data/bmdataset/preprocessed/`.

**Rationale.** The preprocessed variant is already (a) recursively include-resolved (multi-file LilyPond projects concatenated into a single self-contained file per part/score) and (b) translated from Italian pitch notation (`do`, `re`, `mi`, …) to Nederlands (`c`, `d`, `e`, …), which is LilyPond's default language. Using it lets us skip re-implementing the `file_resolver` and Italian→Nederlands stages while still exposing the model to realistic LilyPond source.

**Rejected alternatives.** (i) `raw/` only — more layout diversity per piece (nested `ly/`, `midi/`, `pdf/` folders; Italian notation) but would force us to re-run include resolution and a language-mapping step, with no clear quality gain for our goal (code generation, not preprocessing robustness). (ii) Mixed raw + preprocessed — doubles volume and adds notation duplication that would complicate conditioning and evaluation.

**Source:** [data/bmdataset/README.md](data/bmdataset/README.md) documents the preprocessing steps; [manifest.json](data/bmdataset/manifest.json) enumerates the 391 pieces.

### 1.2 Normalization pipeline is dropped from the training path

**Decision.** Feed the raw bytes of each preprocessed `.ly` file to the training dataset. The prior normalization pipeline ([lilybench/normalize.py](lilybench/normalize.py)) is retained in-tree for ablation experiments but is no longer invoked by the default build.

**Rationale.** The project's goal is to benchmark LilyPond **code generation**. The prior pipeline stripped or rewrote large classes of real LilyPond constructs — `%` comments, `%{…}%` block comments, `\override`, `\markup`, `\mark`, dynamics, hairpins, `\quote`, `\transpose` blocks (resolved via `lilypond --displayLilyMusic`), `\repeat unfold N{…}` (unrolled), `\times` (rewritten to `\tuplet`), `\forma = {…}` (inlined and deleted), and all "empty" variable assignments. Training on this sanitized subset would teach the model a narrower syntax than what human LilyPond users actually write, degrading benchmark validity.

**Rejected alternatives.** (i) Keep the whole pipeline — would effectively measure generation of a LilyPond *dialect* rather than LilyPond itself. (ii) Keep only comment and whitespace stripping — negligible benefit, same principle violated.

**Ablation to run.** Retrain with the full legacy pipeline on the same splits; compare `lilypond --parse` success rate on generated samples. Expected: normalization-trained model shows lower lexical diversity and lower parse success on held-out idioms.

**Pipeline steps being dropped**, for the paper:

- Stage 0 `file_resolver` — include resolution, language mapping, forma splitting ([file_resolver.py](lilybench/stages/normalization/file_resolver.py)).
- Stage 1 `preprocess` — `%` / `%{…}%` comment stripping, whitespace normalization, inline-assignment spacing ([preprocess.py](lilybench/stages/normalization/preprocess.py)).
- Stage 2 `normalize_syntax` — music-function expansion, `\transpose` resolution, `\repeat unfold` expansion, tuplet rewriting, chord-bracket canonicalization ([normalize_syntax.py](lilybench/stages/normalization/normalize_syntax.py)).
- Stage 3 `forma` — prepend `\key`/`\time`/`\tempo`/`\partial` into variables; strip skips and layout hints from `forma=`; inline forma into `<< \forma … >>` sites; delete the `forma=` assignment ([forma.py](lilybench/stages/normalization/forma.py)).
- Stage 4 `engrave_strip` — ~70 regex cleanups removing `\override`, `\markup`, `\mark`, dynamics, hairpins, `\quote`, non-music variable assignments ([engrave_strip.py](lilybench/stages/normalization/engrave_strip.py)).
- Stages 5–6 `postprocessing` — orphan `\time` removal, malformed-`\tempo=` fixes, empty-voice collapse, empty-variable-assignment removal ([postprocessing.py](lilybench/stages/normalization/postprocessing.py)).

---

## 2. Training example granularity

### 2.1 Full-file examples

**Decision.** Each training example is a full `.ly` file (or a chunk thereof when it exceeds context).

**Rationale.** Code generation requires the model to learn file-level structure — how variables are declared at the top and cross-referenced in `<< \var … >>` simultaneous blocks, how `\score { … }` assembles parts, how forma blocks inject `\key`/`\time` into music variables. Per-assignment extraction (the prior approach, [build_assignment_dataset.py](lilybench/stages/dataset/build_assignment_dataset.py)) produces `name = { … }` completions in isolation and loses this structure.

**Rejected alternative.** Keep per-assignment granularity. Cheaper to train (small, uniform examples; no context pressure) and sidesteps the boilerplate problem (the prelude appears once per piece, not per assignment). Rejected because the downstream evaluation is code generation, not assignment completion.

**Trade-off accepted.** Long files exceed `max_length`; solved by §2.2 below.

### 2.2 Deterministic chunking at variable-assignment boundaries

**Decision.** When a file variant exceeds `max_length` tokens, split it at top-level `name = { … }` boundaries (same matcher as [build_assignment_dataset.py:12](lilybench/stages/dataset/build_assignment_dataset.py#L12)). Consecutive assignments are greedily packed into chunks ≤ `max_length`. A single variable whose body alone exceeds the limit is split at the nearest whitespace/barline inside its braces, with no loss-masking on the split.

**Rationale.** Three properties follow: (i) the entire dataset is seen in every epoch, (ii) mid-statement cuts — which would train the model on fragment grammar — are avoided, (iii) chunk IDs are deterministic, so experiments are reproducible.

**Rejected alternative.** Random-window cropping per epoch (pick a random start offset each `__getitem__` call). Its only real benefit — implicit augmentation across epochs — is already supplied by our K-variant augmentation (§4). Its costs are concrete: probabilistic coverage, tail under-sampling for long files, and loss of reproducible example IDs.

### 2.3 Prelude only on chunk #0; minimal header on chunks #1..N

**Decision.** Chunk #0 carries the full `\version` / `\header` / `\paper` / `\language` / `variabili.ly` prelude. Chunks #1..N are prefixed only by the original `\version` and `\language` lines — just enough to make each chunk a parseable LilyPond fragment on its own.

**Rationale.** Duplicating the ~80-line prelude across chunks would reintroduce the very boilerplate-memorization problem this plan is trying to mitigate (§3). The minimal two-line header preserves parseability without injecting repeatable boilerplate.

---

## 3. Anti-memorization strategy

**Problem statement.** Every preprocessed file begins with a near-identical ~80-line prelude: `\version "2.x"`, a `\header { … }` block with subtitle/composer macros, a `\paper { … }` block with margins and spacing, `\language "nederlands"`, and the `variabili.ly` articulation library (declarations of `su`, `giu`, `tr`, `tasto`, `solo`, `dolce`, `pad`, etc.). Under a full-file training objective this produces a strong fixed-prefix memorization prior: the model would learn to copy the prelude verbatim and get large gradient reward for doing so, without learning anything about the music body.

### 3.1 Prelude loss-masking

**Decision.** On chunks that contain the prelude (chunk #0 only), the labels over the prelude character range are set to `-100`, i.e. excluded from the loss. The model sees the prelude as context but is not trained to emit its tokens.

**Rationale.** The most direct fix: if we do not want the model to memorize the prelude as output, we stop rewarding it for doing so. This is a structural solution at the loss level, orthogonal to data augmentation. It also generalizes — any region we mark as "conditioning, not output" (the metadata block, §5) becomes free of memorization pressure.

### 3.2 K=4 augmented variants per file

**Decision.** For each base file, emit K=4 variants: `original`, `shuffle_prelude`, `drop_unused_prelude ∘ shuffle_prelude`, `inline_variables ∘ shuffle_prelude`. Each variant carries its RNG seed so regeneration is deterministic.

**Rationale.** A single fixed prelude becomes an empirical distribution over preludes. The model can no longer rely on an exact-match fixed prefix, and whatever capacity it would have spent on memorizing the prelude is redirected to the music body.

**Orthogonal priors each augmentation targets.**

| Augmentation | Prior it breaks |
|---|---|
| `shuffle_prelude` | "Variable X always appears before variable Y" |
| `drop_unused_prelude` | "Variable Z is always declared" |
| `inline_variables` | "`\varname` always refers to a named macro; bodies only appear in `varname = { … }` position" |

K=4 is the starting point. If the memorization metric (§7) stays high, raise to K ∈ {6, 8} or enable the optional `rename_variables` augmentation (§3.4).

### 3.3 Syntactic correctness gate (`lilypond --parse`)

**Decision.** Every augmentation output is passed through `lilypond --parse` (we're on LilyPond 2.24.4) in a subprocess. Variants that fail to parse are dropped and logged; the base file is never dropped because we always emit the unmodified original as variant #0.

**Rationale.** Augmentations that break the grammar would train the model on nonsense. The gate is paid once at build time (hundreds of milliseconds per file × ~10k variants ≈ 20–30 minutes on a workstation), not per-epoch.

### 3.4 Variable-renaming augmentation: off by default

**Decision.** `rename_variables` (rename prelude vars to random identifiers like `su → aux_7f2a` consistently within a file) is implemented but disabled in the default K=4 variant set.

**Rationale.** Renaming has the largest capacity to kill the fixed-name prior but also the largest cost: it produces many tokens that don't appear anywhere else, bloating the vocabulary utilization, and may confuse smaller models. Turn it on only if the empirical memorization metric (§7) remains high after K=4 + prelude masking.

---

## 4. Metadata conditioning

### 4.1 Structured `%% === METADATA ===` block on every chunk

**Decision.** Every chunk — including chunks #1..N — begins with a structured comment block:

```
%% === METADATA ===
%% composer: Charpentier
%% period: Late Baroque
%% musical_form: motet
%% ensemble: violin, viola, cello, flute
%% part: violino2
%% === END METADATA ===
```

**Rationale.** (i) The metadata is musicologist-assigned ground truth from [bmdataset/metadata.json](data/bmdataset/metadata.json), not engineered context, so using it at training time is honest supervision. (ii) It gives the model a conditioning signal for steerable generation at inference time: "generate a Charpentier-style motet for violino2" becomes a well-formed prompt. (iii) Placing the block on every chunk, not only chunk #0, mirrors inference-time prompting (where the user always starts with metadata) and ensures chunks #1..N also carry conditioning.

**Not a new boilerplate problem.** The metadata **varies per piece** (different composer, period, form, ensemble, part). Duplicating across chunks does not create a fixed memorizable prefix; it creates a conditioning distribution aligned 1:1 with the piece distribution.

### 4.2 Metadata is always loss-masked

**Decision.** The metadata block's character range is always included in `label_mask_char_ranges`, so the model computes loss only on the LilyPond body that follows.

**Rationale.** The model should **condition on** the metadata, not learn to **generate** it. If the metadata were in the loss, the model would learn to hallucinate composer names that happen to correlate with common musical patterns (e.g., "files that start with many quarter-note runs → likely Vivaldi"), which is the wrong direction of causation and degrades the conditioning interface.

### 4.3 Field-level dropout during variant generation

**Decision.** When building each variant, replace individual metadata values with `<unk>` (per-field probability `p_field=0.15`) and omit the entire block (probability `p_block=0.10`).

**Rationale.** Two goals. (i) **Mitigate imbalance.** The dataset is heavily unbalanced across composers (Vivaldi, Charpentier, Telemann are well-represented; many composers have 1–3 files). Without dropout, the model overfits majority labels and under-learns from rare composers. Field dropout decorrelates per-field labels from the output, forcing the model to rely on the LilyPond body when conditioning is weak. (ii) **Robustness to missing conditioning at inference.** Some downstream eval scenarios will not supply a full metadata block; the model must still generate plausibly.

### 4.4 Excluded fields: per-movement `key` / `tempo` / `time`

**Decision.** The metadata block excludes `movements.*.key`, `movements.*.tempo`, `movements.*.time` from `metadata.json`. Only global fields are encoded.

**Rationale.** These fields vary **within** a file at movement transitions. A concerto may have an Allegro in G-major, an Adagio in E-minor, and a Presto back in G-major. Pinning a single key/tempo/time in the file-level header would create systematic mismatches between the header claim and the `\key`/`\time` directives in the LilyPond body at section boundaries — a training signal that actively teaches the model to ignore its own conditioning. The LilyPond body already carries authoritative per-section directives, so the information is not lost.

### 4.5 `part:` is parsed from filename, not taken from metadata

**Decision.** The `part:` field is parsed from the filename suffix (`_violino2` from `…_violino2.ly`, `full` from `…_score.ly`). The `midi_instruments` field from metadata is encoded separately as `ensemble:`.

**Rationale.** `metadata.json.midi_instruments` is the full *ensemble* — the list of instruments the **piece** calls for. Each preprocessed file, however, is usually a single *part* file (one instrument's line, extracted from the multi-file LilyPond source). Encoding the ensemble list as the part would teach the model that a `violino2.ly` file contains viola, cello, and flute material. We give the model both signals with correct semantics: `ensemble:` tells it what piece this is from, `part:` tells it what role this file plays within the ensemble.

---

## 5. Infrastructure

### 5.1 Deprecate `build_assignment_dataset.py` and hardcoded `data/normalized_dataset`

**Decision.** The per-assignment builder and the entire `data/normalized_dataset` → `data/assignment_dataset` chain stop being invoked by the default pipeline. Source files remain in-tree for ablation experiments.

**Rationale.** The new pipeline has no normalization step, so `data/normalized_dataset` never exists on disk; feeding the assignment builder with raw preprocessed files would not make sense without resurrecting the normalization pipeline.

### 5.2 Parametrize `build_splits.py` input path

**Decision.** `build_splits.py` accepts `--input-jsonl` and `--output-dir` command-line arguments. The previous hardcoded `data/assignment_dataset/all_examples.jsonl` → `data/splits_full` is replaced by `data/fullfile_dataset/all_examples.jsonl` → `data/splits_full` by default.

**Rationale.** Hardcoded paths have broken twice in this pipeline (normalize CLI, assignment builder). Parametrization stops the bleeding.

---

## 6. JSONL schema

**Decision.** Each training example is one JSONL line with this schema:

```json
{
  "id": "charpentier_lauda_sion_violino2__v0__c0",
  "source_file": "NO_PUB__charpentier_lauda_sion_H_268_egredimini_H_280_violino2.ly",
  "variant": "original",
  "chunk_index": 0,
  "chunk_total": 3,
  "seed": 42,
  "full_text": "%% === METADATA ===\n…\n\\version \"2.18.0\"\n…",
  "metadata_char_range": [0, 187],
  "prelude_char_range": [187, 2670],
  "label_mask_char_ranges": [[0, 2670]]
}
```

**Rationale.** Char-based ranges (rather than token-based) survive tokenizer changes — if we swap the model and its tokenizer, the JSONL does not need to be regenerated. `label_mask_char_ranges` is a list so that future uses (e.g. masking a second region) do not require schema changes. `seed` + `variant` + `chunk_index` + `source_file` together uniquely determine `full_text`, so the whole dataset is reproducible from its JSONL.

---

## 7. Experiments to run (filled in as results come in)

### 7.1 Memorization baseline

**Metric.** Exact-prefix-match @ 128 tokens on the validation prelude. For each validation file, strip the body and ask the model to generate the prelude; measure the longest prefix of the generated prelude that exactly matches the ground-truth prelude.

**Hypothesis.** Without prelude masking or augmentation, this metric is ≫ 50% (the prelude is near-identical across files). With prelude masking + K=4, it should be ≪ 10%.

*Result: to be filled.*

### 7.2 Augmentation ablation

**Grid.** K ∈ {1, 2, 4, 8} × {prelude-masking on/off}.

**Metric.** (i) §7.1 memorization metric. (ii) `lilypond --parse` success rate on 1,000 unconditional samples. (iii) Held-out validation loss (body tokens only, by construction).

*Result: to be filled.*

### 7.3 Conditioning ablation

**Grid.** {metadata block present / absent during training} × {metadata provided / withheld at inference}.

**Metric.** Controllability — for each (composer, period, form) triple in the test set, generate N samples conditioned on that triple and measure how often a composer-classifier (trained separately on the body-only text) assigns the sample to the target composer.

*Result: to be filled.*

### 7.4 Normalization ablation

**Grid.** {new pipeline on preprocessed} vs {legacy normalization pipeline on preprocessed}.

**Metric.** `lilypond --parse` success rate on 1,000 unconditional samples; coverage of out-of-normalized-subset constructs (e.g., `\markup`, `\override`) in generated samples.

*Result: to be filled.*

---

## 8. Empirical findings (build-time, pre-training)

### 8.1 `lilypond --parse` initially failed on ~97% of raw `bmdataset/preprocessed/` files — root cause was mixed-language pitch notation; fixed in-place with `python-ly`

**Finding.** On a random sample of 30 files from `data/bmdataset/preprocessed/`, only 1/30 passed `lilypond --parse` under stock LilyPond. The most common failure was mixed-language pitch notation — e.g., `re4 la8 f' fa4(mi8) cis\staccatissimo` in the same bar mixes italian (`re`, `la`, `fa`, `mi`) with nederlands (`cis`, `f'`). Most files carried a `\language "nederlands"` declaration while their bodies actually contained italian notes — a discrepancy inherited from the upstream preprocessing.

**Fix.** Added `python-ly` as a dependency and ran [scripts/translate_preprocessed_to_nederlands.py](scripts/translate_preprocessed_to_nederlands.py) in-place on the full corpus. The script force-rewrites the `\language` directive to `italiano`, runs `ly.pitch.translate.translate(..., "nederlands")`, and writes the result back. python-ly's italiano pitch reader only matches italian note names (`do`, `re`, …) — single-letter nederlands notes (`a`, `b`, `c`, `cis`, `bes`, …) pass through unchanged because they aren't italian note tokens — so the translate call is safe on mixed-notation files and idempotent on already-nederlands files. Result: 2,620 / 2,645 files modified, 0 script errors.

**Impact.** Parse rate on the same 30-file sample jumped from 1/30 to 28/30. The 2 remaining failures are undefined-variable references (e.g., `\IXflII` — forma block that was renamed or deleted upstream) — orthogonal to pitch language. The dataset was rebuilt from the corrected corpus.

**What this means for the paper.** The training set is now ~93%+ `lilypond --parse`-valid. `lilypond --parse` still isn't used as a per-variant build gate (cost + residual unparseable tail), but generated samples at inference time will be scored on parse success, and the training distribution now matches the evaluation distribution much more closely.

### 8.2 `check_brace_balance` initially checked paren balance too — 284/2,645 false positives from Scheme

**Finding.** An earlier version of `check_brace_balance` tracked `(` / `)` in addition to `{` / `}`. On the full corpus it flagged 284 files. All flagged files contained legitimate Scheme blocks (`#(...)`) with constructs the character-level scanner cannot model without a full Scheme reader: quoted lists `'(a b c)`, character literals `#\(`, and embedded parens inside Scheme strings.

**Decision.** The paren check is removed. The validator now checks only brace balance, string termination, and block-comment closure. Paren correctness is delegated to `lilypond_parse_ok` (the heavyweight subprocess gate) when it is ever used. This dropped the false-positive rate from ~10.7% to 1/2,645 ≈ 0.04% — the remaining file has a genuine extra `}` on line 896 and is legitimately broken source data.

---

## 9. Revision history

- **2026-04-18** — Initial draft. All decisions above taken during design planning.
- **2026-04-18** — Implementation pass. Added §8 with the two build-time empirical findings: (a) raw bmdataset parse rate ≈3% under stock LilyPond, so `lilypond --parse` is unusable as a per-variant gate; (b) Scheme blocks force the fast validator to drop paren balance checking. Final production build emitted 92,237 chunks from 2,644 files with 0 augmentation dropouts.
- **2026-04-18** — Pitch-language fix. Added `python-ly` dependency and ran `scripts/translate_preprocessed_to_nederlands.py` in-place on `data/bmdataset/preprocessed/`: 2,620/2,645 files rewritten, italian note names translated to nederlands, parse rate on a 30-file sample 1/30 → 28/30. §8.1 updated to describe the fix. Dataset rebuilt from the corrected corpus.
- **2026-04-19** — Physically removed the legacy normalization pipeline (`lilybench/normalize.py`, `lilybench/stages/normalization/`, `lilybench/stages/dataset/build_assignment_dataset.py`, `lilybench/utils/options.py`, the `lilybench normalize` CLI subcommand, and the `--mask-input` flag on `train_lora`/`eval_lora`). The §1.2 / §5.1 decisions to stop invoking this code stand; this revision eliminates the dead source. The normalization ablation (§7.4) is now sourced from git history rather than in-tree files. Also: ported the `tests/` directory from standalone scripts to a pytest suite (`pytest` + `pytest-xdist` added under `[dependency-groups] dev`); added unit tests for `prelude`, `augmentations`, `metadata_header`, `build_fullfile_dataset`, and `build_splits`.
- **2026-04-19** — Package layout moved from src-layout (`src/lilybench/…`) to flat-layout (`lilybench/…` at the repo root). `[tool.setuptools.packages.find]` switched from `where = ["src"]` to `include = ["lilybench*"]`; `pythonpath = ["src"]` dropped from `[tool.pytest.ini_options]` (the editable install makes the package importable without it). All docs updated in lock-step; no behavioural or import-path changes inside the package.
- **2026-04-19** — Dropped the planned `<KEY:…>` / `<TIME:…>` / `<TEMPO:…>` / `<VOICE>` structural tokens. All models now consume and generate **raw LilyPond text only**; no tokenizer vocabulary is extended with special tokens. Key/time/tempo/voice signals are carried by the native LilyPond directives (`\key`, `\time`, `\tempo`, `\new Voice …`) already present in the source. The `lilybench/` source was already structural-token-free, but the `slurm/{train,infer,test}/` directory carried 12 exp10/11/12 voice-token wrappers — `infer_exp10_voice_tokens.slurm` actively prepended `<VOICE>` to the inference prompt. All 12 files (`train_exp1{0,1,2}_*`, `infer_exp1{0,1,2}_*`, `infer_exp10_two_voice_*`, `infer_exp10_command_text_prompt`, `eval_exp1{0,1,2}_test`) were removed; `slurm/train/train_multimodel.slurm` and `slurm/infer/infer_multimodel.slurm` remain as the canonical entry points. CLAUDE.md §"Model registry", README.md §"Training" and §"Run order" updated to match.
- **2026-04-19** — Flattened `lilybench/stages/*` to semantic top-level modules `lilybench/preprocess/`, `lilybench/data/`, `lilybench/evaluate/`. Added Hydra entry points `lilybench/train.py` and `lilybench/infer.py` (three regimes `zero`/`few`/`lora` in one module) plus three evaluation entry points under `lilybench/evaluate/{loss,text_midi,fmd}.py`. Introduced a `configs/` tree at the repo root with per-entry-point yaml, a `model/` group (one file per registered id, each a thin `id: <name>` pointer — the Python registry remains the single source of truth), a `regime/` group (zero/few/lora), and `hydra/launcher/{slurm_train,slurm_infer}.yaml` for `hydra-submitit-launcher`. `hydra-core>=1.3` and `hydra-submitit-launcher>=1.2` added to `[project.dependencies]`. `scripts/eval_extracted_ly.py`, `scripts/eval_fmd.py`, `scripts/extract_detokenized.py` moved under `lilybench/evaluate/`. The `lilybench build-dataset` / `build-splits` argparse CLI is preserved — one-shot preprocessing jobs don't benefit from Hydra sweeps. SLURM templates in `slurm/{train,infer}/` rewritten as thin wrappers around the Hydra entry points; the existing env-var contract (`MODEL_ID`, `REGIME`, `TRAIN_JSONL`, ...) is kept for backwards compatibility with external scripts. Clean break — no back-compat shims in the Python package. Tests rewired to the new import paths; four import-smoke tests and a Hydra config-compose smoke test added. No behavioural changes to dataset building, loss masking, augmentations, or the training loop.
- **2026-04-19** — Added `configs/hf.yaml` Hydra group controlling Hugging Face cache/auth: `hf.home` (exported as `HF_HOME`, with `HUGGINGFACE_HUB_CACHE`/`TRANSFORMERS_CACHE`/`HF_DATASETS_CACHE` defaulting to subdirs thereof unless overridden), `hf.token` (`HF_TOKEN`, needed for gated Codestral), `hf.offline` (`HF_HUB_OFFLINE` + `TRANSFORMERS_OFFLINE`). Applied by `lilybench/hf_cache.py::apply_hf_env`, called at the top of each of the four entry points that load HF weights (`train`, `infer`, `evaluate/loss`, `evaluate/fmd`); `evaluate/text_midi` is excluded (pure music21/lilypond, no `from_pretrained`). All fields nullable and applied with `os.environ.setdefault`, so existing SLURM-template `export HF_HOME=$SCRATCH/huggingface` lines remain authoritative — Hydra only wins when the shell has not already exported the variable. No behaviour change to downloads themselves; this is purely config plumbing.
- **2026-04-19** — Added `muspy>=0.5.0` as a runtime dep and integrated 12 of its standard symbolic-music metrics into the per-sample MIDI evaluation record under a `muspy_*` namespace: `pitch_range`, `n_pitches_used`, `n_pitch_classes_used`, `pitch_entropy`, `pitch_class_entropy`, `scale_consistency`, `pitch_in_scale_rate` (skipped to `None` when no key signature), `polyphony`, `polyphony_rate`, `empty_beat_rate`, `empty_measure_rate`, `groove_consistency`. Drum metrics (`drum_in_pattern_rate`, `drum_pattern_consistency`) excluded — corpus is solo / small ensemble (§1.1). Implementation lives in `lilybench/evaluate/muspy_metrics.py`, called from `eval_midi` in `lilybench/evaluate/text_midi.py` immediately after the existing music21 analysis; the music21 metrics (drift detection, contour analysis, tonal closure, step-vs-leap) are **retained**, not replaced — muspy is purely additive so the paper can cite both standard and domain-specific framings of the same MIDI. `_build_summary` extended with mean aggregates for each new key; `tooling` block now also reports `muspy_version` for reproducibility. The MIDI hand-off is the existing `lilypond \midi {}` output, so no LilyPond → muspy converter (e.g. `python-ly`) is needed.
- **2026-04-19** — P0+P1 review pass across `infer.py`, `train.py`, `evaluate/{loss,fmd,text_midi}.py`. Six substantive fixes, all paper-correctness-critical:
  1. **Inference prompt unification (Option C hybrid).** `regime=lora` previously primed with `\relative do'' {\n` — Italian pitch notation that §8.1 translated *out* of the training distribution — and also bypassed the chat template used by `zero`/`few`, making the three regimes a three-way prompt mismatch. Replaced with `_build_lora_preamble()` returning a raw preamble (`%% === METADATA ===` block + `\version` + `\language`) that mirrors the training distribution exactly; zero/few keep `apply_chat_template` (field-standard for instruction-tuned baselines, cf. HumanEval / BigCode-Eval). Missing `chat_template` on a registry model is now a hard `RuntimeError` at prompt-build time, not a silent string-concat fallback. LoRA output is re-joined with its preamble after generation (`raw_text = f"{prompt}{raw_text}"`) so the saved `.ly` is compilable.
  2. **FMD reference-loader field fix.** `evaluate/fmd.py` was concatenating `rec["input"]` + `rec["output"]` — legacy keys from the removed per-assignment dataset. The full-file JSONL carries only `full_text` (§6). With `reference_kind=test`, this silently produced an all-empty reference, either aborting or computing FMD against a degenerate distribution. Any "in-domain FMD" number from before this fix is invalid. Fixed to read `rec.get("full_text", "")`.
  3. **FORBIDDEN_PATTERNS removal.** `evaluate/text_midi.py` was flagging `\repeat`, `\tuplet`, `~` (ties), `<< … >>`, chords `<c e g>`, `\grace/\acciaccatura/\appoggiatura`, `s` skips, `\score`, `\layout` as violations — but §1.2 explicitly kept all of these in the training distribution by dropping the normalization pipeline, so the eval was penalising correct behaviour. Dict and every call site (`allowed_forbidden` CLI/config arg, `no_forbidden` + `forbidden_hits` return fields, `_parse_allowed_forbidden`, orphan `ListConfig` import, `allowed_forbidden: null` in `configs/evaluate/text_midi.yaml`) removed. Tonal, drift, key/time, contour, and muspy blocks untouched.
  4. **Training `save_strategy="steps"`.** Was `"no"`; `resume_from_checkpoint` was dead code. Changed to `save_strategy="steps"` with `save_total_limit=2`, aligned with the existing `save_steps` config knob so SLURM pre-emption on long Codestral runs no longer loses the entire job.
  5. **Loss-eval adapter compatibility check + tokenizer source fix.** `evaluate/loss.py` now reads `adapter_config.json` from the LoRA path and aborts with a clear error when `base_model_name_or_path != spec.hf_id` — previously a typo like `model=phi4 lora_path=…/qwen_lora/final` would `PeftModel.from_pretrained` with silently-dropped layers and print plausible-looking numbers. Tokenizer is now loaded from `spec.hf_id` (same as `infer.py` zero/few), not from the adapter dir with `local_files_only=True`; decouples eval from the adapter save format. Also: explicit startup log `[eval_loss] loss computed on body tokens only (metadata+prelude char ranges masked to -100, matching training)` so the reported metric is unambiguous.
  6. **`no_repeat_ngram_size=3` removed from `model.generate`.** LilyPond is intrinsically repetitive (`c4 c4 c4`, `g8 g8 g8 g8`); a 3-gram repeat penalty at decode time was suppressing idiomatic patterns the training data is full of. Dropped the kwarg; also hardened `_wrap_score` to only prepend missing `\version`/`\language` headers rather than forcing partial generations into a `\score { … }` block (which can turn broken output into differently-broken output). Decode switched to `skip_special_tokens=True` with `generated_ids = out[0][prompt_len:]` slicing so prompt echo is stripped cleanly regardless of tokenizer.

  P2 hardening items (dead `ModelSpec` fields `max_seq_len` / `gated`; empty-batch guard in `collate_standard_batch`; banner-width regex fragility in `extract_detokenized.py`; `translate_preprocessed_to_nederlands.py` backup flag; stale `mxlGPT/` directory) deferred to a separate pass after the first end-to-end run. Full test suite (77 tests) green post-fix.
- **2026-04-24** — Ported the **JS Divergence Similarity** metric verbatim from mxlGPT (`mxlGPT/src/evaluation/generation/muspy_eval.py:209-288`) into `lilybench/evaluate/js_similarity.py`. Three muspy metrics (`polyphony_rate`, `groove_consistency`, `scale_consistency`) per sample are aggregated to `{mean, std, n}`, each pair (model, reference) is approximated as a Gaussian, JS divergence is computed numerically over 2000 evenly-spaced points covering both 5σ supports, and the metric reports `100 * exp(-2 * mean_JS)` (higher = closer to reference; floor ≈ 25 for fully disjoint distributions). Wired into `lilybench/evaluate/text_midi.py` via two new optional config keys (`reference_midi_dir`, `reference_aggregate_path`) following the same load-or-compute cache pattern as `evaluate/fmd.py::reference_embeddings_path`. When a reference resolves, `summary.json` gains a `js_divergence_similarity` field at both the `all` and per-`by_group` levels; when neither key is set, behaviour is byte-identical to before (null fields are dropped from the summary). `slurm/evaluate/eval_text_midi.slurm` accepts the two new env vars (`REFERENCE_MIDI_DIR`, `REFERENCE_AGGREGATE_PATH`); `scripts/build_report.py` now appends a `JS-sim` column to the muspy table. Why: lets the paper compare each (model, regime) against the held-out test split *and* against Mutopia (out-of-domain) using the same single-number Gaussian-JS approximation as mxlGPT, side-by-side with the LilyBERT-FMD column. Eleven new unit tests in `tests/test_js_similarity.py` cover the math (identical → 100, fully-disjoint → ≈25, monotone in distance) and the I/O round-trip (load JSON cache, compute from synthesized MIDIs, save to cache). Full suite: 103 passed.
- **2026-05-15** — **Pivot.** The article framing narrows to out-of-box benchmarking on 4 models (phi4, qwen-coder, codestral, deepseek-coder); LoRA fine-tuning and Gemma-2-27B leave the paper. Training code (`lilybench/train/`, `lilybench/evaluate/loss.py`, `slurm/train/*`) stays in the repo so a future revision can revive the LoRA story; the LoRA-specific commentary in `RESULTS.md` moves to `RESULTS_lora_archive.md`. *In* the paper now: a new music-understanding benchmark suite adapted from arXiv-2509.23350v1 ("ABC-Eval"), implemented under `lilybench/understanding/` (eight tasks on Mutopia, prompt templates copied verbatim from the paper, scoring reimplemented from textual descriptions — we did *not* fetch the anonymous reference repo). Two paper tasks are dropped because Mutopia carries no labels: emotion recognition (no Russell-quadrant labels — paper used EMOPIA / ADL-piano) and error detection (no annotated error catalogue exists for Mutopia scores). Two adaptations to the paper's spec: (a) input format is raw LilyPond text rather than ABC notation; (b) ABC's "note length" (`L:1/N` field) becomes the denominator of `\time` since LilyPond has no explicit unit-length field. Titles for music captioning come from parsing `\header { title = "..." }` blocks in the score body — `dataset_mutopia.json` has no title field. New files: `lilybench/understanding/{__init__,tasks,dataset_builder,title_parser,bar_utils,score_metadata,scoring}.py`, `lilybench/infer_understanding.py`, `lilybench/evaluate/understanding.py`, `configs/infer_understanding.yaml`, `configs/evaluate/understanding.yaml`, `scripts/build_understanding_bench.py`, `slurm/{infer,evaluate}/{infer,eval}_understanding.slurm`. `scripts/build_report.py` extended with `--understanding-root` for a per-task × model matrix. Inference is zero-shot only (temperature=0 / greedy, max_new_tokens=20, paper convention); a separate entry point keeps the existing infer.py's LilyPond-generation-shaped post-processing from corrupting single-digit answers. Bench is byte-stable per seed so 4 models see identical questions. TDD: five new test files (`test_understanding_*`) cover title extraction, bar splitting / counting (with `\bar "||"` and quoted-string `|` edge cases), Kendall-τ scoring (perfect / reverse / partial / duplicate / out-of-range / too-long / unparseable), distractor sampling (4-way MC, gold excluded, deterministic under seed), prompt template formatting (paper-verbatim strings), bench reproducibility (same seed ⇒ byte-identical JSONL), and evaluator end-to-end (MC accuracy, bar_count accuracy, bar_sequencing penalised Kendall-τ, macro vs weighted aggregation).
