# status.md — first full eval sweep

**Last updated:** 2026-05-15 (Phase 5 sweep closed — all SLURM jobs settled by 2026-05-01; no lilybench jobs in queue since)

**Plan file:** `/home/matteo/.claude/plans/i-want-to-start-cozy-dove.md`

## Phase

**Phase 5 — closed.** No lilybench jobs in `squeue --me`; last sacct entry for the sweep is `infer-deepseek-coder-lora{,-l40s}` TIMEOUT at 2026-05-01 05:45. Activity since (`fma-*`, `sonic-*`, `sig-*`, `study-cache-*`) is unrelated. Local `REPORT.md` regenerated against the synced `data/eval/` mirror — 15 `(model, regime)` cells covered (5 models × 3 regimes), of which 11 carry numbers and 4 are documented failures (see below).

## Final cell coverage (15 cells total)

| | zero | few | lora |
|---|---|---|---|
| phi4 | ✅ 1000 | ✅ 1000 | ⚠ raw 200 (0% compile) → rescued_v2 74/200 (25.7%) |
| qwen-coder | ✅ 1000 | ✅ 1000 | ⚠ raw 200 (0% compile) → rescued_v2 78/200 (17.9%) |
| codestral | ✅ 1000 | ✅ 1000 | ⚠ raw 40 partial (0% compile) → rescued_v2 14/40 (14.3%) — full rerun (4276113) FAILED at 4h27m |
| deepseek-coder | ⚠ 693/1000 partial | ⚠ 666/1000 partial | ❌ never produced output (racing twins 4285864/4285870 both TIMEOUT 12h, A40 & L40S) |
| gemma-2-27b | ❌ TIMEOUT 24h (370 samples emitted, but evaluated under `gemma_zero_gemma2_dropped/`) | ❌ TIMEOUT 24h (192/80 samples, `gemma_few_gemma2_dropped/`) | ❌ infer-gemma-lora cancelled (chain abandoned after gemma2 dropped from registry) |

Outputs live at `/nfsd/voce/machine_learning/experiments/lilybench/eval/<model>_<regime>/{summary,fmd_test,fmd_mutopia,loss}.json` and are mirrored locally under `data/eval/` (JSON-only rsync; 64 files, ~25 KB).

## What landed during the gap (2026-04-27 → 2026-05-01)

Picked up after the previous status.md cutoff:

| JobID | Name | Elapsed | State | Notes |
|---|---|---|---|---|
| 4235412 | infer-phi4-few (v2 stratified demos) | 05:01:15 | COMPLETED | 1000 samples |
| 4235413 | infer-qwen-coder-few (v2) | 06:24:53 | COMPLETED | 1000 samples |
| 4235415 | infer-codestral-few (v2) | 22:38:43 | COMPLETED | 1000 samples |
| 4235416 | infer-gemma-few (v2) | 1-00:00:27 | TIMEOUT | 192 samples → archived as `gemma_few_gemma2_dropped/` |
| 4254345 | js-ref-mutopia-converted | 00:06:54 | COMPLETED | Re-ran after `convert-ly` lifted Mutopia compile from 40% → 58% |
| 4254347-52 | eval-fmd-mut + eval-tm postconvert × 6 | ~3-15m | COMPLETED | Refreshed zero-shot summaries against improved Mutopia ref |
| 4254408 | infer-phi4-lora | 09:16:57 | COMPLETED | 200/1000 samples (LoRA inference was throttled to 200 in this iteration) |
| 4254409 | infer-qwen-coder-lora | 11:24:46 | COMPLETED | 200/1000 |
| 4254411 | infer-codestral-lora | 12:00:11 | TIMEOUT | 40 partial samples |
| 4254412 | infer-gemma-lora | 04:04:40 | CANCELLED | Gemma-2 dropped from registry |
| 4254413-15 | eval-tm + eval-fmd phi4_lora | ~3m each | COMPLETED/FAILED | Raw LoRA: 0% compile → triggered the rescue work |
| 4254416 | eval-loss-phi4_lora | 01:12:11 | COMPLETED | loss = 0.5317 |
| 4254417-19 | eval-tm + eval-fmd qwen-coder_lora | ~3m each | COMPLETED | Raw: 0% compile, FMD/test = 0.147 (best in table) |
| 4254420 | eval-loss-qwen-coder_lora | 00:37:43 | CANCELLED+ | replaced by 4276261 (fixed) |
| 4268028-33 | eval-tm + eval-fmd gemma_{zero,few} partial | <1m | COMPLETED | Numbers archived under `gemma_*_gemma2_dropped/`; not included in `_lora/_zero/_few` matchable folders |
| 4272209/4272650 | eval-tm-phi4_lora-rescued{,-v2} | ~3m | COMPLETED | Rescue pipeline: brace-balance v1 → score-rebuild v2 |
| 4272657-59 | eval-tm + eval-fmd qwen-coder_lora_rescued_v2 | ~3m | COMPLETED | 17.9% compile after rescue; JS/test=93.89 (table-best) |
| 4272662-63 | eval-fmd phi4_lora_rescued_v2 | ~30s | COMPLETED | FMD/test=0.417 after rescue |
| 4272692-99 | infer-codestral/deepseek-coder-few-old + their eval chains | various | COMPLETED | Old simple-demo ablation rows for codestral and deepseek-coder |
| 4276105-12 | codestral_lora partial eval + rescue chain | ~30s each | COMPLETED | Rescued 14/40 = 14.3% compile |
| 4276113 | infer-codestral-lora (full rerun) | 04:27:02 | FAILED | Full 1000-sample LoRA attempt collapsed; never re-eval'd |
| 4276260/4276261 | eval-loss-{phi4,qwen-coder}_lora-fixed | ~1h12m / 1h13m | COMPLETED | Final loss numbers (phi4: 0.5317, qwen-coder: 0.5549) |
| 4285856-61 | eval-tm + eval-fmd deepseek-coder_{zero,few} partial | ~3-12m | COMPLETED | 693/666 samples — these are the partial deepseek-coder rows in REPORT.md |
| 4285863 | eval-loss-codestral_lora-4h | 02:18:36 | FAILED | Codestral LoRA loss never landed (rerun failed first) |
| 4285864/4285870 | infer-deepseek-coder-lora (A40 / L40S race) | 12:00:10 each | TIMEOUT | Both timed out; deepseek-coder_lora has no inference output at all |

## Final eval numbers (REPORT.md)

REPORT.md has been regenerated locally from the synced eval JSONs. Highlights from the 11 populated cells:

| metric leader | cell | value |
|---|---|---|
| highest compile (real demos) | codestral_zero | 79.3% |
| highest compile (with caveat) | qwen-coder_few_old_simple_demos | 98.9% — distribution collapse |
| highest JS/test | qwen-coder_lora_rescued_v2 | 93.89 |
| highest JS/mutopia | qwen-coder_lora_rescued_v2 | 77.69 |
| lowest FMD/test (raw, no compile) | qwen-coder_lora | 0.147 |
| lowest FMD/test (compiles>0) | qwen-coder_lora_rescued_v2 | 0.359 |
| lowest FMD/mutopia | qwen-coder_lora_rescued_v2 | 1.669 (raw qwen-coder_lora at 1.263 is lower but 0% compile) |
| lowest LoRA loss | phi4_lora | 0.5317 |

Full results table + interpretation: `RESULTS.md` (preceded the sweep close, but conclusions stand).

## Failures the sweep ate, with root cause

1. **deepseek-coder LoRA inference, all attempts.** Vendored DeepSeek-V2 modeling code crashed on `transformers` ≥5.x cache API; even after the `seen_tokens → get_seq_length()` shim, the racing 12h jobs on A40 and L40S both TIMEOUT without writing samples. **Not retried** — would need either a pinned `transformers <5` env on cluster or a port of the cache compat patch to the generation loop.
2. **gemma-2-27b throughout.** Wall budget too small for 27B model in any regime; 24h produces ≤370 zero-shot and ≤192 few-shot samples. Dropped from the headline matrix; partial outputs archived under `gemma_*_gemma2_dropped/`.
3. **codestral_lora full 1000 inference (4276113).** Failed at 4h27m. The 40-sample partial + rescue chain stayed as the codestral_lora row.
4. **codestral_lora loss eval.** Two attempts (4276109 timeout, 4285863 fail). No `loss.json` exists for codestral_lora.

## Reconciliation against cluster state

- `squeue --me` empty.
- Last lilybench-tagged sacct entry: 4285864/70 ending 2026-05-01 05:45 (TIMEOUT).
- All other sacct entries since are unrelated projects (sonic, fma, sig, study-cache).
- Local `data/eval/` matches NFS `eval/` (rsync'd JSONs; bulky MIDI/sample subtrees left on cluster).
- `REPORT.md` regenerated 2026-05-15 — `python scripts/build_report.py --eval-root data/eval --out REPORT.md` (15 runs).

## Open items (not retried this session)

- deepseek-coder LoRA pipeline never produced samples — would need cache-compat patch + rerun.
- codestral LoRA never had a clean 1000-sample run nor a successful loss eval.
- gemma-2-27b never had enough wall time per job — would need a different model footprint (or accept "dropped").

If a Phase 6 happens, the obvious moves are: (a) pin `transformers<5` and rerun deepseek-coder; (b) request 48h walls for codestral LoRA + loss; (c) drop or swap gemma-2-27b.

## Critical file paths

- Splits (NFS): `/nfsd/voce/machine_learning/experiments/lilybench/data/splits_full/{train,val,test}.jsonl`
- Prompt bank (NFS): `/nfsd/voce/machine_learning/experiments/lilybench/data/prompt_bank/bank_1000.jsonl`
- HF cache (NFS): `/nfsd/voce/machine_learning/experiments/lilybench/huggingface_cache`
- bmdataset preprocessed (NFS): `/nfsd/voce/machine_learning/experiments/lilybench/data/bmdataset/preprocessed/`
- FMD refs (NFS): `/nfsd/voce/machine_learning/experiments/lilybench/fmd_refs/{test,mutopia}_lilybert_L6.npz`
- JS refs (NFS): `/nfsd/voce/machine_learning/experiments/lilybench/js_refs/{bmdataset,mutopia}_muspy_agg.json`
- LoRA adapters: `/nfsd/voce/machine_learning/experiments/lilybench/runs/<model>_lora{,_l40s}/final`
- Inference outputs: `/nfsd/voce/machine_learning/experiments/lilybench/inference/<model>_<regime>/samples/`
- Evaluation outputs: `/nfsd/voce/machine_learning/experiments/lilybench/eval/<model>_<regime>/{summary,fmd_test,fmd_mutopia,loss}.json`
- Local mirror (JSONs only): `data/eval/<model>_<regime>/*.json`
- Old few-shot / dropped archives: `/nfsd/voce/machine_learning/experiments/lilybench/{inference,eval}/{phi4,qwen-coder,codestral,deepseek-coder}_few_old_simple_demos/` and `…/gemma_*_gemma2_dropped/`
- Mutopia corpus: `/nfsd/voce/machine_learning/datasets/mutopia/dataset_mutopia.json`
- LilyPond binary: `/home/spanio/lilypond-2.24.4/bin/lilypond`
