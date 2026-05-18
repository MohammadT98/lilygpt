"""Unified CLI for LilyBench experiments.

The CLI deliberately exposes a small surface — one verb per pipeline
step — so a paper-reproduction script can be expressed as a handful of
shell lines:

.. code-block:: bash

    lilybench prompt-bank build  --corpus bmdataset --n 200 \\
        --bmdataset-dir data/bmdataset/preprocessed \\
        --bmdataset-metadata data/bmdataset/metadata.json \\
        --out data/prompt_bank.jsonl

    lilybench generation run --model phi4 --regime zero \\
        --prompts data/prompt_bank.jsonl --out runs/phi4_zero

    lilybench understanding build --tasks all --corpus mutopia \\
        --mutopia-manifest data/mutopia/dataset_mutopia.json \\
        --out data/understanding/bench.jsonl

    lilybench understanding run --model phi4 \\
        --bench data/understanding/bench.jsonl \\
        --out runs/und/phi4

    lilybench metrics generation --samples runs/phi4_zero/samples \\
        --reference-test data/splits/test.jsonl \\
        --lilybert /path/to/lilybert

    lilybench metrics understanding --bench data/understanding/bench.jsonl \\
        --predictions runs/und/phi4 --out runs/und/phi4/summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lilybench import __version__


def _add_prompt_bank(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("prompt-bank", help="Build the generation prompt bank")
    pp = p.add_subparsers(dest="action", required=True)

    b = pp.add_parser("build", help="Sample N prompts from a corpus")
    b.add_argument("--corpus", choices=["bmdataset"], default="bmdataset")
    b.add_argument("--bmdataset-dir", type=Path, required=True)
    b.add_argument("--bmdataset-metadata", type=Path, required=True)
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--seed", type=int, default=1234)
    b.add_argument("--bars", type=int, default=None,
                   help="If set, emit short-fragment prompts capped to ~N bars.")
    b.add_argument("--out", type=Path, required=True)


def _add_generation(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("generation", help="Run the generation benchmark")
    pp = p.add_subparsers(dest="action", required=True)

    r = pp.add_parser("run", help="Run a registered model on a prompt bank")
    r.add_argument("--model", required=True)
    r.add_argument("--regime", default="zero",
                   help="Regime name (zero, few, or a custom registered one).")
    r.add_argument("--prompts", type=Path, required=True)
    r.add_argument("--fewshot-file", type=Path,
                   help="Demonstrations file (required when --regime=few).")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--max-new-tokens", type=int, default=3000)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--top-p", type=float, default=0.9)
    r.add_argument("--seed-base", type=int, default=1234)
    r.add_argument("--quantization", choices=["int8", "int4"], default=None)


def _add_understanding(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("understanding", help="Build & run the understanding benchmark")
    pp = p.add_subparsers(dest="action", required=True)

    b = pp.add_parser("build", help="Build a bench JSONL from a corpus")
    b.add_argument("--tasks", default="all",
                   help="'all' or comma-separated task names.")
    b.add_argument("--corpus", choices=["mutopia", "emopia"], default="mutopia")
    b.add_argument("--mutopia-manifest", type=Path)
    b.add_argument("--emopia-manifest", type=Path)
    b.add_argument("--emopia-ly-root", type=Path)
    b.add_argument("--seed", type=int, default=1234)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--out", type=Path, required=True)

    r = pp.add_parser("run", help="Run a registered model on a bench JSONL")
    r.add_argument("--model", required=True)
    r.add_argument("--bench", type=Path, required=True)
    r.add_argument("--tasks", default="all")
    r.add_argument("--limit", type=int, default=None,
                   help="Cap records per task (smoke runs).")
    r.add_argument("--max-new-tokens", type=int, default=20)
    r.add_argument("--max-input-chars", type=int, default=None)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--quantization", choices=["int8", "int4"], default=None)
    r.add_argument("--out", type=Path, required=True)


def _add_metrics(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("metrics", help="Score generation / understanding outputs")
    pp = p.add_subparsers(dest="action", required=True)

    g = pp.add_parser("generation", help="Compile rate + JS-similarity + FMD")
    g.add_argument("--samples", type=Path, required=True,
                   help="Directory of generated .ly files.")
    g.add_argument("--midi-dir", type=Path, default=None,
                   help="Where to drop compiled MIDI (omit to discard).")
    g.add_argument("--reference-test", type=Path,
                   help="In-domain reference JSONL (BMdataset test split) for FMD.")
    g.add_argument("--reference-mutopia", type=Path,
                   help="Out-of-domain reference manifest (Mutopia) for FMD.")
    g.add_argument("--reference-test-midi-dir", type=Path,
                   help="MIDI directory used to build the JS in-domain reference.")
    g.add_argument("--reference-mutopia-midi-dir", type=Path,
                   help="MIDI directory used to build the JS out-of-domain reference.")
    g.add_argument("--js-reference-cache", type=Path,
                   help="Optional load/save path for the JS aggregate.")
    g.add_argument("--lilybert", type=Path,
                   help="LilyBERT checkpoint for FMD.")
    g.add_argument("--lilybert-layer", type=int, default=6,
                   help="LilyBERT hidden-state layer (paper: 6).")
    g.add_argument("--lilybert-cache", type=Path,
                   help="Optional reference-embeddings .npz cache.")
    g.add_argument("--out", type=Path, required=True)

    u = pp.add_parser("understanding", help="Per-task scoring over predictions")
    u.add_argument("--bench", type=Path, required=True)
    u.add_argument("--predictions", type=Path, required=True,
                   help="Directory with one <task>.jsonl per task.")
    u.add_argument("--model-id", required=False, default=None)
    u.add_argument("--out", type=Path, required=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lilybench",
        description="Companion CLI for the LilyBench paper (Ital-IA 2026).",
    )
    parser.add_argument("--version", action="version", version=f"lilybench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_prompt_bank(sub)
    _add_generation(sub)
    _add_understanding(sub)
    _add_metrics(sub)
    return parser


# --------------------------------------------------------------------- handlers

def _run_prompt_bank(args: argparse.Namespace) -> int:
    from lilybench.data import load_bmdataset
    from lilybench.generation.prompt_bank import build_prompt_bank, write_prompt_bank

    corpus = load_bmdataset(args.bmdataset_dir, args.bmdataset_metadata)
    bank = build_prompt_bank(corpus, n=args.n, seed=args.seed, bars=args.bars)
    write_prompt_bank(bank, args.out)
    print(f"[lilybench] wrote {len(bank)} prompts to {args.out}")
    return 0


def _run_generation(args: argparse.Namespace) -> int:
    from lilybench.generation import (
        GenerationConfig, FewShot, REGIME_REGISTRY, generate, load_prompt_bank,
    )
    from lilybench.models import get_spec

    spec = get_spec(args.model)
    if args.regime not in REGIME_REGISTRY:
        print(f"[lilybench] unknown regime {args.regime!r}; "
              f"known: {sorted(REGIME_REGISTRY)}", file=sys.stderr)
        return 2

    if args.regime == "few":
        if not args.fewshot_file:
            print("[lilybench] --fewshot-file is required for --regime few",
                  file=sys.stderr)
            return 2
        regime = FewShot.from_file(args.fewshot_file)
    else:
        regime = REGIME_REGISTRY[args.regime]()

    bank = load_prompt_bank(args.prompts)
    cfg = GenerationConfig(
        output_dir=args.out,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed_base=args.seed_base,
        quantization=args.quantization,
    )
    generate(spec=spec, regime=regime, bank=bank, cfg=cfg)
    return 0


def _filter_task_names(value: str) -> set[str] | None:
    if value == "all" or not value:
        return None
    return {name.strip() for name in value.split(",") if name.strip()}


def _build_understanding(args: argparse.Namespace) -> int:
    from lilybench import understanding as _und  # noqa: F401 - triggers registration
    from lilybench.understanding.registry import TASK_REGISTRY, get_task
    from lilybench.utils import write_jsonl

    only = _filter_task_names(args.tasks)
    if args.corpus == "mutopia":
        if not args.mutopia_manifest:
            print("[lilybench] --mutopia-manifest required", file=sys.stderr)
            return 2
        from lilybench.data import load_mutopia
        corpus = load_mutopia(args.mutopia_manifest)
    else:
        if not args.emopia_manifest or not args.emopia_ly_root:
            print("[lilybench] --emopia-manifest and --emopia-ly-root required",
                  file=sys.stderr)
            return 2
        from lilybench.data import load_emopia
        corpus = load_emopia(args.emopia_manifest, args.emopia_ly_root)
    print(f"[lilybench] loaded {len(corpus)} {args.corpus} entries")

    records: list[dict] = []
    for name in sorted(TASK_REGISTRY):
        if only is not None and name not in only:
            continue
        task = get_task(name)
        if args.corpus == "mutopia" and task.corpus_kind == "emopia":
            continue
        if args.corpus == "emopia" and task.corpus_kind != "emopia":
            continue
        built = task.build(
            corpus,
            n=args.limit or task.default_n,
            seed=args.seed,
        )
        for r in built:
            records.append(r.to_dict())
    write_jsonl(records, args.out)
    print(f"[lilybench] wrote {len(records)} bench records to {args.out}")
    return 0


def _run_understanding(args: argparse.Namespace) -> int:
    from lilybench import understanding as _und  # noqa: F401
    from lilybench.models import get_spec
    from lilybench.understanding.runner import UnderstandingConfig, run_understanding
    from lilybench.utils import iter_jsonl

    only = _filter_task_names(args.tasks)
    bench = []
    per_task: dict[str, int] = {}
    for rec in iter_jsonl(args.bench):
        t = rec.get("task")
        if only is not None and t not in only:
            continue
        if args.limit is not None and per_task.get(t, 0) >= args.limit:
            continue
        per_task[t] = per_task.get(t, 0) + 1
        bench.append(rec)

    spec = get_spec(args.model)
    cfg = UnderstandingConfig(
        output_dir=args.out,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        quantization=args.quantization,
        max_input_chars=args.max_input_chars,
    )
    run_understanding(spec=spec, bench=bench, cfg=cfg)
    return 0


def _metrics_generation(args: argparse.Namespace) -> int:
    from lilybench.metrics import (
        aggregate_descriptor_stats,
        compile_to_midi,
        compute_muspy_descriptors,
        frechet_music_distance,
        js_descriptor_similarity,
        lilybert_embed,
    )
    from lilybench.metrics.compile_rate import compile_rate
    from lilybench.metrics.fmd import load_documents
    from lilybench.metrics.js_similarity import load_or_build_reference
    from lilybench.utils import write_json, read_jsonl

    samples = sorted(Path(args.samples).glob("*.ly"))
    print(f"[lilybench] {len(samples)} samples")

    midi_dir = args.midi_dir or (args.out.parent / "midi")
    rate, comp_results = compile_rate(samples, midi_dir=midi_dir)
    print(f"[lilybench] compile_rate = {rate:.4f} ({sum(r.ok for r in comp_results)}/{len(comp_results)})")

    summary: dict = {
        "n_samples": len(samples),
        "compile_rate": rate,
        "compile_results": [
            {"path": str(r.path), "ok": r.ok, "seconds": r.seconds,
             "midi": str(r.midi_path) if r.midi_path else None, "error": r.error}
            for r in comp_results
        ],
    }

    # JS similarity over the three muspy descriptors.
    compiled_midis = [r.midi_path for r in comp_results if r.ok and r.midi_path]
    per_file_desc = {str(p): compute_muspy_descriptors(p) for p in compiled_midis}
    model_agg = aggregate_descriptor_stats(per_file_desc)
    summary["muspy_descriptors"] = model_agg

    js_results: dict[str, float] = {}
    if args.reference_test_midi_dir:
        ref_test = load_or_build_reference(
            args.js_reference_cache if args.js_reference_cache else None,
            args.reference_test_midi_dir,
        )
        if ref_test is not None:
            v = js_descriptor_similarity(model_agg, ref_test)
            if v is not None:
                js_results["test"] = v
    if args.reference_mutopia_midi_dir:
        ref_mut = load_or_build_reference(
            None, args.reference_mutopia_midi_dir,
        )
        if ref_mut is not None:
            v = js_descriptor_similarity(model_agg, ref_mut)
            if v is not None:
                js_results["mutopia"] = v
    summary["js_similarity"] = js_results

    # FMD with LilyBERT.
    fmd_results: dict[str, float] = {}
    if args.lilybert:
        device = "cuda" if _cuda_available() else "cpu"
        gens = load_documents(samples)
        if len(gens) >= 2:
            x = lilybert_embed(
                gens, checkpoint=args.lilybert, device=device,
                embed_layer=args.lilybert_layer,
            )
            for label, source in (
                ("test", args.reference_test),
                ("mutopia", args.reference_mutopia),
            ):
                if not source or not source.exists():
                    continue
                refs = _load_ref_docs(source)
                if len(refs) < 2:
                    continue
                y = lilybert_embed(
                    refs, checkpoint=args.lilybert, device=device,
                    embed_layer=args.lilybert_layer,
                )
                fmd_results[label] = frechet_music_distance(x, y)
                print(f"[lilybench] FMD({label}) = {fmd_results[label]:.4f}")
    summary["fmd"] = fmd_results

    write_json(summary, args.out)
    print(f"[lilybench] wrote {args.out}")
    return 0


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _load_ref_docs(path: Path) -> list[str]:
    """Reference loader: BMdataset JSONL (``full_text`` field) or Mutopia manifest."""
    from lilybench.data import load_mutopia
    from lilybench.utils import iter_jsonl

    path = Path(path)
    if path.suffix == ".jsonl":
        docs: list[str] = []
        for rec in iter_jsonl(path):
            txt = (rec.get("full_text") or rec.get("text") or "").strip()
            if len(txt) >= 40:
                docs.append(txt)
        return docs
    if path.suffix == ".json":
        return [e.text for e in load_mutopia(path) if len(e.text.strip()) >= 40]
    return []


def _metrics_understanding(args: argparse.Namespace) -> int:
    from lilybench import understanding as _und  # noqa: F401
    from lilybench.understanding.registry import TASK_REGISTRY, get_task
    from lilybench.utils import read_jsonl, write_json

    bench = read_jsonl(args.bench)
    by_task: dict[str, list[dict]] = {}
    for r in bench:
        by_task.setdefault(r["task"], []).append(r)

    per_task: dict[str, dict] = {}
    for name in sorted(TASK_REGISTRY):
        pred_path = args.predictions / f"{name}.jsonl"
        if not pred_path.exists():
            per_task[name] = {"task": name, "n": 0, "missing": True}
            continue
        preds = read_jsonl(pred_path)
        task = get_task(name)
        per_task[name] = task.score(by_task.get(name, []), preds)

    available = {k: v for k, v in per_task.items() if not v.get("missing")}
    pairs = [
        (
            t.get("accuracy", t.get("score", t.get("macro_f1", 0.0))),
            t.get("n", 0),
        )
        for t in available.values()
    ]
    macro = sum(s for s, _ in pairs) / len(pairs) if pairs else 0.0
    total_n = sum(n for _, n in pairs)
    weighted = sum(s * n for s, n in pairs) / total_n if total_n else 0.0
    summary = {
        "model": args.model_id,
        "bench": str(args.bench),
        "predictions": str(args.predictions),
        "tasks": per_task,
        "overall": {"macro_avg": macro, "weighted_avg": weighted},
    }
    write_json(summary, args.out)
    print(f"[lilybench] wrote {args.out}")
    for name, t in per_task.items():
        if t.get("missing"):
            print(f"  {name:24s} (missing)")
        elif "score" in t:
            print(f"  {name:24s} score={t['score']:.3f}  n={t['n']}")
        elif "macro_f1" in t:
            print(f"  {name:24s} macro_f1={t['macro_f1']:.3f}  n={t['n']}")
        else:
            print(f"  {name:24s} acc={t['accuracy']:.3f}  n={t['n']}")
    print(f"  overall.macro_avg    = {macro:.3f}")
    print(f"  overall.weighted_avg = {weighted:.3f}")
    return 0


_HANDLERS = {
    ("prompt-bank", "build"): _run_prompt_bank,
    ("generation", "run"): _run_generation,
    ("understanding", "build"): _build_understanding,
    ("understanding", "run"): _run_understanding,
    ("metrics", "generation"): _metrics_generation,
    ("metrics", "understanding"): _metrics_understanding,
}


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    handler = _HANDLERS.get((args.command, args.action))
    if handler is None:
        print(f"[lilybench] no handler for {args.command}/{args.action}", file=sys.stderr)
        return 2
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
