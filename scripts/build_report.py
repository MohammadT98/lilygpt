#!/usr/bin/env python3
"""Aggregate per-(model, regime) evaluation outputs into REPORT.md.

Expected filesystem layout (conventionally under ``data/eval/`` or the NFS mirror)::

    <eval_root>/<model_id>_<regime>/
        summary.json      # from lilybench.evaluate.text_midi
        fmd_test.json     # FMD vs held-out test split
        fmd_mutopia.json  # FMD vs Mutopia out-of-domain corpus
        loss.json         # (regime=lora only) held-out body-token loss

Per-run folder names are parsed with the regex ``^(?P<model>.+?)_(?P<regime>zero|few|lora)$``.
Any files missing from a run are simply left blank in the tables (with a "—"
placeholder); the report is idempotent and safe to re-run after any subset of
jobs re-completes.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RUN_DIR_RE = re.compile(r"^(?P<model>.+?)_(?P<regime>zero|few|lora)$")
REGIME_ORDER = ("zero", "few", "lora")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[report] could not parse {path}: {exc}")
        return None


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _collect_runs(eval_root: Path) -> list[dict]:
    runs: list[dict] = []
    for child in sorted(eval_root.iterdir() if eval_root.exists() else []):
        if not child.is_dir():
            continue
        m = RUN_DIR_RE.match(child.name)
        if not m:
            continue
        runs.append(
            {
                "model": m["model"],
                "regime": m["regime"],
                "dir": child,
                "summary": _load_json(child / "summary.json"),
                "fmd_test": _load_json(child / "fmd_test.json"),
                "fmd_mutopia": _load_json(child / "fmd_mutopia.json"),
                "loss": _load_json(child / "loss.json"),
            }
        )
    # Sort: model alphabetical, regime in canonical order.
    runs.sort(key=lambda r: (r["model"], REGIME_ORDER.index(r["regime"]) if r["regime"] in REGIME_ORDER else 99))
    return runs


def _summary_all(run: dict) -> dict:
    s = run.get("summary") or {}
    return s.get("all") or {}


def _table_text_midi(runs: list[dict]) -> str:
    columns = [
        ("n", "count", 0),
        ("lily_ok", "lily_ok_rate", 3),
        ("compiles", "compiles_rate", 3),
        ("midi_ok", "midi_render_ok_rate", 3),
        ("has_key_time", "has_key_time_rate", 3),
        ("in_key_pct", "in_key_pct_time_weighted_avg", 3),
        ("ends_tonic", "ends_on_tonic_rate", 3),
        ("step_vs_leap", "step_vs_leap_avg", 3),
        ("drift_rate", "drift_detected_rate", 3),
    ]
    header = "| model | regime | " + " | ".join(c[0] for c in columns) + " |"
    sep = "|" + "|".join("---" for _ in range(2 + len(columns))) + "|"
    lines = [header, sep]
    for r in runs:
        s = _summary_all(r)
        row = [r["model"], r["regime"]] + [_fmt(s.get(k), d) for _, k, d in columns]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_muspy(runs: list[dict]) -> str:
    columns = [
        ("pitch_range", "muspy_pitch_range_avg", 2),
        ("n_pitches", "muspy_n_pitches_used_avg", 2),
        ("pitch_H", "muspy_pitch_entropy_avg", 3),
        ("pclass_H", "muspy_pitch_class_entropy_avg", 3),
        ("scale_cons", "muspy_scale_consistency_avg", 3),
        ("in_scale", "muspy_pitch_in_scale_rate_avg", 3),
        ("polyphony", "muspy_polyphony_avg", 3),
        ("groove_cons", "muspy_groove_consistency_avg", 3),
        ("JS-sim/test", "js_divergence_similarity_test", 2),
        ("JS-sim/mutopia", "js_divergence_similarity_mutopia", 2),
    ]
    header = "| model | regime | " + " | ".join(c[0] for c in columns) + " |"
    sep = "|" + "|".join("---" for _ in range(2 + len(columns))) + "|"
    lines = [header, sep]
    for r in runs:
        s = _summary_all(r)
        row = [r["model"], r["regime"]] + [_fmt(s.get(k), d) for _, k, d in columns]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_fmd(runs: list[dict]) -> str:
    header = "| model | regime | FMD (test, in-domain) | FMD (mutopia, OOD) | n_gen | ref_layer |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in runs:
        t = r.get("fmd_test") or {}
        m = r.get("fmd_mutopia") or {}
        n_gen = t.get("n_generations") or m.get("n_generations")
        layer = t.get("embed_layer") if t.get("embed_layer") is not None else m.get("embed_layer")
        row = [
            r["model"],
            r["regime"],
            _fmt(t.get("fmd"), 4),
            _fmt(m.get("fmd"), 4),
            _fmt(n_gen, 0),
            _fmt(layer, 0),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_loss(runs: list[dict]) -> str:
    header = "| model | eval_loss | n_samples | lora_path |"
    sep = "|---|---|---|---|"
    lines = [header, sep]
    for r in runs:
        if r["regime"] != "lora":
            continue
        ld = r.get("loss") or {}
        metrics = ld.get("metrics") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    r["model"],
                    _fmt(metrics.get("eval_loss"), 4),
                    _fmt(ld.get("n_samples"), 0),
                    str(ld.get("lora_path", "—")),
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| (no LoRA loss JSONs found) | — | — | — |")
    return "\n".join(lines)


def _provenance_table(runs: list[dict]) -> str:
    lines = ["| model | regime | eval dir | summary | fmd_test | fmd_mutopia | loss |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        d = r["dir"]
        lines.append(
            "| "
            + " | ".join(
                [
                    r["model"],
                    r["regime"],
                    str(d),
                    _fmt_exists(d / "summary.json"),
                    _fmt_exists(d / "fmd_test.json"),
                    _fmt_exists(d / "fmd_mutopia.json"),
                    _fmt_exists(d / "loss.json") if r["regime"] == "lora" else "n/a",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _fmt_exists(p: Path) -> str:
    return p.name if p.exists() else "missing"


def build_report(eval_root: Path, out_path: Path) -> None:
    runs = _collect_runs(eval_root)
    if not runs:
        out_path.write_text(
            f"# REPORT\n\nNo evaluation runs found under `{eval_root}`. "
            f"Populate it with `<model>_<regime>/` subfolders (e.g. from the sweep) and re-run this script.\n",
            encoding="utf-8",
        )
        print(f"[report] no runs found under {eval_root}")
        return

    parts = [
        "# LilyBench evaluation REPORT",
        "",
        f"Aggregated from `{eval_root}`. Each row maps to one `(model, regime)` sample directory of 1000 generations.",
        "",
        "## 1. Text + MIDI quality",
        "",
        _table_text_midi(runs),
        "",
        "## 2. Muspy symbolic-music metrics",
        "",
        _table_muspy(runs),
        "",
        "## 3. Fréchet Music Distance (LilyBERT layer 6)",
        "",
        _table_fmd(runs),
        "",
        "## 4. LoRA held-out loss (body tokens only)",
        "",
        _table_loss(runs),
        "",
        "## 5. Provenance",
        "",
        "Every number above comes from the JSON file in the named run directory. `missing` means that evaluation job has not (yet) completed.",
        "",
        _provenance_table(runs),
        "",
    ]
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[report] wrote {out_path} ({len(runs)} runs)")


UNDERSTANDING_TASK_ORDER = (
    "bar_count",
    "metadata_qa",
    "bar_sequencing",
    "next_bar_prediction",
    "metadata_prediction",
    "music_captioning",
    "composer_recognition",
    "genre_recognition",
    "emotion_recognition",
    "error_detection",
)


def _collect_understanding(root: Path) -> dict[str, dict]:
    """Read ``<root>/<model>/summary.json`` from the understanding eval root.

    Returns ``{model_id: summary_dict}``. Missing or unreadable summaries are
    silently skipped — call sites render them as ``—``.
    """
    out: dict[str, dict] = {}
    if not root.exists():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        summary = _load_json(child / "summary.json")
        if summary is None:
            continue
        out[child.name] = summary
    return out


def _table_understanding(summaries: dict[str, dict]) -> str:
    if not summaries:
        return "_(no understanding evaluations found)_"
    models = sorted(summaries.keys())
    header = "| task | " + " | ".join(models) + " |"
    sep = "|" + "|".join("---" for _ in range(1 + len(models))) + "|"
    lines = [header, sep]
    for task in UNDERSTANDING_TASK_ORDER:
        row = [task]
        for m in models:
            entry = (summaries[m].get("tasks") or {}).get(task)
            if not entry or entry.get("missing"):
                row.append("—")
                continue
            # error_detection uses macro_f1, others use accuracy or score.
            score = entry.get("accuracy", entry.get("score", entry.get("macro_f1")))
            n = entry.get("n", 0)
            row.append(f"{_fmt(score, 3)} (n={n})")
        lines.append("| " + " | ".join(row) + " |")
    # Overall averages row.
    avg_row = ["**macro_avg**"]
    weighted_row = ["**weighted_avg**"]
    for m in models:
        overall = summaries[m].get("overall") or {}
        avg_row.append(_fmt(overall.get("macro_avg"), 3))
        weighted_row.append(_fmt(overall.get("weighted_avg"), 3))
    lines.append("| " + " | ".join(avg_row) + " |")
    lines.append("| " + " | ".join(weighted_row) + " |")
    return "\n".join(lines)


def _table_bar_count_tolerance(summaries: dict[str, dict]) -> str:
    """Per-model bar_count accuracy under ±1 / ±5 / ±10 tolerance windows."""
    models = sorted(summaries.keys())
    rows_have_any = False
    header = "| model | exact | within ±1 | within ±5 | within ±10 | mean abs err | median abs err |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for m in models:
        entry = (summaries[m].get("tasks") or {}).get("bar_count")
        if not entry or entry.get("missing"):
            continue
        tol = entry.get("tolerance") or {}
        if not tol:
            continue
        rows_have_any = True
        lines.append(
            "| " + " | ".join([
                m,
                _fmt(entry.get("accuracy"), 3),
                _fmt(tol.get("within_1"), 3),
                _fmt(tol.get("within_5"), 3),
                _fmt(tol.get("within_10"), 3),
                _fmt(tol.get("mean_abs_err"), 2),
                _fmt(tol.get("median_abs_err"), 0),
            ]) + " |"
        )
    if not rows_have_any:
        return "_(no bar_count tolerance data available)_"
    return "\n".join(lines)


def _table_emotion_confusion(summaries: dict[str, dict]) -> str:
    """Per-model 4×4 confusion matrices for emotion_recognition."""
    models = sorted(summaries.keys())
    parts: list[str] = []
    quads = ("Q1", "Q2", "Q3", "Q4")
    for m in models:
        entry = (summaries[m].get("tasks") or {}).get("emotion_recognition")
        if not entry or entry.get("missing"):
            continue
        conf = entry.get("confusion")
        if not conf or not conf.get("matrix"):
            continue
        matrix = conf["matrix"]
        parts.append(f"**{m}** (off-grid predictions: {conf.get('n_off_grid', 0)})")
        parts.append("")
        parts.append("| gold ↓ / pred → | " + " | ".join(quads) + " |")
        parts.append("|" + "|".join(["---"] * (1 + len(quads))) + "|")
        for g in quads:
            row = [g]
            row_total = sum(matrix.get(g, {}).get(p, 0) for p in quads)
            for p in quads:
                cell = matrix.get(g, {}).get(p, 0)
                pct = (cell / row_total) if row_total else 0.0
                row.append(f"{cell} ({pct:.0%})")
            parts.append("| " + " | ".join(row) + " |")
        parts.append("")
    if not parts:
        return "_(no emotion_recognition confusion matrices available)_"
    return "\n".join(parts)


def build_understanding_report(und_root: Path, out_path: Path) -> None:
    summaries = _collect_understanding(und_root)
    parts = [
        "# LilyBench music-understanding REPORT",
        "",
        f"Aggregated from `{und_root}`. One column per model; rows are the 10 tasks plus overall averages.",
        "",
        "## 1. Task × model matrix",
        "",
        _table_understanding(summaries),
        "",
        "## 2. Bar-count tolerance breakdown",
        "",
        "Exact-match is brutally narrow on bar_count (the gold spans 1-1500 bars). "
        "These columns relax to ±N tolerance windows and show the underlying "
        "magnitude of the error.",
        "",
        _table_bar_count_tolerance(summaries),
        "",
        "## 3. Emotion-recognition confusion matrices",
        "",
        "Per-model 4×4 confusion matrices over Russell quadrants. Rows are the "
        "gold quadrant; columns are the model's predicted quadrant. Counts and "
        "row-percentages.",
        "",
        _table_emotion_confusion(summaries),
        "",
    ]
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[report] wrote {out_path} ({len(summaries)} models)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("data/eval"))
    parser.add_argument("--out", type=Path, default=Path("REPORT.md"))
    parser.add_argument(
        "--understanding-root",
        type=Path,
        default=None,
        help="If set, additionally emit a per-task × model matrix from this dir "
             "(structure: <root>/<model>/summary.json). Output: REPORT_understanding.md "
             "next to --out unless --understanding-out is given.",
    )
    parser.add_argument("--understanding-out", type=Path, default=None)
    args = parser.parse_args()
    build_report(args.eval_root, args.out)
    if args.understanding_root is not None:
        und_out = args.understanding_out or args.out.with_name("REPORT_understanding.md")
        build_understanding_report(args.understanding_root, und_out)


if __name__ == "__main__":
    main()
