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
        ("JS-sim", "js_divergence_similarity", 2),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("data/eval"))
    parser.add_argument("--out", type=Path, default=Path("REPORT.md"))
    args = parser.parse_args()
    build_report(args.eval_root, args.out)


if __name__ == "__main__":
    main()
