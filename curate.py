#!/usr/bin/env python3
"""Curate a small, faithful demo subset from the cluster pull (``_pull/``).

Writes everything the static site needs into ``demo-src/`` (which ships on the
gh-pages branch). Needs the cluster artifacts staged under ``_pull/`` — that is
the only non-reproducible-without-cluster step; ``build.py`` renders SVG/MP3
from ``demo-src/`` alone. Re-run only when changing which samples are featured.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

PULL = Path("_pull")
OUT = Path("demo-src")
MODELS = ["phi4", "qwen-coder", "deepseek-coder", "codestral"]
MODEL_NAME = {
    "phi4": "Phi-4",
    "qwen-coder": "Qwen2.5-Coder-14B",
    "deepseek-coder": "DeepSeek-Coder-V2-Lite",
    "codestral": "Codestral-22B",
}
# Curated generation prompts (verified: compile + real music, diverse composers).
GEN_PROMPTS = [240, 17, 193]
CATEGORY = {  # paper Table: reasoning depth
    "bar_count": "Basic", "metadata_qa": "Basic",
    "bar_sequencing": "Segment", "next_bar_prediction": "Segment",
    "metadata_prediction": "Segment",
    "music_captioning": "Sequence", "composer_recognition": "Sequence",
    "genre_recognition": "Sequence", "emotion_recognition": "Sequence",
    "error_detection": "Sequence",
}
# one item per task: target #models-correct (out of 4) + the story it tells
UND_TARGET = {
    "genre_recognition": (4, "Recognition: all four models agree with the gold label."),
    "composer_recognition": (4, "Recognition: composer identified from surface style."),
    "music_captioning": (4, "Recognition: the right title picked from four options."),
    "metadata_qa": (4, "Reading metadata straight from the header."),
    "next_bar_prediction": (2, "Local continuation: models split."),
    "metadata_prediction": (2, "Inferring a masked field from the notes: mixed."),
    "bar_sequencing": (1, "Ordering shuffled bars: mostly wrong."),
    "emotion_recognition": (0, "Valence/arousal collapses — models default to one quadrant."),
    "bar_count": (0, "Counting bars: every model misses, defaulting to round phrase lengths."),
    "error_detection": (0, "Spotting the corrupted bar in one shot is out of reach."),
}


def jload(p: Path):
    return [json.loads(l) for l in p.open()] if p.exists() else []


# ---------------------------------------------------------------- generation
def curate_generation():
    bank = jload(PULL / "bank_1000.jsonl")
    summ = {m: json.load((PULL / f"eval/{m}_zero/summary.json").open())["by_group"] for m in MODELS}
    (OUT / "generation/ly").mkdir(parents=True, exist_ok=True)
    (OUT / "generation/midi").mkdir(parents=True, exist_ok=True)
    prompts = []
    for idx in GEN_PROMPTS:
        b = bank[idx]
        md = b["metadata"]
        entry = {
            "idx": idx,
            "metadata": {
                "composer": md.get("composer"),
                "period": md.get("period"),
                "form": ", ".join(md.get("musical_form") or []),
                "ensemble": ", ".join(md.get("ensemble") or []),
                "part": md.get("part"),
            },
            "user_prompt": b["user_prompt"],
            "models": [],
        }
        for m in MODELS:
            ly_src = PULL / f"inference/{m}_zero/samples/sample_{idx:04d}.ly"
            mid_src = PULL / f"midi/{m}_zero/sample_{idx:04d}.ly/sample_{idx:04d}_midi.mid"
            g = summ[m].get(f"sample_{idx:04d}.ly", {})
            ly_name = f"{idx:04d}_{m}.ly"
            mid_name = f"{idx:04d}_{m}.mid"
            shutil.copyfile(ly_src, OUT / "generation/ly" / ly_name)
            has_midi = mid_src.exists()
            if has_midi:
                shutil.copyfile(mid_src, OUT / "generation/midi" / mid_name)
            asc = g.get("ascending_tendency_avg")
            entry["models"].append({
                "id": m,
                "name": MODEL_NAME[m],
                "ly": ly_name,
                "midi": mid_name if has_midi else None,
                "compiles": (g.get("compiles_rate", 0) or 0) >= 0.5,
                "notes": int(g.get("note_count_avg") or 0),
                "scale_drift": asc is not None and asc >= 0.85,  # data-driven caption
            })
        prompts.append(entry)
    (OUT / "generation/prompts.json").write_text(json.dumps(prompts, indent=2))
    print(f"generation: {len(prompts)} prompts, {len(prompts)*len(MODELS)} panels")


# ------------------------------------------------------------- understanding
def _bench_index():
    bench = {}
    for f in ["bench.jsonl", "bench_emotion.jsonl", "bench_errors.jsonl"]:
        for r in jload(PULL / "understanding" / f):
            bench.setdefault(r["task"], {})[r["id"]] = r
    return bench


def _preds_for(task):
    for sub in ["predictions_l40s", "predictions_emotion", "predictions_errors"]:
        d, ok = {}, True
        for m in MODELS:
            p = PULL / "understanding" / sub / m / f"{task}.jsonl"
            if not p.exists():
                ok = False
                break
            d[m] = {r["id"]: r for r in jload(p)}
        if ok:
            return d
    return None


def _digits(s):
    return [int(t) for t in str(s).replace(",", " ").split() if t.lstrip("-").isdigit()]


def _correct(task, item, pred):
    pa = pred.get("parsed_answer")
    if pa in (None, ""):
        return False
    if item.get("template_kind") == "multiple_choice":
        try:
            return int(pa) == item["gold_index"]
        except ValueError:
            return False
    if task == "bar_count":
        try:
            return int(pa) == int(item["gold"])
        except (ValueError, TypeError):
            return False
    if task == "bar_sequencing":
        return str(pa) == str(item.get("gold"))
    if task == "error_detection":
        gold = set(item.get("gold_bars") or [])
        return bool(gold) and set(_digits(pred.get("raw_output", ""))) == gold
    return False


def _excerpt(text, limit=1500):
    """Show the musically-interesting head: skip \\paper/\\header boilerplate."""
    lines = text.splitlines()
    out, n = [], 0
    for ln in lines:
        out.append(ln)
        n += len(ln) + 1
        if n > limit:
            out.append("…")
            break
    return "\n".join(out)


def _answer_display(task, item, pred):
    pa = pred.get("parsed_answer")
    if item.get("template_kind") == "multiple_choice":
        try:
            return item["options"][int(pa)]
        except (ValueError, TypeError, IndexError):
            return pred.get("raw_output", "—").strip()[:24] or "—"
    if task == "error_detection":
        d = _digits(pred.get("raw_output", ""))
        return ("bars " + " ".join(map(str, d))) if d else (pred.get("raw_output", "—").strip()[:20] or "none")
    return (str(pa) if pa not in (None, "") else (pred.get("raw_output", "—").strip()[:20] or "—"))


def curate_understanding():
    bench = _bench_index()
    items_out = []
    for task in sorted(bench, key=lambda t: (["Basic", "Segment", "Sequence"].index(CATEGORY[t]), t)):
        preds = _preds_for(task)
        if not preds:
            continue
        target, story = UND_TARGET[task]
        # score every item; pick the one matching target with the shortest input
        cands = []
        for iid, item in bench[task].items():
            if not all(m in preds and iid in preds[m] for m in MODELS):
                continue
            c = sum(_correct(task, item, preds[m][iid]) for m in MODELS)
            cands.append((abs(c - target), len(item.get("input_content", "")), iid, c))
        cands.sort()
        _, _, iid, c = cands[0]
        item = bench[task][iid]
        rec = {
            "task": task,
            "category": CATEGORY[task],
            "instruction": item["task_instruction"],
            "template_kind": item["template_kind"],
            "story": story,
            "input_excerpt": _excerpt(item.get("input_content", "")),
            "options": item.get("options"),
            "gold": item.get("gold") if item.get("gold") is not None else (
                "bars " + " ".join(map(str, item.get("gold_bars", []))) if item.get("gold_bars") else None),
            "n_correct": c,
            "predictions": [
                {"id": m, "name": MODEL_NAME[m],
                 "answer": _answer_display(task, item, preds[m][iid]),
                 "correct": _correct(task, item, preds[m][iid])}
                for m in MODELS
            ],
        }
        items_out.append(rec)
    (OUT / "understanding").mkdir(parents=True, exist_ok=True)
    (OUT / "understanding/understanding.json").write_text(json.dumps(items_out, indent=2))
    print(f"understanding: {len(items_out)} task items")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    curate_generation()
    curate_understanding()
    print("curate.py done ->", OUT)
