from __future__ import annotations

"""Score the music-understanding predictions against the bench.

For multiple-choice tasks the metric is exact-match accuracy on the option
index. For ``bar_count`` it is exact-match accuracy on the integer answer.
For ``bar_sequencing`` it is the penalised Kendall-tau score from
:mod:`lilybench.understanding.scoring`.

Output: one ``summary.json`` per model under ``<eval_root>/<model_id>/``.

Invocation::

    python -m lilybench.evaluate.understanding model_id=phi4 \\
        predictions_dir=data/understanding/predictions/phi4 \\
        bench_path=data/understanding/bench.jsonl
"""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from collections import defaultdict

from lilybench.understanding import tasks
from lilybench.understanding.scoring import (
    accuracy,
    bar_sequencing_score,
    error_detection_f1,
    parse_bar_list,
)


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _score_task(task: str, bench: list[dict], preds: list[dict]) -> dict:
    by_id = {r["id"]: r for r in bench}
    by_pred = {p["id"]: p for p in preds}
    aligned = [(by_id[i], by_pred[i]) for i in by_pred if i in by_id]
    n = len(aligned)
    if task == "bar_sequencing":
        if n == 0:
            return {"task": task, "n": 0, "score": 0.0, "n_valid": 0}
        per_item = [
            bar_sequencing_score(pred=p["parsed_answer"], gold=b["gold"])
            for b, p in aligned
        ]
        n_valid = sum(1 for s in per_item if s > 0)
        return {
            "task": task,
            "n": n,
            "score": sum(per_item) / n,
            "n_valid": n_valid,
        }

    if task == "bar_count":
        preds_str = [p["parsed_answer"] for _, p in aligned]
        golds = [b["gold"] for b, _ in aligned]
        n_parsed = sum(1 for s in preds_str if s.isdigit())
        tolerance = _bar_count_tolerance(preds_str, golds)
        return {
            "task": task,
            "n": n,
            "accuracy": accuracy(preds_str, golds) if n else 0.0,
            "n_parsed": n_parsed,
            "tolerance": tolerance,
        }

    if task == "error_detection":
        return _score_error_detection(aligned)

    # 4-way MC tasks: compare option-index strings.
    preds_str = [p["parsed_answer"] for _, p in aligned]
    golds = [str(b["gold_index"]) for b, _ in aligned]
    n_parsed = sum(1 for s in preds_str if s in {"0", "1", "2", "3"})
    summary = {
        "task": task,
        "n": n,
        "accuracy": accuracy(preds_str, golds) if n else 0.0,
        "n_parsed": n_parsed,
    }
    if task == "emotion_recognition":
        summary["confusion"] = _emotion_confusion_matrix(aligned)
    return summary


def _bar_count_tolerance(preds_str: list[str], golds: list[str]) -> dict[str, float]:
    """Return {within_1, within_5, within_10} accuracy + mean/median abs err.

    Skips records where the model's output didn't parse as an integer (n_parsed
    captures that count separately).
    """
    diffs = []
    for p, g in zip(preds_str, golds):
        try:
            diffs.append(abs(int(p) - int(g)))
        except ValueError:
            continue
    if not diffs:
        return {"within_1": 0.0, "within_5": 0.0, "within_10": 0.0,
                "mean_abs_err": None, "median_abs_err": None, "n_parsed": 0}
    diffs_sorted = sorted(diffs)
    return {
        "within_1":  sum(1 for d in diffs if d <= 1) / len(diffs),
        "within_5":  sum(1 for d in diffs if d <= 5) / len(diffs),
        "within_10": sum(1 for d in diffs if d <= 10) / len(diffs),
        "mean_abs_err":   sum(diffs) / len(diffs),
        "median_abs_err": diffs_sorted[len(diffs_sorted) // 2],
        "n_parsed": len(diffs),
    }


def _emotion_confusion_matrix(aligned: list) -> dict:
    """Return a 4×4 confusion matrix gold→pred over Q1..Q4."""
    quads = ("Q1", "Q2", "Q3", "Q4")
    quad_set = set(quads)
    # rows: gold, cols: predicted option label
    matrix = {g: {p: 0 for p in quads} for g in quads}
    n_off_grid = 0
    for b, p in aligned:
        gold = b.get("gold")
        if gold not in quad_set:
            continue
        parsed = p.get("parsed_answer", "")
        if parsed in {"0", "1", "2", "3"}:
            try:
                pred_label = b["options"][int(parsed)]
            except (KeyError, IndexError, ValueError):
                pred_label = None
        else:
            pred_label = None
        if pred_label in quad_set:
            matrix[gold][pred_label] += 1
        else:
            n_off_grid += 1
    return {"matrix": matrix, "n_off_grid": n_off_grid}


def _score_error_detection(aligned: list) -> dict:
    """F1 per record, macro-averaged across the 5 categories."""
    per_record_f1: list[float] = []
    by_cat: dict[str, list[float]] = defaultdict(list)
    for b, p in aligned:
        gold = set(b.get("gold_bars") or [])
        pred = set(parse_bar_list(p.get("parsed_answer", "") or p.get("raw_output", "")))
        f1 = error_detection_f1(pred=pred, gold=gold)
        per_record_f1.append(f1)
        by_cat[b.get("category", "unknown")].append(f1)

    per_cat_f1 = {c: (sum(v) / len(v) if v else 0.0) for c, v in by_cat.items()}
    macro_f1 = sum(per_cat_f1.values()) / len(per_cat_f1) if per_cat_f1 else 0.0
    return {
        "task": "error_detection",
        "n": len(aligned),
        "macro_f1": macro_f1,
        "mean_f1": sum(per_record_f1) / len(per_record_f1) if per_record_f1 else 0.0,
        "per_category_f1": per_cat_f1,
    }


def _aggregate(per_task: dict[str, dict]) -> dict:
    """Macro- and weighted-average over tasks.

    For ``bar_sequencing`` the per-task value is the mean penalised Kendall-tau
    score (already in [0, 1]). For all other tasks it is accuracy. We treat
    both as scores in [0, 1] when averaging.
    """
    pairs = []
    for name, summary in per_task.items():
        score = summary.get(
            "accuracy",
            summary.get("score", summary.get("macro_f1", 0.0)),
        )
        n = summary.get("n", 0)
        pairs.append((name, score, n))
    if not pairs:
        return {"macro_avg": 0.0, "weighted_avg": 0.0}
    macro = sum(s for _, s, _ in pairs) / len(pairs)
    total_n = sum(n for _, _, n in pairs)
    weighted = (
        sum(s * n for _, s, n in pairs) / total_n if total_n else 0.0
    )
    return {"macro_avg": macro, "weighted_avg": weighted}


@hydra.main(config_path="../../configs", config_name="evaluate/understanding", version_base=None)
def main(cfg: DictConfig) -> int:
    print(OmegaConf.to_yaml(cfg))

    bench_path = Path(cfg.bench_path).expanduser().resolve()
    preds_dir = Path(cfg.predictions_dir).expanduser().resolve()
    if not bench_path.exists():
        raise FileNotFoundError(f"bench not found: {bench_path}")
    if not preds_dir.exists():
        raise FileNotFoundError(f"predictions dir not found: {preds_dir}")

    bench = _load_jsonl(bench_path)
    by_task: dict[str, list[dict]] = {}
    for r in bench:
        by_task.setdefault(r["task"], []).append(r)

    per_task: dict[str, dict] = {}
    for task in sorted(tasks.TASKS.keys()):
        pred_path = preds_dir / f"{task}.jsonl"
        if not pred_path.exists():
            per_task[task] = {"task": task, "n": 0, "missing": True}
            continue
        preds = _load_jsonl(pred_path)
        per_task[task] = _score_task(task, by_task.get(task, []), preds)

    overall = _aggregate({k: v for k, v in per_task.items() if not v.get("missing")})
    summary = {
        "model": str(cfg.model_id),
        "bench_path": str(bench_path),
        "predictions_dir": str(preds_dir),
        "tasks": per_task,
        "overall": overall,
    }

    out_path = Path(cfg.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[eval-und] wrote {out_path}")
    for name, t in per_task.items():
        if t.get("missing"):
            print(f"  {name:24s} (missing predictions)")
        elif "score" in t:
            print(f"  {name:24s} score={t['score']:.3f}  n={t['n']}  n_valid={t['n_valid']}")
        else:
            print(f"  {name:24s} acc={t['accuracy']:.3f}  n={t['n']}  n_parsed={t['n_parsed']}")
    print(f"  overall.macro_avg    = {overall['macro_avg']:.3f}")
    print(f"  overall.weighted_avg = {overall['weighted_avg']:.3f}")
    return 0


if __name__ == "__main__":
    main()
