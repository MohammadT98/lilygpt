"""Scoring primitives used by the per-task understanding scorers.

Four primitives cover the ten tasks:

* :func:`accuracy` — exact-match over option indices (4-way MC).
* :func:`bar_count_tolerance` — tolerance breakdown for bar-count.
* :func:`bar_sequencing_score` — penalised Kendall-τ over a permutation.
* :func:`error_detection_f1` — F1 between predicted / gold bar sets.
"""

from __future__ import annotations

import re
from typing import Sequence


# --------------------------------------------------------------------- accuracy

def accuracy(preds: Sequence, golds: Sequence) -> float:
    """Fraction of ``str(preds[i]) == str(golds[i])`` matches."""
    if len(preds) != len(golds):
        raise ValueError(f"length mismatch: preds={len(preds)} golds={len(golds)}")
    if not preds:
        return 0.0
    return sum(1 for p, g in zip(preds, golds) if str(p) == str(g)) / len(preds)


# --------------------------------------------------------------------- bar count

def bar_count_tolerance(
    preds: Sequence[str], golds: Sequence[str]
) -> dict[str, float | int | None]:
    """Return tolerance bands + abs-error statistics for ``bar_count``."""
    diffs: list[int] = []
    for p, g in zip(preds, golds):
        try:
            diffs.append(abs(int(p) - int(g)))
        except (TypeError, ValueError):
            continue
    if not diffs:
        return {
            "within_1": 0.0, "within_5": 0.0, "within_10": 0.0,
            "mean_abs_err": None, "median_abs_err": None, "n_parsed": 0,
        }
    diffs_sorted = sorted(diffs)
    return {
        "within_1": sum(1 for d in diffs if d <= 1) / len(diffs),
        "within_5": sum(1 for d in diffs if d <= 5) / len(diffs),
        "within_10": sum(1 for d in diffs if d <= 10) / len(diffs),
        "mean_abs_err": sum(diffs) / len(diffs),
        "median_abs_err": diffs_sorted[len(diffs_sorted) // 2],
        "n_parsed": len(diffs),
    }


# --------------------------------------------------------------------- sequencing

_DIGIT_RE = re.compile(r"\d")


def parse_digit_sequence(text: str) -> str:
    if not text:
        return ""
    return "".join(_DIGIT_RE.findall(text))


def bar_sequencing_score(*, pred: str, gold: str) -> float:
    """Penalised Kendall-τ in ``[0, 1]``. Returns 0 for malformed outputs."""
    from scipy.stats import kendalltau

    n = len(gold)
    parsed = parse_digit_sequence(pred)
    if len(parsed) != n:
        return 0.0
    if len(set(parsed)) != n:
        return 0.0
    if any(int(c) >= n or int(c) < 0 for c in parsed):
        return 0.0
    gold_idx = [int(c) for c in gold]
    pred_idx = [int(c) for c in parsed]
    if n == 1:
        return 1.0 if pred_idx == gold_idx else 0.0
    tau, _ = kendalltau(gold_idx, pred_idx)
    if tau is None:
        return 0.0
    return (float(tau) + 1.0) / 2.0


# --------------------------------------------------------------------- errors

_INT_RE = re.compile(r"\d+")


def parse_bar_list(text: str) -> list[int]:
    """Extract integer bar indices from a model's free-form output."""
    if not text:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for m in _INT_RE.findall(text):
        v = int(m)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def error_detection_f1(*, pred: set[int], gold: set[int]) -> float:
    """F1 between predicted and gold bar sets (both empty → 1.0)."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall)
