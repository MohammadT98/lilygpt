"""Scoring functions for the music-understanding benchmark.

Two metrics:

* ``accuracy(preds, golds)`` — exact-match fraction. Used by the bar-count
  task and all 4-way multiple-choice tasks (after the index of the predicted
  option has been extracted).

* ``bar_sequencing_score(pred, gold)`` — penalised Kendall-tau. The model
  outputs a digit sequence (e.g. ``"2031"``); we parse it, validate that it
  is a permutation of ``[0..n-1]`` matching ``gold``'s length, compute
  Kendall tau, map ``[-1, 1] -> [0, 1]``, and multiply by a completeness
  penalty. Invalid outputs score 0.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from scipy.stats import kendalltau


def accuracy(preds: Sequence, golds: Sequence) -> float:
    """Fraction of ``preds[i] == golds[i]`` (string-coerced).

    Returns 0.0 for an empty input. Raises ``ValueError`` on length mismatch.
    """
    if len(preds) != len(golds):
        raise ValueError(f"length mismatch: preds={len(preds)} golds={len(golds)}")
    n = len(preds)
    if n == 0:
        return 0.0
    correct = sum(1 for p, g in zip(preds, golds) if str(p) == str(g))
    return correct / n


_DIGIT_RE = re.compile(r"\d")


def parse_digit_sequence(text: str) -> str:
    """Strip everything non-digit from ``text`` and return the digit string.

    Used to recover ``"0312"`` from outputs like ``"the answer is 0 3 1 2"``.
    """
    if not text:
        return ""
    return "".join(_DIGIT_RE.findall(text))


def bar_sequencing_score(*, pred: str, gold: str) -> float:
    """Penalised Kendall-tau in ``[0, 1]``; 0 for invalid outputs.

    ``gold`` is the canonical digit string (e.g. ``"0123"``). ``pred`` is the
    raw model output. We coerce both to digit sequences, then require:

    * ``len(parsed_pred) == len(gold)``
    * No duplicate digits
    * All digits in ``[0, n)`` where ``n == len(gold)``

    On success, score = ``(tau + 1) / 2 * min(1, len(parsed) / len(gold))``.
    The completeness factor is 1.0 when the validity guard passes (lengths
    are equal); it is retained to make the formula match the paper and to
    leave room for a more permissive parser in future iterations.

    For ``n == 1`` Kendall tau is undefined; we treat the trivially-correct
    case as 1.0 and any other as 0.0.
    """
    n = len(gold)
    parsed = parse_digit_sequence(pred)
    if len(parsed) != n:
        return 0.0
    if len(set(parsed)) != n:
        return 0.0
    if any(int(c) >= n or int(c) < 0 for c in parsed):
        return 0.0
    gold_indices = [int(c) for c in gold]
    pred_indices = [int(c) for c in parsed]
    if n == 1:
        return 1.0 if pred_indices == gold_indices else 0.0
    tau, _ = kendalltau(gold_indices, pred_indices)
    if tau is None:
        return 0.0
    scaled = (float(tau) + 1.0) / 2.0
    completeness = min(1.0, len(parsed) / n)
    return scaled * completeness
