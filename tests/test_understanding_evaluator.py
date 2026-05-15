"""End-to-end smoke test for the understanding evaluator.

Builds a synthetic bench + matching predictions JSONL on disk and calls the
pure scoring helpers directly. The Hydra ``main`` entrypoint is exercised
implicitly by the per-task helpers being called the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

from lilybench.evaluate.understanding import _aggregate, _score_task


def _bench_record(task, id_, **extra) -> dict:
    r = {"task": task, "id": id_}
    r.update(extra)
    return r


def _pred(id_, parsed) -> dict:
    return {"id": id_, "raw_output": parsed, "parsed_answer": parsed, "latency_ms": 0}


def test_score_task_mc_accuracy():
    bench = [
        _bench_record("composer_recognition", "c_0", gold_index=2),
        _bench_record("composer_recognition", "c_1", gold_index=0),
        _bench_record("composer_recognition", "c_2", gold_index=3),
    ]
    preds = [
        _pred("c_0", "2"),
        _pred("c_1", "0"),
        _pred("c_2", "1"),
    ]
    summary = _score_task("composer_recognition", bench, preds)
    assert summary["n"] == 3
    assert summary["accuracy"] == 2 / 3
    assert summary["n_parsed"] == 3


def test_score_task_bar_count_accuracy():
    bench = [
        _bench_record("bar_count", "b_0", gold="8"),
        _bench_record("bar_count", "b_1", gold="12"),
        _bench_record("bar_count", "b_2", gold="3"),
    ]
    preds = [_pred("b_0", "8"), _pred("b_1", "10"), _pred("b_2", "3")]
    summary = _score_task("bar_count", bench, preds)
    assert summary["accuracy"] == 2 / 3
    assert summary["n_parsed"] == 3


def test_score_task_bar_sequencing_uses_kendall():
    bench = [
        _bench_record("bar_sequencing", "s_0", gold="0123"),
        _bench_record("bar_sequencing", "s_1", gold="0123"),
    ]
    preds = [_pred("s_0", "0123"), _pred("s_1", "3210")]
    summary = _score_task("bar_sequencing", bench, preds)
    assert 0 <= summary["score"] <= 1
    # perfect + reverse → mean = (1 + 0)/2 = 0.5
    assert summary["score"] == 0.5
    assert summary["n_valid"] == 1


def test_score_task_skips_predictions_not_in_bench():
    bench = [_bench_record("composer_recognition", "c_0", gold_index=1)]
    preds = [_pred("c_0", "1"), _pred("c_phantom", "2")]
    summary = _score_task("composer_recognition", bench, preds)
    assert summary["n"] == 1
    assert summary["accuracy"] == 1.0


def test_aggregate_macro_and_weighted():
    per_task = {
        "bar_count": {"n": 100, "accuracy": 0.4},
        "composer_recognition": {"n": 96, "accuracy": 0.8},
        "bar_sequencing": {"n": 119, "score": 0.5},
    }
    agg = _aggregate(per_task)
    # macro = (0.4 + 0.8 + 0.5) / 3 ≈ 0.5667
    assert abs(agg["macro_avg"] - (0.4 + 0.8 + 0.5) / 3) < 1e-9
    # weighted = (0.4*100 + 0.8*96 + 0.5*119) / 315
    expected_w = (0.4 * 100 + 0.8 * 96 + 0.5 * 119) / (100 + 96 + 119)
    assert abs(agg["weighted_avg"] - expected_w) < 1e-9


def test_aggregate_empty_returns_zeros():
    assert _aggregate({}) == {"macro_avg": 0.0, "weighted_avg": 0.0}
