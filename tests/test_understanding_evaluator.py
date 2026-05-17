"""End-to-end smoke test for the understanding evaluator.

Builds a synthetic bench + matching predictions JSONL on disk and calls the
pure scoring helpers directly. The Hydra ``main`` entrypoint is exercised
implicitly by the per-task helpers being called the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_score_task_bar_count_emits_tolerance():
    bench = [
        _bench_record("bar_count", "b_0", gold="32"),
        _bench_record("bar_count", "b_1", gold="100"),
        _bench_record("bar_count", "b_2", gold="50"),
    ]
    preds = [_pred("b_0", "32"), _pred("b_1", "97"), _pred("b_2", "abc")]
    s = _score_task("bar_count", bench, preds)
    assert "tolerance" in s
    tol = s["tolerance"]
    # exact=1 (b_0); within_5=2 (b_0, b_1 off by 3); within_10=2
    assert tol["within_1"] == pytest.approx(0.5)   # b_0 exact, b_1 off by 3
    assert tol["within_5"] == pytest.approx(1.0)
    assert tol["within_10"] == pytest.approx(1.0)
    assert tol["n_parsed"] == 2  # b_2 didn't parse


def test_score_task_emotion_emits_confusion_matrix():
    bench = [
        _bench_record(
            "emotion_recognition", "e_0",
            gold="Q1", gold_index=0, options=["Q1", "Q2", "Q3", "Q4"],
        ),
        _bench_record(
            "emotion_recognition", "e_1",
            gold="Q2", gold_index=1, options=["Q1", "Q2", "Q3", "Q4"],
        ),
        _bench_record(
            "emotion_recognition", "e_2",
            gold="Q3", gold_index=2, options=["Q1", "Q2", "Q3", "Q4"],
        ),
    ]
    preds = [
        _pred("e_0", "0"),  # correct
        _pred("e_1", "0"),  # predicts Q1 instead of Q2
        _pred("e_2", "2"),  # correct
    ]
    s = _score_task("emotion_recognition", bench, preds)
    assert "confusion" in s
    matrix = s["confusion"]["matrix"]
    assert matrix["Q1"]["Q1"] == 1   # e_0 correct
    assert matrix["Q2"]["Q1"] == 1   # e_1 mispredicted
    assert matrix["Q3"]["Q3"] == 1   # e_2 correct
    assert s["confusion"]["n_off_grid"] == 0


def test_score_task_error_detection_macro_f1():
    bench = [
        _bench_record("error_detection", "err_0", gold_bars=[3], category="invalid_metadata"),
        _bench_record("error_detection", "err_1", gold_bars=[5], category="invalid_metadata"),
        _bench_record("error_detection", "err_2", gold_bars=[2], category="melodic_leap"),
    ]
    preds = [
        _pred("err_0", "3"),       # F1=1
        _pred("err_1", "1 5"),     # P=0.5 R=1 F1=2/3
        _pred("err_2", "none"),    # F1=0
    ]
    s = _score_task("error_detection", bench, preds)
    # per-category: invalid_metadata mean = (1 + 2/3)/2 = 5/6; melodic_leap = 0
    # macro = (5/6 + 0)/2 = 5/12
    assert s["macro_f1"] == pytest.approx(5 / 12)
    assert s["n"] == 3
    assert s["per_category_f1"]["invalid_metadata"] == pytest.approx(5 / 6)
    assert s["per_category_f1"]["melodic_leap"] == 0.0
