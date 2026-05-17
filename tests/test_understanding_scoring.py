"""Tests for music-understanding scoring functions.

Two metrics:
  * exact-match accuracy (bar count + all 4-way MC tasks)
  * bar-sequencing score = (kendall_tau + 1)/2 × completeness_penalty,
    or 0 if the parsed output is invalid (wrong length, duplicates,
    out-of-range indices, or unparseable).
"""

from __future__ import annotations

import math

import pytest

from lilybench.understanding.scoring import (
    accuracy,
    bar_sequencing_score,
    error_detection_f1,
    parse_bar_list,
    parse_digit_sequence,
)


def test_accuracy_empty_returns_zero():
    assert accuracy([], []) == 0.0


def test_accuracy_basic_fraction():
    preds = ["0", "1", "2", "0"]
    golds = ["0", "2", "2", "0"]
    assert accuracy(preds, golds) == pytest.approx(0.75)


def test_accuracy_int_and_str_compatible():
    assert accuracy([0, 1, 2], ["0", "1", "2"]) == pytest.approx(1.0)


def test_accuracy_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])


def test_parse_digit_sequence_strips_prose():
    assert parse_digit_sequence("answer is 0312") == "0312"
    assert parse_digit_sequence("  2031  ") == "2031"
    assert parse_digit_sequence("the order is 2 0 3 1") == "2031"


def test_parse_digit_sequence_empty_for_nondigits():
    assert parse_digit_sequence("no digits at all") == ""


def test_bar_sequencing_score_perfect():
    # gold permutation 0123 → predict 0123 → tau=1 → score = (1+1)/2 = 1.0
    assert bar_sequencing_score(pred="0123", gold="0123") == pytest.approx(1.0)


def test_bar_sequencing_score_full_reverse():
    # gold 0123, predicted 3210 → kendall tau = -1 → (-1+1)/2 = 0.0
    assert bar_sequencing_score(pred="3210", gold="0123") == pytest.approx(0.0)


def test_bar_sequencing_score_partial():
    # gold 0123, predict 0213 → one swap out of 6 pairs in agreement
    # kendalltau([0,1,2,3], [0,2,1,3]) = 0.666... -> (1+0.666)/2 = 0.833
    score = bar_sequencing_score(pred="0213", gold="0123")
    assert 0.7 < score < 0.95


def test_bar_sequencing_score_invalid_wrong_length():
    assert bar_sequencing_score(pred="01", gold="0123") == 0.0


def test_bar_sequencing_score_invalid_duplicates():
    assert bar_sequencing_score(pred="0011", gold="0123") == 0.0


def test_bar_sequencing_score_invalid_out_of_range():
    assert bar_sequencing_score(pred="0124", gold="0123") == 0.0


def test_bar_sequencing_score_invalid_too_long():
    assert bar_sequencing_score(pred="01234", gold="0123") == 0.0


def test_bar_sequencing_score_unparseable_returns_zero():
    assert bar_sequencing_score(pred="not a digit anywhere", gold="0123") == 0.0


def test_bar_sequencing_score_strips_prose_around_digits():
    # "the answer is 0123" should parse as 0123
    assert bar_sequencing_score(pred="the answer is 0123", gold="0123") == pytest.approx(1.0)


def test_bar_sequencing_score_single_element_perfect():
    # Kendall tau is undefined for n=1; we treat the trivially-correct case as 1.0.
    assert bar_sequencing_score(pred="0", gold="0") == pytest.approx(1.0)


def test_bar_sequencing_score_two_elements_swap():
    # gold "01" predicted "10" → tau=-1 → score 0
    assert bar_sequencing_score(pred="10", gold="01") == pytest.approx(0.0)
    assert bar_sequencing_score(pred="01", gold="01") == pytest.approx(1.0)


# ---------- error_detection ----------


def test_parse_bar_list_extracts_integers():
    assert parse_bar_list("3, 7, 12") == [3, 7, 12]
    assert parse_bar_list("3 7 12") == [3, 7, 12]
    assert parse_bar_list("bars 3 and 7") == [3, 7]
    assert parse_bar_list("12") == [12]
    assert parse_bar_list("") == []
    assert parse_bar_list("none") == []


def test_parse_bar_list_deduplicates():
    assert sorted(parse_bar_list("3 3 7 7 7")) == [3, 7]


def test_error_detection_f1_perfect():
    f1 = error_detection_f1(pred={3}, gold={3})
    assert f1 == pytest.approx(1.0)


def test_error_detection_f1_no_overlap():
    f1 = error_detection_f1(pred={2}, gold={5})
    assert f1 == 0.0


def test_error_detection_f1_partial_recall():
    f1 = error_detection_f1(pred={3}, gold={3, 5})
    # precision=1, recall=0.5, F1 = 2*1*0.5/1.5 = 0.6667
    assert f1 == pytest.approx(2 / 3)


def test_error_detection_f1_partial_precision():
    f1 = error_detection_f1(pred={3, 5}, gold={3})
    # precision=0.5, recall=1, F1 = 0.6667
    assert f1 == pytest.approx(2 / 3)


def test_error_detection_f1_empty_pred_empty_gold():
    # Both empty: defined as 1.0 (correctly identified "no errors").
    assert error_detection_f1(pred=set(), gold=set()) == 1.0


def test_error_detection_f1_pred_empty_gold_nonempty():
    assert error_detection_f1(pred=set(), gold={3}) == 0.0


def test_error_detection_f1_gold_empty_pred_nonempty():
    assert error_detection_f1(pred={3}, gold=set()) == 0.0
