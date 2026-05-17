"""Tests for the generation-side metrics primitives."""

from __future__ import annotations

import numpy as np
import pytest

from lilybench.metrics.fmd import frechet_music_distance
from lilybench.metrics.js_similarity import (
    aggregate_descriptor_stats,
    js_descriptor_similarity,
)
from lilybench.metrics.muspy_descriptors import DESCRIPTOR_KEYS


def test_descriptor_keys_match_paper_three():
    assert DESCRIPTOR_KEYS == (
        "polyphony_rate", "groove_consistency", "scale_consistency",
    )


def test_aggregate_handles_missing_and_solo_metrics():
    per_file = {
        "a": {"polyphony_rate": 0.1, "groove_consistency": None, "scale_consistency": 0.7},
        "b": {"polyphony_rate": 0.2, "groove_consistency": 0.5, "scale_consistency": 0.8},
        "c": {"polyphony_rate": None, "groove_consistency": 0.6, "scale_consistency": 0.9},
    }
    agg = aggregate_descriptor_stats(per_file)
    assert agg["polyphony_rate"]["n"] == 2
    assert agg["groove_consistency"]["n"] == 2
    assert agg["scale_consistency"]["n"] == 3
    # std == None when only one value is available, but here all >= 2.
    assert agg["polyphony_rate"]["std"] is not None


def test_js_self_similarity_close_to_100():
    agg = {
        "polyphony_rate": {"mean": 0.4, "std": 0.05, "n": 50},
        "groove_consistency": {"mean": 0.6, "std": 0.04, "n": 50},
        "scale_consistency": {"mean": 0.8, "std": 0.03, "n": 50},
    }
    sim = js_descriptor_similarity(agg, agg)
    assert sim is not None
    assert sim == pytest.approx(100.0, rel=1e-2)


def test_js_drops_to_lower_when_distributions_differ():
    a = {
        "polyphony_rate": {"mean": 0.4, "std": 0.05, "n": 50},
        "groove_consistency": {"mean": 0.6, "std": 0.04, "n": 50},
        "scale_consistency": {"mean": 0.8, "std": 0.03, "n": 50},
    }
    b = {
        "polyphony_rate": {"mean": 0.9, "std": 0.05, "n": 50},
        "groove_consistency": {"mean": 0.1, "std": 0.04, "n": 50},
        "scale_consistency": {"mean": 0.2, "std": 0.03, "n": 50},
    }
    # JS-similarity floors at 100 * exp(-2 * ln(2)) = 25 when distributions
    # are maximally disjoint; identical distributions give 100. We only check
    # that highly-shifted distributions land well below the identical case.
    assert js_descriptor_similarity(a, b) < 50.0


def test_fmd_self_distance_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 8))
    assert frechet_music_distance(x, x) == pytest.approx(0.0, abs=1e-6)


def test_fmd_positive_for_shifted_distributions():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.0, size=(50, 8))
    y = rng.normal(loc=5.0, size=(50, 8))
    assert frechet_music_distance(x, y) > 1.0


def test_fmd_requires_at_least_two_documents():
    x = np.zeros((1, 4))
    y = np.zeros((4, 4))
    with pytest.raises(ValueError):
        frechet_music_distance(x, y)
