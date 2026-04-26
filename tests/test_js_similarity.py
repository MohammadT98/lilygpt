from __future__ import annotations

import json
import math
from pathlib import Path

import muspy
import pytest

from lilybench.evaluate.js_similarity import (
    JS_METRICS,
    _js_divergence_gaussian,
    aggregate,
    compute_js_similarity,
    load_reference_aggregate,
)


def test_aggregate_basic() -> None:
    per_file = {
        "a.mid": {"x": 1.0, "y": 10.0},
        "b.mid": {"x": 2.0, "y": None},
        "c.mid": {"x": 3.0, "y": 20.0},
    }
    agg = aggregate(per_file)

    assert agg["x"]["n"] == 3
    assert agg["x"]["mean"] == pytest.approx(2.0)
    assert agg["x"]["std"] == pytest.approx(1.0)

    assert agg["y"]["n"] == 2
    assert agg["y"]["mean"] == pytest.approx(15.0)
    assert agg["y"]["std"] == pytest.approx(math.sqrt(50.0))


def test_js_identical_distributions() -> None:
    agg = {m: {"mean": 0.5, "std": 0.1, "n": 50} for m in JS_METRICS}
    sim = compute_js_similarity(agg, agg)
    assert sim == pytest.approx(100.0, abs=1e-3)


def test_js_far_distributions() -> None:
    model = {m: {"mean": 0.0, "std": 1.0, "n": 50} for m in JS_METRICS}
    ref = {m: {"mean": 10.0, "std": 1.0, "n": 50} for m in JS_METRICS}
    sim = compute_js_similarity(model, ref)
    assert sim is not None
    assert sim == pytest.approx(25.0, abs=0.5)


def test_js_similarity_strictly_decreases_with_distance() -> None:
    near_model = {m: {"mean": 0.5, "std": 0.1, "n": 50} for m in JS_METRICS}
    near_ref = {m: {"mean": 0.6, "std": 0.1, "n": 50} for m in JS_METRICS}
    far_model = {m: {"mean": 0.5, "std": 0.1, "n": 50} for m in JS_METRICS}
    far_ref = {m: {"mean": 1.0, "std": 0.1, "n": 50} for m in JS_METRICS}
    near_sim = compute_js_similarity(near_model, near_ref)
    far_sim = compute_js_similarity(far_model, far_ref)
    assert near_sim is not None and far_sim is not None
    assert near_sim > far_sim


def test_js_returns_none_on_missing_std() -> None:
    model = {m: {"mean": 0.5, "std": 0.1, "n": 50} for m in JS_METRICS}
    ref = {m: {"mean": 0.5, "std": None, "n": 50} for m in JS_METRICS}
    assert compute_js_similarity(model, ref) is None


def test_js_returns_none_on_zero_std() -> None:
    model = {m: {"mean": 0.5, "std": 0.0, "n": 50} for m in JS_METRICS}
    ref = {m: {"mean": 0.5, "std": 0.1, "n": 50} for m in JS_METRICS}
    assert compute_js_similarity(model, ref) is None


def test_load_reference_aggregate_roundtrip(tmp_path: Path) -> None:
    cache_path = tmp_path / "agg.json"
    payload = {
        "muspy_polyphony_rate": {"mean": 0.7, "std": 0.05, "n": 100},
        "muspy_groove_consistency": {"mean": 0.9, "std": 0.02, "n": 100},
        "muspy_scale_consistency": {"mean": 0.85, "std": 0.04, "n": 100},
    }
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_reference_aggregate(None, cache_path)
    assert loaded == payload


def _write_trivial_midi(path: Path, pitches: list[int]) -> None:
    music = muspy.Music(resolution=24)
    music.tempos.append(muspy.Tempo(time=0, qpm=120))
    track = muspy.Track(program=0, is_drum=False)
    t = 0
    for pitch in pitches:
        track.notes.append(muspy.Note(time=t, duration=24, pitch=pitch, velocity=80))
        t += 24
    music.tracks.append(track)
    muspy.write_midi(str(path), music)


def test_load_reference_aggregate_from_midis(tmp_path: Path) -> None:
    midi_dir = tmp_path / "ref_midis"
    midi_dir.mkdir()
    _write_trivial_midi(midi_dir / "a.mid", [60, 62, 64, 65, 67, 69, 71, 72])
    _write_trivial_midi(midi_dir / "b.mid", [60, 64, 67, 60, 64, 67, 60, 64])

    cache_path = tmp_path / "ref_agg.json"
    agg = load_reference_aggregate(midi_dir, cache_path)

    assert agg is not None
    for metric in JS_METRICS:
        assert metric in agg
        assert agg[metric]["n"] >= 1

    assert cache_path.exists()
    reloaded = json.loads(cache_path.read_text(encoding="utf-8"))
    assert reloaded == agg


def test_load_reference_aggregate_returns_none_when_unset() -> None:
    assert load_reference_aggregate(None, None) is None


def test_js_divergence_gaussian_zero_for_identical() -> None:
    js = _js_divergence_gaussian(0.0, 1.0, 0.0, 1.0)
    assert js == pytest.approx(0.0, abs=1e-6)


def test_js_divergence_gaussian_positive_for_different() -> None:
    js = _js_divergence_gaussian(0.0, 1.0, 5.0, 1.0)
    assert js > 0.1
