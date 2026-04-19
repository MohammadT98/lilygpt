"""Tests for the muspy metric helper used by the text+MIDI evaluator."""

from __future__ import annotations

from pathlib import Path

import muspy
import pytest

from lilybench.evaluate.muspy_metrics import (
    EXPECTED_KEYS,
    compute_muspy_metrics,
)


@pytest.fixture
def c_major_midi(tmp_path: Path) -> Path:
    """Four ascending C-major notes (C4, D4, E4, G4) with a key signature."""
    music = muspy.Music(
        resolution=24,
        tempos=[muspy.Tempo(time=0, qpm=120)],
        key_signatures=[muspy.KeySignature(time=0, root=0, mode="major")],
        time_signatures=[muspy.TimeSignature(time=0, numerator=4, denominator=4)],
        tracks=[
            muspy.Track(
                program=0,
                is_drum=False,
                name="m",
                notes=[
                    muspy.Note(time=0, pitch=60, duration=24, velocity=80),
                    muspy.Note(time=24, pitch=62, duration=24, velocity=80),
                    muspy.Note(time=48, pitch=64, duration=24, velocity=80),
                    muspy.Note(time=72, pitch=67, duration=24, velocity=80),
                ],
            )
        ],
    )
    path = tmp_path / "c_major.midi"
    muspy.write_midi(str(path), music)
    return path


@pytest.fixture
def no_key_midi(tmp_path: Path) -> Path:
    """Same notes as ``c_major_midi`` but with no key signature recorded."""
    music = muspy.Music(
        resolution=24,
        tempos=[muspy.Tempo(time=0, qpm=120)],
        time_signatures=[muspy.TimeSignature(time=0, numerator=4, denominator=4)],
        tracks=[
            muspy.Track(
                program=0,
                is_drum=False,
                name="m",
                notes=[
                    muspy.Note(time=0, pitch=60, duration=24, velocity=80),
                    muspy.Note(time=24, pitch=62, duration=24, velocity=80),
                    muspy.Note(time=48, pitch=64, duration=24, velocity=80),
                    muspy.Note(time=72, pitch=67, duration=24, velocity=80),
                ],
            )
        ],
    )
    path = tmp_path / "no_key.midi"
    muspy.write_midi(str(path), music)
    return path


def test_returns_all_expected_keys(c_major_midi: Path) -> None:
    metrics = compute_muspy_metrics(c_major_midi)
    assert set(metrics.keys()) == set(EXPECTED_KEYS)


def test_pitch_range_matches_synthetic_span(c_major_midi: Path) -> None:
    metrics = compute_muspy_metrics(c_major_midi)
    assert metrics["muspy_pitch_range"] == 7  # G4 (67) - C4 (60)
    assert metrics["muspy_n_pitches_used"] == 4
    assert metrics["muspy_n_pitch_classes_used"] == 4


def test_in_key_rate_one_when_all_notes_in_scale(c_major_midi: Path) -> None:
    metrics = compute_muspy_metrics(c_major_midi)
    assert metrics["muspy_pitch_in_scale_rate"] == 1.0
    assert metrics["muspy_scale_consistency"] == 1.0


def test_in_scale_rate_none_when_key_missing(no_key_midi: Path) -> None:
    metrics = compute_muspy_metrics(no_key_midi)
    assert metrics["muspy_pitch_in_scale_rate"] is None
    # Other metrics should still compute
    assert metrics["muspy_pitch_range"] == 7
    assert metrics["muspy_n_pitches_used"] == 4


def test_missing_file_returns_all_none(tmp_path: Path) -> None:
    metrics = compute_muspy_metrics(tmp_path / "does_not_exist.midi")
    assert set(metrics.keys()) == set(EXPECTED_KEYS)
    assert all(v is None for v in metrics.values())


def test_corrupted_midi_returns_all_none(tmp_path: Path) -> None:
    bad = tmp_path / "broken.midi"
    bad.write_bytes(b"not a midi file at all")
    metrics = compute_muspy_metrics(bad)
    assert set(metrics.keys()) == set(EXPECTED_KEYS)
    assert all(v is None for v in metrics.values())
