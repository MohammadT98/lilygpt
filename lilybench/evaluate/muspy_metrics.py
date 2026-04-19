"""Standard symbolic-music metrics from `muspy`, namespaced with a `muspy_` prefix.

Loaded from the MIDI file that the LilyPond binary already produces during
``text_midi`` evaluation, so no LilyPond → muspy converter is needed. Drum
metrics (`drum_in_pattern_rate`, `drum_pattern_consistency`) are intentionally
excluded — the corpus is solo / small ensemble, never percussion (see
``notelog.md`` §1.1).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import muspy


EXPECTED_KEYS: tuple[str, ...] = (
    "muspy_pitch_range",
    "muspy_n_pitches_used",
    "muspy_n_pitch_classes_used",
    "muspy_pitch_entropy",
    "muspy_pitch_class_entropy",
    "muspy_scale_consistency",
    "muspy_pitch_in_scale_rate",
    "muspy_polyphony",
    "muspy_polyphony_rate",
    "muspy_empty_beat_rate",
    "muspy_empty_measure_rate",
    "muspy_groove_consistency",
)


def _safe(fn: Callable, *args) -> float | None:
    try:
        result = fn(*args)
    except Exception:
        return None
    if result is None:
        return None
    if isinstance(result, float) and math.isnan(result):
        return None
    return float(result)


def _primary_key(music: muspy.Music) -> tuple[int, str] | None:
    for ks in music.key_signatures:
        if ks.root is not None and ks.mode is not None:
            return ks.root, ks.mode
    return None


def _all_none() -> dict[str, float | None]:
    return {k: None for k in EXPECTED_KEYS}


def compute_muspy_metrics(midi_path: Path) -> dict[str, float | None]:
    """Compute the 12 standard muspy metrics for a single MIDI file.

    Returns a dict with `EXPECTED_KEYS` always present; values are `None`
    when the file is missing/corrupt or the underlying muspy call fails (e.g.
    `pitch_in_scale_rate` requires a key signature).
    """
    midi_path = Path(midi_path)
    if not midi_path.exists():
        return _all_none()

    try:
        music = muspy.read_midi(str(midi_path))
    except Exception:
        return _all_none()

    resolution = music.resolution
    metrics: dict[str, float | None] = {
        "muspy_pitch_range": _safe(muspy.pitch_range, music),
        "muspy_n_pitches_used": _safe(muspy.n_pitches_used, music),
        "muspy_n_pitch_classes_used": _safe(muspy.n_pitch_classes_used, music),
        "muspy_pitch_entropy": _safe(muspy.pitch_entropy, music),
        "muspy_pitch_class_entropy": _safe(muspy.pitch_class_entropy, music),
        "muspy_scale_consistency": _safe(muspy.scale_consistency, music),
        "muspy_polyphony": _safe(muspy.polyphony, music),
        "muspy_polyphony_rate": _safe(muspy.polyphony_rate, music),
        "muspy_empty_beat_rate": _safe(muspy.empty_beat_rate, music),
        "muspy_empty_measure_rate": _safe(muspy.empty_measure_rate, music, resolution),
        "muspy_groove_consistency": _safe(muspy.groove_consistency, music, resolution),
    }

    key = _primary_key(music)
    if key is None:
        metrics["muspy_pitch_in_scale_rate"] = None
    else:
        root, mode = key
        metrics["muspy_pitch_in_scale_rate"] = _safe(
            muspy.pitch_in_scale_rate, music, root, mode
        )

    return metrics
