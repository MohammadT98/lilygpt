"""The three MusPy descriptors used by LilyBench's JS-similarity metric.

Following §3 of the paper, JS-similarity is computed over Gaussian fits
to three MusPy descriptors:

* ``polyphony_rate``
* ``groove_consistency``
* ``scale_consistency``

The descriptors are extracted from the MIDI file produced by
``lilypond``; outputs that fail to compile are excluded from the JS
calculation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import muspy


DESCRIPTOR_KEYS = ("polyphony_rate", "groove_consistency", "scale_consistency")


def _safe(fn: Callable, *args) -> float | None:
    try:
        v = fn(*args)
    except Exception:
        return None
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def compute_muspy_descriptors(midi_path: str | Path) -> dict[str, float | None]:
    """Compute the three descriptors for a single MIDI file."""
    midi_path = Path(midi_path)
    if not midi_path.exists():
        return {k: None for k in DESCRIPTOR_KEYS}
    try:
        music = muspy.read_midi(str(midi_path))
    except Exception:
        return {k: None for k in DESCRIPTOR_KEYS}
    resolution = music.resolution
    return {
        "polyphony_rate": _safe(muspy.polyphony_rate, music),
        "groove_consistency": _safe(muspy.groove_consistency, music, resolution),
        "scale_consistency": _safe(muspy.scale_consistency, music),
    }
