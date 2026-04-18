"""Shared fixtures for the lilybench pytest suite."""

from __future__ import annotations

import pytest


_MINIMAL_VARIABILI_END = "% === END INCLUDE: variabili.ly ==="


@pytest.fixture
def preprocessed_ly() -> str:
    """A synthetic preprocessed LilyPond file with a realistic prelude + body.

    Mirrors the structure produced by ``data/bmdataset/preprocessed/*.ly``:
    ``\\version`` → ``\\header`` → ``\\language`` → variabili block (closed by the
    end-include marker) → two top-level music variables.
    """
    return (
        '\\version "2.24.0"\n'
        "\\header {\n"
        '  title = "Synthetic"\n'
        '  composer = "Tester"\n'
        "}\n"
        '\\language "nederlands"\n'
        "% === BEGIN INCLUDE: variabili.ly ===\n"
        "su = \\markup { \\italic su }\n"
        "giu = \\markup { \\italic giu }\n"
        "tr = \\markup { \\italic tr }\n"
        f"{_MINIMAL_VARIABILI_END}\n"
        "violinoI = {\n"
        "  \\key g \\major\n"
        "  c'4 d' e' f' |\n"
        "  g'2 a'2 |\n"
        "}\n"
        "violinoII = {\n"
        "  \\key g \\major\n"
        "  e'4 f' g' a' |\n"
        "  b'2 c''2 |\n"
        "}\n"
    )


@pytest.fixture
def sample_metadata() -> dict:
    """Mini metadata.json matching the bmdataset schema."""
    return {
        "charpentier_lauda_sion_h_268_egredimini_h_280": {
            "composer": "Charpentier",
            "period": "Late Baroque",
            "musical_form": "motet",
            "midi_instruments": ["violin", "viola", "cello", "flute"],
        },
        "vivaldi_rv_589_gloria": {
            "composer": "Vivaldi",
            "period": "Late Baroque",
            "musical_form": ["mass", "gloria"],
            "midi_instruments": ["violin", "viola", "cello", "organ"],
        },
    }
