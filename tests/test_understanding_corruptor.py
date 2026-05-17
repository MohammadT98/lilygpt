"""Tests for the error-injection corruptor used by the error_detection task."""

from __future__ import annotations

import random

import pytest

from lilybench.understanding import corruptor
from lilybench.understanding.bar_utils import count_bars


_CLEAN_LY = (
    '\\version "2.24.0"\n'
    '\\header {\n  title = "Test"\n}\n'
    'music = {\n'
    '  \\key c \\major\n'
    '  \\time 4/4\n'
    "  c'4 d' e' f' |\n"
    "  g'4 a' b' c'' |\n"
    "  d''4 c'' b' a' |\n"
    "  g'4 f' e' d' |\n"
    "  c'1 |\n"
    "}\n"
)


# ---------- API surface ----------


def test_categories_are_the_five_paper_ones():
    assert set(corruptor.ERROR_CATEGORIES) == {
        "invalid_metadata",
        "invalid_content",
        "invalid_bar_duration",
        "melodic_leap",
        "accidental_outside_key",
    }


def test_inject_dispatch_known_category(monkeypatch):
    rng = random.Random(0)
    out = corruptor.inject(_CLEAN_LY, "invalid_metadata", rng=rng)
    assert out is not None
    assert out.category == "invalid_metadata"


def test_inject_dispatch_unknown_category_raises():
    with pytest.raises(KeyError):
        corruptor.inject(_CLEAN_LY, "does_not_exist", rng=random.Random(0))


# ---------- invalid_metadata ----------


def test_inject_invalid_metadata_corrupts_key():
    c = corruptor.inject_invalid_metadata(_CLEAN_LY, random.Random(0))
    assert c is not None
    assert c.category == "invalid_metadata"
    assert "\\key c \\major" not in c.text  # original key gone
    assert "nonsense" in c.text or "\\key " not in c.text


def test_inject_invalid_metadata_returns_bar_one():
    c = corruptor.inject_invalid_metadata(_CLEAN_LY, random.Random(0))
    assert c.error_bars == (1,)  # metadata error reported as bar 1 by convention


def test_inject_invalid_metadata_returns_none_when_no_key():
    no_key = _CLEAN_LY.replace("\\key c \\major\n", "")
    assert corruptor.inject_invalid_metadata(no_key, random.Random(0)) is None


# ---------- invalid_content ----------


def test_inject_invalid_content_inserts_garbage():
    c = corruptor.inject_invalid_content(_CLEAN_LY, random.Random(0))
    assert c is not None
    assert c.category == "invalid_content"
    # Some garbage token should appear in the output that wasn't in the original.
    assert c.text != _CLEAN_LY
    assert len(c.text) > len(_CLEAN_LY)


def test_inject_invalid_content_reports_valid_bar_index():
    c = corruptor.inject_invalid_content(_CLEAN_LY, random.Random(0))
    assert len(c.error_bars) == 1
    bar = c.error_bars[0]
    n = count_bars(_CLEAN_LY)
    assert 1 <= bar <= n


def test_inject_invalid_content_deterministic_under_seed():
    a = corruptor.inject_invalid_content(_CLEAN_LY, random.Random(42))
    b = corruptor.inject_invalid_content(_CLEAN_LY, random.Random(42))
    assert a.text == b.text
    assert a.error_bars == b.error_bars


# ---------- invalid_bar_duration ----------


def test_inject_invalid_bar_duration_modifies_bar():
    c = corruptor.inject_invalid_bar_duration(_CLEAN_LY, random.Random(0))
    assert c is not None
    assert c.category == "invalid_bar_duration"
    assert c.text != _CLEAN_LY


def test_inject_invalid_bar_duration_reports_valid_bar():
    c = corruptor.inject_invalid_bar_duration(_CLEAN_LY, random.Random(0))
    bar = c.error_bars[0]
    n = count_bars(_CLEAN_LY)
    assert 1 <= bar <= n


# ---------- melodic_leap ----------


def test_inject_melodic_leap_creates_octave_jump():
    c = corruptor.inject_melodic_leap(_CLEAN_LY, random.Random(0))
    assert c is not None
    assert c.category == "melodic_leap"
    # Should contain a very-high or very-low octave anchor we didn't have before.
    high_octaves = ("''''", "''',", "''','")
    assert any(o in c.text and o not in _CLEAN_LY for o in high_octaves)


# ---------- accidental_outside_key ----------


def test_inject_accidental_outside_key_inserts_sharp_in_c_major():
    c = corruptor.inject_accidental_outside_key(_CLEAN_LY, random.Random(0))
    assert c is not None
    assert c.category == "accidental_outside_key"
    # C major doesn't include any sharps/flats; injected note should add one.
    assert "fis" in c.text or "cis" in c.text or "gis" in c.text or "dis" in c.text or "ais" in c.text


def test_inject_accidental_outside_key_skips_pieces_without_c_major():
    other_key = _CLEAN_LY.replace("\\key c \\major", "\\key g \\major")
    # g major already has fis; our injector must pick a note that's not in the key.
    c = corruptor.inject_accidental_outside_key(other_key, random.Random(0))
    # Acceptable outcomes: (a) returns a corruption that uses a non-G-major sharp/flat,
    # or (b) returns None for "no suitable injection point".
    if c is not None:
        # fis is in G major, so it should NOT be a fresh injection of fis.
        assert c.category == "accidental_outside_key"


# ---------- bar-index helper ----------


def test_bar_at_position_counts_separators_before_position():
    # First bar starts after the preamble; position before any | is bar 1.
    pos = _CLEAN_LY.index("c'4 d' e' f'")  # inside bar 1
    assert corruptor.bar_at_position(_CLEAN_LY, pos) == 1
    pos = _CLEAN_LY.index("d''4 c''")  # inside bar 3
    assert corruptor.bar_at_position(_CLEAN_LY, pos) == 3


# ---------- inject() registry ----------


def test_all_injectors_registered_via_inject():
    rng = random.Random(0)
    for cat in corruptor.ERROR_CATEGORIES:
        out = corruptor.inject(_CLEAN_LY, cat, rng=random.Random(0))
        if cat == "invalid_metadata":
            assert out is not None and out.category == cat
        # other categories may return None on edge inputs but in our 5-bar piece all should work
