"""Tests for the LilyPond metadata extractors used by the bench builder."""

from __future__ import annotations

from lilybench.understanding.score_metadata import (
    extract_key,
    extract_meter,
    extract_note_length,
    mask_field,
)


def test_extract_key_major():
    ly = '\\version "2.24.0"\nfoo = { \\key g \\major\nc4 d4 e4 f4 |\n}\n'
    assert extract_key(ly) == "g \\major"


def test_extract_key_minor():
    ly = "foo = { \\key fis \\minor\nfis4 g4 a4 b4 |\n}\n"
    assert extract_key(ly) == "fis \\minor"


def test_extract_key_missing_returns_none():
    ly = "foo = { c4 d4 e4 f4 |\n}\n"
    assert extract_key(ly) is None


def test_extract_meter_simple():
    ly = "foo = { \\time 4/4\nc4 d4 e4 f4 |\n}\n"
    assert extract_meter(ly) == "4/4"


def test_extract_meter_compound():
    ly = "foo = { \\time 6/8\nc8 d8 e8 f8 g8 a8 |\n}\n"
    assert extract_meter(ly) == "6/8"


def test_extract_meter_missing_returns_none():
    ly = "foo = { c4 d4 |\n}\n"
    assert extract_meter(ly) is None


def test_extract_note_length_from_meter_denominator():
    ly = "foo = { \\time 6/8\nc8 d8 e8 |\n}\n"
    assert extract_note_length(ly) == "8"


def test_extract_note_length_missing_returns_none():
    ly = "foo = { c4 |\n}\n"
    assert extract_note_length(ly) is None


def test_mask_field_replaces_key():
    ly = "foo = { \\key g \\major\nc4 d4 e4 f4 |\n}\n"
    out = mask_field(ly, "key")
    assert "\\key" not in out
    # ``\major`` may still appear if the line contained other backslash
    # directives; we require only that the literal "\key" form is gone.
    assert "c4 d4 e4 f4 |" in out


def test_mask_field_replaces_meter():
    ly = "foo = { \\time 4/4\nc4 d4 e4 f4 |\n}\n"
    out = mask_field(ly, "meter")
    assert "\\time" not in out


def test_mask_field_unknown_field_raises():
    import pytest
    with pytest.raises(ValueError):
        mask_field("x", "no-such-field")


def test_mask_field_idempotent_when_field_absent():
    ly = "foo = { c4 d4 |\n}\n"
    out = mask_field(ly, "key")
    assert out == ly
