"""Tests for the music-understanding title parser.

The Mutopia metadata JSON does not carry titles; we extract them from the
``\\header { title = "..." }`` block in the .ly file itself.
"""

from __future__ import annotations

from lilybench.understanding.title_parser import extract_title


def test_extract_title_basic():
    ly = (
        '\\version "2.24.0"\n'
        '\\header {\n'
        '  title = "Sonata in C"\n'
        '  composer = "Bach"\n'
        '}\n'
    )
    assert extract_title(ly) == "Sonata in C"


def test_extract_title_missing_returns_none():
    ly = '\\version "2.24.0"\n\\header {\n  composer = "Bach"\n}\n'
    assert extract_title(ly) is None


def test_extract_title_empty_string_returns_none():
    ly = '\\version "2.24.0"\n\\header {\n  title = ""\n}\n'
    assert extract_title(ly) is None


def test_extract_title_whitespace_only_returns_none():
    ly = '\\version "2.24.0"\n\\header {\n  title = "   "\n}\n'
    assert extract_title(ly) is None


def test_extract_title_no_header_block_returns_none():
    ly = '\\version "2.24.0"\n\\relative c\' { c d e f }\n'
    assert extract_title(ly) is None


def test_extract_title_with_escaped_quote():
    ly = '\\header {\n  title = "He said \\"hi\\""\n}\n'
    assert extract_title(ly) == 'He said \\"hi\\"'


def test_extract_title_picks_title_field_not_subtitle():
    ly = (
        '\\header {\n'
        '  subtitle = "An Etude"\n'
        '  title = "Op. 25 No. 1"\n'
        '  composer = "Chopin"\n'
        '}\n'
    )
    assert extract_title(ly) == "Op. 25 No. 1"


def test_extract_title_field_indentation_insensitive():
    ly = '\\header{title="Compact"}\n'
    assert extract_title(ly) == "Compact"


def test_extract_title_first_header_block_wins():
    ly = (
        '\\header {\n  title = "First"\n}\n'
        '\\header {\n  title = "Second"\n}\n'
    )
    assert extract_title(ly) == "First"


def test_extract_title_handles_unicode():
    ly = '\\header {\n  title = "Étude Op. 10 — № 3"\n}\n'
    assert extract_title(ly) == "Étude Op. 10 — № 3"


def test_extract_title_strips_leading_trailing_whitespace_in_value():
    ly = '\\header {\n  title = "  Padded  "\n}\n'
    assert extract_title(ly) == "Padded"
