"""Tests for the music-understanding bar-splitting helpers.

We use a regex over the LilyPond text to keep the build offline (no LilyPond
compilation needed). The expected behaviour is intentionally permissive: count
the ``|`` bar separators inside the first music block, ignoring those that
appear in the ``\\header`` / ``\\paper`` preamble.
"""

from __future__ import annotations

import pytest

from lilybench.understanding.bar_utils import count_bars, split_bars


def _ly(body: str) -> str:
    return (
        '\\version "2.24.0"\n'
        '\\header {\n  title = "Test"\n}\n'
        '\\language "nederlands"\n'
        + body
    )


def test_count_bars_simple():
    body = (
        "music = {\n"
        "  c'4 d' e' f' |\n"
        "  g'2 a'2 |\n"
        "  b'1 |\n"
        "}\n"
    )
    assert count_bars(_ly(body)) == 3


def test_split_bars_returns_three_segments():
    body = (
        "music = {\n"
        "  c'4 d' e' f' |\n"
        "  g'2 a'2 |\n"
        "  b'1 |\n"
        "}\n"
    )
    bars = split_bars(_ly(body))
    assert len(bars) == 3
    assert all(b.strip() for b in bars)
    assert "c'4 d' e' f'" in bars[0]
    assert "g'2 a'2" in bars[1]
    assert "b'1" in bars[2]


def test_count_bars_zero_when_no_separators():
    body = "music = {\n  c'4 d' e' f'\n}\n"
    assert count_bars(_ly(body)) == 0


def test_count_bars_ignores_header_pipe_characters():
    ly = (
        '\\version "2.24.0"\n'
        '\\header {\n  title = "Has | pipe in title"\n}\n'
        "music = {\n  c'4 d' | e' f' |\n}\n"
    )
    assert count_bars(ly) == 2


def test_count_bars_handles_double_bar_separator():
    body = (
        "music = {\n"
        "  c'4 d' e' f' |\n"
        '  g\'2 a\'2 \\bar "||"\n'
        "  b'1 |\n"
        "}\n"
    )
    n = count_bars(_ly(body))
    assert n == 2


def test_split_bars_strips_terminal_partial_after_last_pipe():
    body = (
        "music = {\n"
        "  c'4 d' e' f' |\n"
        "  g'2 a'2 |\n"
        "  trailing without pipe\n"
        "}\n"
    )
    bars = split_bars(_ly(body))
    assert len(bars) == 2


def test_count_bars_empty_text_zero():
    assert count_bars("") == 0


def test_count_bars_realistic_voice_block():
    body = (
        "violinoI = {\n"
        "  \\key g \\major\n"
        "  c'4 d' e' f' |\n"
        "  g'2 a'2 |\n"
        "  b'1 |\n"
        "  c''4 b' a' g' |\n"
        "}\n"
        "violinoII = {\n"
        "  \\key g \\major\n"
        "  e'4 f' g' a' |\n"
        "  b'2 c''2 |\n"
        "}\n"
    )
    n = count_bars(_ly(body))
    assert n >= 4


@pytest.mark.parametrize(
    "body,expected",
    [
        ("music = {\n  c'4 |\n}\n", 1),
        ("music = {\n  c'4 | d'4 |\n}\n", 2),
        ("music = {\n  c'4 | d'4 | e'4 | f'4 |\n}\n", 4),
    ],
)
def test_count_bars_parametrized(body, expected):
    assert count_bars(_ly(body)) == expected
