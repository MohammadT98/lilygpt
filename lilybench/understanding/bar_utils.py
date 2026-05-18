"""Bar splitting + counting on raw LilyPond text.

Permissive regex-based parsing — keeps the bench builder offline (no LilyPond
compilation needed) at the cost of accepting some false positives that a
strict parser would reject. Compilation-based bar counting is available via
the existing muspy pipeline when needed.
"""

from __future__ import annotations

import re

# Strip the ``\header { ... }`` block before counting so that ``|`` characters
# inside titles or composer fields do not get counted as bar separators.
_HEADER_BLOCK_RE = re.compile(r"\\header\s*\{[^{}]*\}", re.DOTALL)

# Strip the ``\paper { ... }`` block for the same reason.
_PAPER_BLOCK_RE = re.compile(r"\\paper\s*\{[^{}]*\}", re.DOTALL)

# Strip ``\bar "..."`` directives — their quoted argument can contain ``|``
# characters (e.g. ``\bar "||"``, ``\bar "|."``) that should not count as bar
# separators.
_BAR_DIRECTIVE_RE = re.compile(r'\\bar\s*"[^"]*"')

# Strip the contents of any double-quoted string outside the header — covers
# ``\mark "|"``, lyric text, etc.
_QUOTED_STRING_RE = re.compile(r'"[^"\n]*"')


def _strip_preamble_blocks(ly_text: str) -> str:
    """Remove header/paper blocks and quoted-string ``|`` so they do not leak
    into the bar count."""
    text = _HEADER_BLOCK_RE.sub("", ly_text)
    text = _PAPER_BLOCK_RE.sub("", text)
    text = _BAR_DIRECTIVE_RE.sub("", text)
    text = _QUOTED_STRING_RE.sub('""', text)
    return text


def split_bars(ly_text: str) -> list[str]:
    """Return the bar segments delimited by ``|``.

    Each segment is the text between consecutive bar separators. Any trailing
    content after the last ``|`` is discarded as a partial / pickup measure.
    Empty segments are filtered out.
    """
    if not ly_text:
        return []
    cleaned = _strip_preamble_blocks(ly_text)
    parts = cleaned.split("|")
    # Drop trailing partial (no closing ``|``).
    if len(parts) <= 1:
        return []
    bars = parts[:-1]
    return [b.strip() for b in bars if b.strip()]


def count_bars(ly_text: str) -> int:
    """Number of bars (``|``-delimited segments) in the score body."""
    return len(split_bars(ly_text))
