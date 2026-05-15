"""Extract and mask key / meter / note-length from raw LilyPond text.

These are needed by the metadata_qa and metadata_prediction tasks. The
regexes mirror the ones in ``lilybench/evaluate/text_midi.py`` but are
duplicated here so importing the bench builder does not pull in music21.
"""

from __future__ import annotations

import re


_KEY_RE = re.compile(
    r"\\key\s+((?:[a-g]|do|re|mi|fa|sol|la|si))(isis|eses|is|es|dd|d|bb|b)?\s+\\([A-Za-z]+)"
)
_TIME_RE = re.compile(r"\\time\s+(\d+)\s*/\s*(\d+)")


def extract_key(ly_text: str) -> str | None:
    """Return the first ``\\key`` declaration as ``"<note> \\<mode>"``."""
    m = _KEY_RE.search(ly_text)
    if m is None:
        return None
    letter, acc, mode = m.group(1), m.group(2) or "", m.group(3)
    return f"{letter}{acc} \\{mode}"


def extract_meter(ly_text: str) -> str | None:
    """Return the first ``\\time N/M`` declaration as ``"N/M"``."""
    m = _TIME_RE.search(ly_text)
    if m is None:
        return None
    return f"{m.group(1)}/{m.group(2)}"


def extract_note_length(ly_text: str) -> str | None:
    """Proxy for ABC's ``L:`` field — the denominator of ``\\time``.

    LilyPond does not carry an explicit unit-length field, so we use the
    time signature's denominator as the closest equivalent. Returns ``None``
    when no ``\\time`` is declared.
    """
    m = _TIME_RE.search(ly_text)
    if m is None:
        return None
    return m.group(2)


# Removal patterns — note the trailing ``\\<mode>`` for ``\\key`` so the
# masking strips the whole declaration.
_KEY_FULL_RE = re.compile(
    r"\\key\s+(?:[a-g]|do|re|mi|fa|sol|la|si)(?:isis|eses|is|es|dd|d|bb|b)?\s+\\[A-Za-z]+"
)
_TIME_FULL_RE = re.compile(r"\\time\s+\d+\s*/\s*\d+")


def mask_field(ly_text: str, field: str) -> str:
    """Remove the chosen field's declaration from the score text.

    Supported fields: ``key``, ``meter``, ``note_length`` (the last masks the
    meter declaration since note length is derived from it).
    """
    if field == "key":
        return _KEY_FULL_RE.sub("", ly_text)
    if field in {"meter", "note_length"}:
        return _TIME_FULL_RE.sub("", ly_text)
    raise ValueError(f"unknown field {field!r}; expected key, meter, note_length")
