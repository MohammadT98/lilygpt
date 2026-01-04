"""
Voice extraction utilities for LilyPond music.
"""
from __future__ import annotations

import re


VOICE_ASSIGN_RE = re.compile(r"(?m)^([IVX]{1,4}[A-Za-z0-9_-]*)\s*=")
NOTE_RE = re.compile(r"\b(?:do|re|mi|fa|sol|la|si|[a-g])[',#isbf]*\d", re.I)


def find_voice_blocks(text: str) -> list[tuple[str, int, int]]:
    """
    Find all voice assignment blocks in LilyPond source.

    Uses heuristics: variable names like IviolinoI, IIvln, bassoN with actual note content.
    Returns list of (name, start, end) tuples.
    """
    matches = list(VOICE_ASSIGN_RE.finditer(text))
    blocks: list[tuple[str, int, int]] = []

    for idx, match in enumerate(matches):
        name = match.group(1)
        # Voices typically end in 'n' or 'N'
        if not name.endswith(("n", "N")):
            continue

        start = match.start()
        body_start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:end]

        # Only include if body has actual notes
        if NOTE_RE.search(body):
            blocks.append((name, start, end))

    return blocks


