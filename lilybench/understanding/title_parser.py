"""Extract ``title`` from a LilyPond ``\\header { ... }`` block.

Mutopia's ``dataset_mutopia.json`` carries composer + style + instruments but
no title; we recover titles directly from the score header.
"""

from __future__ import annotations

import re

# Match the *first* ``\header { ... }`` block. ``{[^{}]*}`` deliberately
# refuses nested braces — LilyPond headers can technically nest, but the
# Mutopia corpus headers are flat in practice and the simpler regex avoids a
# recursive parse.
_HEADER_BLOCK_RE = re.compile(r"\\header\s*\{([^{}]*)\}", re.DOTALL)

# Match ``title = "..."`` with escape-aware double-quoted string content.
_TITLE_FIELD_RE = re.compile(
    r'(?<![A-Za-z_])title\s*=\s*"((?:[^"\\]|\\.)*)"',
)


def extract_title(ly_text: str) -> str | None:
    """Return the title string from the first ``\\header { ... }`` block.

    Returns ``None`` if no header block is present, the block lacks a
    ``title`` field, or the title is empty / whitespace-only.
    """
    if not ly_text:
        return None
    block_match = _HEADER_BLOCK_RE.search(ly_text)
    if block_match is None:
        return None
    block_body = block_match.group(1)
    field_match = _TITLE_FIELD_RE.search(block_body)
    if field_match is None:
        return None
    value = field_match.group(1).strip()
    return value or None
