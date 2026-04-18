from __future__ import annotations

"""Metadata-conditioning header for training chunks.

Loads ``bmdataset/metadata.json``, resolves a piece key from each preprocessed
filename, parses the ``part:`` suffix from the filename, and emits a structured
``%% === METADATA ===`` LilyPond comment block. The block is always placed in the
loss mask — the model conditions on it but never generates it.
"""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

METADATA_MARKER_START = "%% === METADATA ==="
METADATA_MARKER_END = "%% === END METADATA ==="

_NO_PUB_PREFIX = re.compile(r"^NO_PUB(?:_codifica(?:_\d+)?)?(?:__NO_PUB_codifica)?__")


def _normalize_stem(stem: str) -> str:
    """Normalize a filename stem to the same shape as metadata.json keys.

    bmdataset metadata keys already have: no apostrophes (replaced with ``_``),
    ``&`` → ``and``, commas dropped, hyphens → ``_``, collapsed underscores,
    lowercase. Filenames on disk still carry the raw characters, so we apply the
    same transformation here before matching.
    """
    s = _NO_PUB_PREFIX.sub("", stem).lower()
    s = s.replace("'", "_").replace("&", "and")
    s = s.replace(",", "").replace("-", "_")
    s = re.sub(r"_+", "_", s)
    return s


@dataclass(frozen=True)
class ResolvedMetadata:
    """Resolved metadata for a single preprocessed file."""

    piece_key: str | None
    composer: str | None
    period: str | None
    musical_form: tuple[str, ...]
    ensemble: tuple[str, ...]
    part: str


def _build_key_index(metadata: Mapping[str, dict]) -> list[str]:
    """Return metadata keys sorted by length descending for longest-prefix match."""
    return sorted(metadata.keys(), key=len, reverse=True)


def resolve_piece_key(
    filename_stem: str,
    metadata: Mapping[str, dict],
    key_index: list[str] | None = None,
) -> tuple[str | None, str]:
    """Return ``(piece_key, part)`` for a preprocessed-file stem.

    ``filename_stem`` is the stem without ``.ly``. The function:

    1. Strips the ``NO_PUB[_codifica[_N][__NO_PUB_codifica]]__`` prefix and
       applies the same normalization as metadata.json keys (apostrophes,
       ampersands, commas, hyphens, collapsed underscores, lowercase).
    2. Finds the longest metadata key that matches as a prefix followed by ``_``.
    3. Returns the remainder (after the matched key and its underscore) as the
       part; if no key matches, returns ``(None, normalized_stem)``.
    """
    trimmed = _normalize_stem(filename_stem)
    if key_index is None:
        key_index = _build_key_index(metadata)
    for key in key_index:
        if trimmed == key:
            return key, "full"
        if trimmed.startswith(key + "_"):
            part = trimmed[len(key) + 1 :]
            return key, part if part else "full"
    return None, trimmed


def _as_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def resolve_metadata(
    filename_stem: str,
    metadata: Mapping[str, dict],
    key_index: list[str] | None = None,
) -> ResolvedMetadata:
    """Look up metadata for a preprocessed-file stem. Missing fields become None."""
    key, part = resolve_piece_key(filename_stem, metadata, key_index=key_index)
    # ``_score`` stem represents the full ensemble, not a single part.
    if part == "score":
        part = "full"
    if key is None:
        return ResolvedMetadata(
            piece_key=None,
            composer=None,
            period=None,
            musical_form=(),
            ensemble=(),
            part=part,
        )
    entry = metadata[key]
    return ResolvedMetadata(
        piece_key=key,
        composer=entry.get("composer"),
        period=entry.get("period"),
        musical_form=_as_tuple(entry.get("musical_form")),
        ensemble=_as_tuple(entry.get("midi_instruments")),
        part=part,
    )


def _format_value(value) -> str:
    if value is None:
        return "<unk>"
    if isinstance(value, tuple):
        if not value:
            return "<unk>"
        return ", ".join(value)
    return str(value)


def render_metadata_block(
    meta: ResolvedMetadata,
    *,
    rng: random.Random | None = None,
    p_field: float = 0.15,
    p_block: float = 0.10,
) -> str:
    """Render the metadata comment block, optionally with field/block dropout.

    Returns an empty string (no block) when the block-level dropout fires. Each
    field independently becomes ``<unk>`` with probability ``p_field``. Pass a
    pre-seeded ``random.Random`` for deterministic variant generation.
    """
    if rng is not None and rng.random() < p_block:
        return ""

    fields = (
        ("composer", meta.composer),
        ("period", meta.period),
        ("musical_form", meta.musical_form),
        ("ensemble", meta.ensemble),
        ("part", meta.part),
    )

    lines = [METADATA_MARKER_START]
    for name, value in fields:
        if rng is not None and rng.random() < p_field:
            formatted = "<unk>"
        else:
            formatted = _format_value(value)
        lines.append(f"%% {name}: {formatted}")
    lines.append(METADATA_MARKER_END)
    return "\n".join(lines) + "\n"


def load_metadata(path: str | Path) -> tuple[dict, list[str]]:
    """Load ``metadata.json`` and return ``(metadata, key_index)`` for reuse."""
    with Path(path).open(encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata, _build_key_index(metadata)
