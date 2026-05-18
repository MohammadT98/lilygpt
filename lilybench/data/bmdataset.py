"""Loader for the BMdataset corpus (in-domain reference).

BMdataset bundles 391 Baroque works as 2,645 self-contained ``.ly`` files
plus a ``metadata.json`` mapping per-work ids to composer / period /
musical form / ensemble fields. The Zenodo release ships
``preprocessed/*.ly`` and ``metadata.json`` together; this loader walks
that layout and emits :class:`~lilybench.data.types.CorpusEntry` records
ready for the generation prompt bank and the understanding tasks.

Note: BMdataset metadata is organised at the *work* level — a filename
like ``vivaldi_rv_589_gloria_violino1.ly`` resolves to the
``vivaldi_rv_589_gloria`` metadata entry. The ``part`` (here
``violino1``) is parsed from the filename suffix.
"""

from __future__ import annotations

import json
from pathlib import Path

from lilybench.data.types import CorpusEntry
from lilybench.understanding.score_metadata import (
    extract_key,
    extract_meter,
    extract_note_length,
)
from lilybench.understanding.title_parser import extract_title


def _normalise_form(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(v) for v in value if v)


def _normalise_ensemble(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(v) for v in value if v)


def _resolve_metadata(stem: str, metadata: dict, key_index: list[str]) -> dict:
    """Find the metadata key for ``stem`` (longest-prefix match)."""
    direct = metadata.get(stem)
    if direct is not None:
        return direct
    for key in key_index:
        if stem.startswith(key + "_") or stem == key:
            return metadata[key]
    return {}


def load_bmdataset(
    preprocessed_dir: str | Path,
    metadata_path: str | Path,
) -> list[CorpusEntry]:
    """Walk ``preprocessed_dir`` and yield one entry per ``.ly`` file."""
    preprocessed_dir = Path(preprocessed_dir)
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    key_index = sorted(metadata, key=len, reverse=True)

    out: list[CorpusEntry] = []
    for ly in sorted(preprocessed_dir.glob("*.ly")):
        text = ly.read_text(encoding="utf-8", errors="ignore")
        stem = ly.stem
        meta = _resolve_metadata(stem, metadata, key_index)
        # part: filename suffix after the matched work key (best-effort).
        part: str | None = None
        for key in key_index:
            if stem.startswith(key + "_"):
                part = stem[len(key) + 1 :]
                break
        out.append(
            CorpusEntry(
                source_id=stem,
                source_file=str(ly),
                text=text,
                composer=(meta.get("composer") or None),
                title=extract_title(text),
                style=meta.get("period") or None,
                key=extract_key(text),
                meter=extract_meter(text),
                note_length=extract_note_length(text),
                musical_form=_normalise_form(meta.get("musical_form")),
                ensemble=_normalise_ensemble(
                    meta.get("ensemble") or meta.get("midi_instruments")
                ),
                extras={"part": part} if part else {},
            )
        )
    return out
