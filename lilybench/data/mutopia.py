"""Loader for the Mutopia corpus (out-of-domain reference).

Mutopia is shipped as a JSON manifest mapping piece ids to file metadata
plus ``.ly`` files reached relative to the manifest. Each entry supplies
``composer`` and ``style`` (the corpus's coarse genre label); the title
is recovered from the score's ``\\header`` block.

The loader silently skips entries whose ``.ly`` file is missing — the
companion download script is responsible for completing the tree.
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


def _resolve_path(entry: dict, root: Path) -> Path | None:
    cly = entry.get("convert_ly_path")
    if cly:
        p = Path(cly).expanduser().resolve()
        if p.exists():
            return p
    rel = entry.get("localPath") or entry.get("path") or entry.get("lyFile")
    if not rel:
        return None
    p = (root / rel).resolve()
    return p if p.exists() else None


def load_mutopia(manifest_path: str | Path) -> list[CorpusEntry]:
    """Read the Mutopia manifest and yield one entry per resolved score."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    items = manifest.items() if isinstance(manifest, dict) else enumerate(manifest)

    out: list[CorpusEntry] = []
    for key, entry in items:
        if not isinstance(entry, dict):
            continue
        composer = (entry.get("composer") or "").strip()
        style = (entry.get("style") or "").strip()
        if not composer or not style:
            continue
        resolved = _resolve_path(entry, root)
        if resolved is None:
            continue
        try:
            text = resolved.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        out.append(
            CorpusEntry(
                source_id=str(key),
                source_file=str(resolved),
                text=text,
                composer=composer,
                title=extract_title(text),
                style=style,
                key=extract_key(text),
                meter=extract_meter(text),
                note_length=extract_note_length(text),
            )
        )
    return out
