"""Shared utilities: JSONL I/O, HF env helpers, LilyPond binary detection."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file. Blank lines are skipped."""
    path = Path(path)
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL records without loading the whole file into memory."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Atomic-ish JSONL writer that creates parent directories on demand."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


# ---------------------------------------------------------------------------
# Hugging Face env
# ---------------------------------------------------------------------------

def apply_hf_env(
    home: str | None = None,
    *,
    token: str | None = None,
    hub_cache: str | None = None,
    transformers_cache: str | None = None,
    datasets_cache: str | None = None,
    offline: bool = False,
) -> None:
    """Export ``HF_*`` env vars, respecting any pre-existing shell exports."""
    if home is not None:
        home_path = str(Path(home).expanduser())
        os.environ.setdefault("HF_HOME", home_path)
        if hub_cache is None:
            hub_cache = str(Path(home_path) / "hub")
        if transformers_cache is None:
            transformers_cache = str(Path(home_path) / "transformers")
        if datasets_cache is None:
            datasets_cache = str(Path(home_path) / "datasets")
    if hub_cache is not None:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hub_cache).expanduser()))
    if transformers_cache is not None:
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(transformers_cache).expanduser()))
    if datasets_cache is not None:
        os.environ.setdefault("HF_DATASETS_CACHE", str(Path(datasets_cache).expanduser()))
    if token is not None:
        os.environ.setdefault("HF_TOKEN", token)
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------------------
# LilyPond binary
# ---------------------------------------------------------------------------

def find_lilypond() -> Path | None:
    """Return the resolved path to a working ``lilypond`` binary, else None.

    Resolution order: ``$LILYPOND_BIN`` env var → first entry on ``$PATH``.
    """
    candidate = os.environ.get("LILYPOND_BIN")
    if candidate and Path(candidate).exists() and _lilypond_works(candidate):
        return Path(candidate)
    name = "lilypond.exe" if os.name == "nt" else "lilypond"
    on_path = shutil.which(name)
    if on_path and _lilypond_works(on_path):
        return Path(on_path)
    return None


def _lilypond_works(binary: str | Path) -> bool:
    try:
        subprocess.run(
            [str(binary), "--version"],
            capture_output=True, timeout=5, check=True,
        )
    except Exception:
        return False
    return True
