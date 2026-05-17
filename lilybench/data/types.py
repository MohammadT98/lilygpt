"""Shared types for corpus loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CorpusEntry:
    """One LilyPond score with resolved metadata.

    Fields with no value should be ``None`` (scalars) or an empty list
    (sequences). Loaders are expected to use this type as the unit of input
    for benchmark builders and metric runners; per-corpus extras (Mutopia
    `style`, EMOPIA `clip_id`, …) live in :attr:`extras` so downstream code
    can stay shape-uniform.
    """

    source_id: str
    source_file: str
    text: str
    composer: str | None = None
    title: str | None = None
    style: str | None = None
    key: str | None = None
    meter: str | None = None
    note_length: str | None = None
    musical_form: tuple[str, ...] = ()
    ensemble: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)
