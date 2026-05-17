"""Stratified prompt bank for the generation benchmark.

The paper uses a 200-prompt bank stratified by composer / period / form
so all four backbones see byte-identical inputs across regimes. Each
prompt carries a metadata block plus a natural-language instruction
rendered from the same metadata tuple.

The same bank construction is reused for larger pre-release sweeps by
varying ``n`` and ``seed``.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from lilybench.data.types import CorpusEntry
from lilybench.utils import iter_jsonl, read_jsonl, write_jsonl


@dataclass(frozen=True)
class Prompt:
    """One bank record. ``metadata`` is rendered into the score header at
    inference time; ``user_prompt`` is the natural-language instruction
    that the chat template wraps for zero/few-shot regimes."""

    id: str
    source_id: str
    metadata: dict
    user_prompt: str


def render_user_prompt(metadata: dict, *, bars: int | None = None) -> str:
    """Default prompt template — composer + period + form + ensemble + part.

    Override this function with a custom callable when calling
    :func:`build_prompt_bank` to study different instruction phrasings.
    """
    composer = metadata.get("composer") or "an unnamed 18th-century composer"
    period = metadata.get("period") or "Baroque"
    forms = metadata.get("musical_form") or []
    form_str = ", ".join(forms) if forms else "a short piece"
    part = metadata.get("part") or "full"

    if bars is not None and bars > 0:
        target = "a melodic line" if part == "full" else f"the {part} part as a melodic line"
        return (
            f"Compose a short LilyPond fragment of approximately {bars} bars "
            f"in the style of {composer} ({period}). Form: {form_str}. "
            f"Write {target}, optionally with simple accompaniment "
            f"(chords or a bass line — at most one extra voice). "
            f"Avoid full multi-instrument scores. "
            f"Use Dutch (nederlands) note names and lowercase relative notation. "
            f"Output only the LilyPond code; no prose, no markdown."
        )

    ensemble = metadata.get("ensemble") or []
    ensemble_str = ", ".join(ensemble) if ensemble else "a small ensemble"
    part_clause = "the full ensemble score" if part == "full" else f"the {part} part"
    return (
        f"Write a LilyPond fragment in the style of {composer} ({period}). "
        f"Form: {form_str}. Ensemble: {ensemble_str}. Produce {part_clause}. "
        "Use Dutch (nederlands) note names and lowercase relative notation. "
        "Output only the LilyPond code; no prose, no markdown."
    )


def _entry_to_metadata(entry: CorpusEntry) -> dict:
    part = entry.extras.get("part") if entry.extras else None
    return {
        "composer": entry.composer,
        "period": entry.style,  # BMdataset's `period` is loaded into CorpusEntry.style
        "musical_form": list(entry.musical_form),
        "ensemble": list(entry.ensemble),
        "part": part or "full",
    }


def build_prompt_bank(
    corpus: Sequence[CorpusEntry],
    *,
    n: int = 200,
    seed: int = 1234,
    bars: int | None = None,
    render_prompt=render_user_prompt,
) -> list[Prompt]:
    """Sample ``n`` prompts uniformly from ``corpus`` (with replacement).

    The empirical metadata distribution of the bank matches the corpus
    distribution exactly: every entry counts as one draw. Two calls with
    the same ``seed`` produce byte-identical banks across reruns.
    """
    if not corpus:
        raise ValueError("corpus is empty; cannot build prompt bank")
    rng = random.Random(seed)
    sampled = [rng.choice(corpus) for _ in range(n)]
    bank: list[Prompt] = []
    for i, entry in enumerate(sampled):
        meta = _entry_to_metadata(entry)
        bank.append(
            Prompt(
                id=f"prompt_{i:04d}",
                source_id=entry.source_id,
                metadata=meta,
                user_prompt=render_prompt(meta, bars=bars),
            )
        )
    return bank


def write_prompt_bank(bank: Iterable[Prompt], path: str | Path) -> None:
    write_jsonl((asdict(p) for p in bank), path)


def load_prompt_bank(path: str | Path) -> list[Prompt]:
    return [Prompt(**rec) for rec in read_jsonl(path)]


def stream_prompt_bank(path: str | Path):
    """Yield :class:`Prompt` records lazily; useful for huge banks."""
    for rec in iter_jsonl(path):
        yield Prompt(**rec)
