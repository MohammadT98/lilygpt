"""Shared helpers for understanding task builders and scorers."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

from lilybench.data.types import CorpusEntry


def task_rng(seed: int, name: str) -> random.Random:
    """Per-task RNG: ``seed ^ hash(name)`` for stable subset runs."""
    return random.Random(seed ^ (hash(name) & 0xFFFFFFFF))


def pick_subset(
    candidates: Sequence[CorpusEntry], n: int, rng: random.Random
) -> list[CorpusEntry]:
    if not candidates:
        return []
    return rng.sample(candidates, min(n, len(candidates)))


def sample_distractors(
    pool: Sequence[str], gold: str, rng: random.Random, k: int = 3
) -> list[str]:
    candidates = [x for x in pool if x != gold]
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) < k:
        raise ValueError(
            f"pool too small for {k} distractors (have {len(candidates)} after "
            f"removing gold={gold!r})"
        )
    return rng.sample(candidates, k)


def align_by_id(
    bench: Sequence[dict], predictions: Sequence[dict]
) -> list[tuple[dict, dict]]:
    """Return aligned ``(record, prediction)`` pairs."""
    by_id = {r["id"]: r for r in bench}
    return [(by_id[p["id"]], p) for p in predictions if p["id"] in by_id]


def group_by(records: Sequence[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        out[str(r.get(key))].append(r)
    return out
