"""Deterministic work-level splits over a corpus.

LilyBench splits at the *work* level rather than the file level so that
parts of the same multi-instrument score never leak across train/val/test
folds. Splits are reproducible from a fixed seed; the paper uses
``seed=1234`` for the BMdataset reference splits shipped on Zenodo.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Iterable, Sequence

from lilybench.data.types import CorpusEntry


_PART_SUFFIX_RE = re.compile(r"_part\d+$|_(score|violin[oI]+|viola|cello|basso|bc|continuo|flauto[12]?|tromba[12]?|corno[12]?|oboe[12]?|fagotto)$")


def _work_id(entry: CorpusEntry) -> str:
    """Group key: strip per-part suffixes so all parts share a work id."""
    return _PART_SUFFIX_RE.sub("", entry.source_id)


def split_by_work(
    entries: Sequence[CorpusEntry],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1234,
) -> tuple[list[CorpusEntry], list[CorpusEntry], list[CorpusEntry]]:
    """Return ``(train, val, test)`` lists. ``test_ratio = 1 - train - val``."""
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"invalid ratios: train={train_ratio}, val={val_ratio}"
        )

    by_work: dict[str, list[CorpusEntry]] = defaultdict(list)
    for e in entries:
        by_work[_work_id(e)].append(e)

    work_ids = sorted(by_work)
    rng = random.Random(seed)
    rng.shuffle(work_ids)

    n = len(work_ids)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    train_ids = set(work_ids[:n_train])
    val_ids = set(work_ids[n_train : n_train + n_val])

    train: list[CorpusEntry] = []
    val: list[CorpusEntry] = []
    test: list[CorpusEntry] = []
    for wid in work_ids:
        bucket = train if wid in train_ids else val if wid in val_ids else test
        bucket.extend(by_work[wid])
    return train, val, test


def stats(splits: Iterable[Sequence[CorpusEntry]]) -> list[dict]:
    """Diagnostic helper: count files, works, and composers per split."""
    out = []
    for split in splits:
        works = {_work_id(e) for e in split}
        composers = {e.composer for e in split if e.composer}
        out.append({
            "files": len(split),
            "works": len(works),
            "composers": len(composers),
        })
    return out
