"""Tests for the work-level deterministic splitter."""

from __future__ import annotations

from lilybench.data.splits import split_by_work, stats
from lilybench.data.types import CorpusEntry


def _entries():
    out = []
    # Three works, each with two parts.
    for work in ("a", "b", "c"):
        for part in ("violinoI", "violinoII"):
            out.append(CorpusEntry(
                source_id=f"work_{work}_{part}",
                source_file=f"/data/work_{work}_{part}.ly",
                text="",
                composer="Test",
                extras={"part": part},
            ))
    return out


def test_split_is_deterministic_per_seed():
    e = _entries()
    train1, val1, test1 = split_by_work(e, train_ratio=0.5, val_ratio=0.25, seed=7)
    train2, val2, test2 = split_by_work(e, train_ratio=0.5, val_ratio=0.25, seed=7)
    assert [x.source_id for x in train1] == [x.source_id for x in train2]
    assert [x.source_id for x in val1] == [x.source_id for x in val2]
    assert [x.source_id for x in test1] == [x.source_id for x in test2]


def test_parts_never_leak_across_splits():
    e = _entries()
    train, val, test = split_by_work(e, train_ratio=0.34, val_ratio=0.33, seed=42)
    # Same work across two splits would mean leakage.
    work_of = lambda x: x.source_id.split("_violino")[0]
    works = {label: {work_of(x) for x in s} for label, s in zip("tvT", (train, val, test))}
    assert works["t"].isdisjoint(works["v"])
    assert works["t"].isdisjoint(works["T"])
    assert works["v"].isdisjoint(works["T"])


def test_stats_counts():
    e = _entries()
    train, val, test = split_by_work(e, train_ratio=0.5, val_ratio=0.25, seed=0)
    s = stats((train, val, test))
    assert sum(d["files"] for d in s) == len(e)
    assert sum(d["works"] for d in s) == 3
