"""Unit tests for train/val/test splitting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilybench.stages.splitting.build_splits import (
    Sample,
    _get_base_work,
    _load_from_jsonl,
    _train_val_test_split,
    _write_jsonl,
)


def test_get_base_work_strips_part_suffix() -> None:
    assert _get_base_work("work_name_part1") == "work_name"
    assert _get_base_work("work_name_part42") == "work_name"
    assert _get_base_work("work_name") == "work_name"
    assert _get_base_work("work_name_violino2") == "work_name_violino2"


def _mk(sample_id: str, source_file: str | None) -> Sample:
    return Sample(id=sample_id, source_file=source_file, raw_data={"id": sample_id})


def test_split_groups_parts_of_same_work_together() -> None:
    samples = [
        _mk(f"w{w}_part{p}", f"w{w}_part{p}")
        for w in range(10)
        for p in range(1, 4)
    ]

    train, val, test = _train_val_test_split(samples, train_ratio=0.8, val_ratio=0.1, seed=0)

    def works(bucket: list[Sample]) -> set[str]:
        return {_get_base_work(s.source_file) for s in bucket if s.source_file}

    train_works, val_works, test_works = works(train), works(val), works(test)

    assert train_works.isdisjoint(val_works)
    assert train_works.isdisjoint(test_works)
    assert val_works.isdisjoint(test_works)
    assert train_works | val_works | test_works == {f"w{w}" for w in range(10)}


def test_split_is_deterministic_for_fixed_seed() -> None:
    samples = [_mk(f"id{i}", f"work{i}") for i in range(50)]

    first = _train_val_test_split(samples, 0.8, 0.1, seed=7)
    second = _train_val_test_split(samples, 0.8, 0.1, seed=7)

    for a, b in zip(first, second):
        assert [s.id for s in a] == [s.id for s in b]


def test_split_handles_samples_without_source_file() -> None:
    samples = [_mk(f"w{w}", f"w{w}") for w in range(20)] + [
        _mk("bare1", None),
        _mk("bare2", None),
    ]

    train, val, test = _train_val_test_split(samples, 0.8, 0.1, seed=1)

    total = len(train) + len(val) + len(test)
    assert total == len(samples)


def test_load_from_jsonl_roundtrip(tmp_path: Path) -> None:
    records = [
        {"id": f"id{i}", "source_file": f"w{i % 3}_part{i}", "full_text": "body"}
        for i in range(6)
    ]
    jsonl = tmp_path / "examples.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    loaded = _load_from_jsonl(jsonl)

    assert [s.id for s in loaded] == [r["id"] for r in records]
    assert all(s.source_file is not None for s in loaded)


def test_load_from_jsonl_empty_file_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="No valid samples"):
        _load_from_jsonl(empty)


def test_write_jsonl_writes_raw_records(tmp_path: Path) -> None:
    samples = [
        Sample(id="a", source_file="x", raw_data={"id": "a", "k": 1}),
        Sample(id="b", source_file="y", raw_data={"id": "b", "k": 2}),
    ]
    out = tmp_path / "out" / "split.jsonl"

    _write_jsonl(samples, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [s.raw_data for s in samples]
