"""Unit tests for the full-file dataset builder's chunking and record schema."""

from __future__ import annotations

import json
from pathlib import Path

from lilybench.stages.dataset.build_fullfile_dataset import (
    VARIANTS,
    _body_segments,
    _file_seed,
    _pack_chunks,
    build_dataset,
)


class TestBodySegments:
    def test_empty_assignments_returns_single_whole_segment(self) -> None:
        body = "c'4 d' e' f'"
        assert _body_segments(body, []) == [(0, len(body))]

    def test_leading_prefix_is_kept_as_first_segment(self) -> None:
        body = "\n\nfoo = { c4 }\nbar = { d4 }\n"
        starts = [body.index("foo"), body.index("bar")]

        segs = _body_segments(body, starts)

        assert segs[0] == (0, starts[0])
        assert segs[1] == (starts[0], starts[1])
        assert segs[2] == (starts[1], len(body))

    def test_no_leading_prefix_when_first_assignment_at_zero(self) -> None:
        body = "foo = { c4 }\nbar = { d4 }\n"
        starts = [0, body.index("bar")]

        segs = _body_segments(body, starts)

        assert segs[0] == (0, starts[1])
        assert segs[1] == (starts[1], len(body))


class TestPackChunks:
    def test_empty_segments_yields_no_chunks(self) -> None:
        assert _pack_chunks([], 100, 100) == []

    def test_greedy_packing_respects_budget(self) -> None:
        segs = [(0, 10), (10, 20), (20, 30), (30, 40)]

        chunks = _pack_chunks(segs, chunk0_body_budget=20, chunki_body_budget=20)

        assert chunks == [(0, 20), (20, 40)]

    def test_chunk_zero_can_use_different_budget(self) -> None:
        segs = [(0, 5), (5, 15), (15, 25), (25, 35)]

        chunks = _pack_chunks(segs, chunk0_body_budget=15, chunki_body_budget=25)

        assert chunks[0][0] == 0
        assert chunks[0][1] <= 15
        assert chunks[-1][1] == 35

    def test_oversize_segment_emitted_as_single_chunk(self) -> None:
        segs = [(0, 100), (100, 110)]

        chunks = _pack_chunks(segs, chunk0_body_budget=20, chunki_body_budget=20)

        assert chunks[0] == (0, 100)
        assert chunks[-1][1] == 110


def test_file_seed_is_deterministic_and_nonzero() -> None:
    assert _file_seed("abc") == _file_seed("abc")
    assert _file_seed("abc") != _file_seed("abd")
    assert _file_seed("any") != 0


def test_build_dataset_emits_expected_schema(
    tmp_path: Path, preprocessed_ly: str, sample_metadata: dict
) -> None:
    input_dir = tmp_path / "preprocessed"
    input_dir.mkdir()
    stem = "vivaldi_rv_589_gloria_violino1"
    (input_dir / f"{stem}.ly").write_text(preprocessed_ly, encoding="utf-8")

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(sample_metadata), encoding="utf-8")

    output_path = tmp_path / "all_examples.jsonl"
    totals = build_dataset(
        input_dir=input_dir,
        metadata_path=metadata_path,
        output_path=output_path,
        max_chars=8192,
        global_seed=42,
    )

    assert totals["files"] == 1
    assert totals["files_with_records"] == 1
    assert totals["variants_attempted"] == len(VARIANTS)
    assert totals["chunks_emitted"] >= len(VARIANTS)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    variants = {rec["variant"] for rec in records}
    assert variants == {v.name for v in VARIANTS}

    for rec in records:
        assert set(rec.keys()) >= {
            "id",
            "source_file",
            "variant",
            "chunk_index",
            "chunk_total",
            "seed",
            "full_text",
            "metadata_char_range",
            "prelude_char_range",
            "label_mask_char_ranges",
        }
        meta_start, meta_end = rec["metadata_char_range"]
        prel_start, prel_end = rec["prelude_char_range"]
        assert meta_start == 0
        assert meta_end == prel_start
        assert rec["label_mask_char_ranges"] == [[0, prel_end]]
        assert rec["full_text"].startswith("%% === METADATA ===")
        assert "%% composer: Vivaldi" in rec["full_text"][:meta_end]


def test_build_dataset_skips_brace_imbalanced_source(
    tmp_path: Path, sample_metadata: dict
) -> None:
    input_dir = tmp_path / "preprocessed"
    input_dir.mkdir()
    (input_dir / "vivaldi_rv_589_gloria_violino1.ly").write_text(
        'tune = { c4 d4\n',  # unbalanced
        encoding="utf-8",
    )

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(sample_metadata), encoding="utf-8")

    output_path = tmp_path / "all_examples.jsonl"
    totals = build_dataset(
        input_dir=input_dir,
        metadata_path=metadata_path,
        output_path=output_path,
    )

    assert totals["source_invalid"] == 1
    assert totals["chunks_emitted"] == 0
    assert output_path.read_text(encoding="utf-8") == ""
