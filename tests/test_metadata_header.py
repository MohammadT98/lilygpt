"""Unit tests for metadata resolution and header rendering."""

from __future__ import annotations

import json
import random
from pathlib import Path

from lilybench.stages.dataset.metadata_header import (
    METADATA_MARKER_END,
    METADATA_MARKER_START,
    load_metadata,
    render_metadata_block,
    resolve_metadata,
    resolve_piece_key,
)


def test_resolve_piece_key_strips_nopub_prefix_and_parses_part(sample_metadata: dict) -> None:
    key, part = resolve_piece_key(
        "NO_PUB__charpentier_lauda_sion_H_268_egredimini_H_280_violino2",
        sample_metadata,
    )

    assert key == "charpentier_lauda_sion_h_268_egredimini_h_280"
    assert part == "violino2"


def test_resolve_piece_key_returns_full_when_stem_matches_key(sample_metadata: dict) -> None:
    key, part = resolve_piece_key(
        "charpentier_lauda_sion_h_268_egredimini_h_280", sample_metadata
    )

    assert key == "charpentier_lauda_sion_h_268_egredimini_h_280"
    assert part == "full"


def test_resolve_piece_key_longest_prefix_wins() -> None:
    metadata = {"a_b": {}, "a_b_c": {}}

    key, _ = resolve_piece_key("a_b_c_violino1", metadata)

    assert key == "a_b_c"


def test_resolve_piece_key_unknown_returns_none(sample_metadata: dict) -> None:
    key, part = resolve_piece_key("telemann_unknown_work", sample_metadata)

    assert key is None
    assert part == "telemann_unknown_work"


def test_resolve_metadata_score_stem_becomes_full_part(sample_metadata: dict) -> None:
    meta = resolve_metadata(
        "charpentier_lauda_sion_h_268_egredimini_h_280_score", sample_metadata
    )

    assert meta.composer == "Charpentier"
    assert meta.period == "Late Baroque"
    assert meta.musical_form == ("motet",)
    assert meta.ensemble == ("violin", "viola", "cello", "flute")
    assert meta.part == "full"


def test_resolve_metadata_handles_list_valued_form(sample_metadata: dict) -> None:
    meta = resolve_metadata("vivaldi_rv_589_gloria_violino1", sample_metadata)

    assert meta.musical_form == ("mass", "gloria")


def test_resolve_metadata_missing_key_yields_none_fields(sample_metadata: dict) -> None:
    meta = resolve_metadata("nobody_unknown", sample_metadata)

    assert meta.piece_key is None
    assert meta.composer is None
    assert meta.musical_form == ()


def test_render_metadata_block_contains_markers_and_fields(sample_metadata: dict) -> None:
    meta = resolve_metadata("vivaldi_rv_589_gloria_violino1", sample_metadata)

    block = render_metadata_block(meta)

    assert block.startswith(METADATA_MARKER_START)
    assert METADATA_MARKER_END in block
    assert "%% composer: Vivaldi" in block
    assert "%% musical_form: mass, gloria" in block
    assert "%% part: violino1" in block
    assert block.endswith("\n")


def test_render_metadata_block_block_dropout_returns_empty(sample_metadata: dict) -> None:
    meta = resolve_metadata("vivaldi_rv_589_gloria_violino1", sample_metadata)

    block = render_metadata_block(meta, rng=random.Random(0), p_field=0.0, p_block=1.0)

    assert block == ""


def test_render_metadata_block_field_dropout_masks_values(sample_metadata: dict) -> None:
    meta = resolve_metadata("vivaldi_rv_589_gloria_violino1", sample_metadata)

    block = render_metadata_block(meta, rng=random.Random(0), p_field=1.0, p_block=0.0)

    for field in ("composer", "period", "musical_form", "ensemble", "part"):
        assert f"%% {field}: <unk>" in block


def test_render_metadata_block_is_deterministic_with_same_seed(sample_metadata: dict) -> None:
    meta = resolve_metadata("vivaldi_rv_589_gloria_violino1", sample_metadata)

    a = render_metadata_block(meta, rng=random.Random(123), p_field=0.5, p_block=0.5)
    b = render_metadata_block(meta, rng=random.Random(123), p_field=0.5, p_block=0.5)

    assert a == b


def test_load_metadata_builds_longest_first_index(
    tmp_path: Path, sample_metadata: dict
) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(sample_metadata), encoding="utf-8")

    metadata, key_index = load_metadata(path)

    assert metadata == sample_metadata
    assert key_index == sorted(sample_metadata.keys(), key=len, reverse=True)
