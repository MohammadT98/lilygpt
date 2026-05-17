"""Tests for the prompt-bank builder + JSONL round-trip."""

from __future__ import annotations

import pytest

from lilybench.data.types import CorpusEntry
from lilybench.generation import (
    Prompt, build_prompt_bank, load_prompt_bank, write_prompt_bank,
)


def _entry(source_id: str, composer: str, period: str) -> CorpusEntry:
    return CorpusEntry(
        source_id=source_id,
        source_file=f"/data/{source_id}.ly",
        text='\\version "2.24.0"\n{ c4 d4 }\n',
        composer=composer,
        style=period,
        musical_form=("concerto",),
        ensemble=("violin",),
        extras={"part": "violino1"},
    )


def _corpus():
    return [
        _entry("vivaldi_a", "Vivaldi", "Late Baroque"),
        _entry("vivaldi_b", "Vivaldi", "Late Baroque"),
        _entry("bach_a", "Bach", "Late Baroque"),
        _entry("bach_b", "Bach", "Late Baroque"),
    ]


def test_build_prompt_bank_is_deterministic_per_seed():
    bank_a = build_prompt_bank(_corpus(), n=10, seed=42)
    bank_b = build_prompt_bank(_corpus(), n=10, seed=42)
    assert len(bank_a) == 10
    assert [p.source_id for p in bank_a] == [p.source_id for p in bank_b]
    assert [p.user_prompt for p in bank_a] == [p.user_prompt for p in bank_b]


def test_build_prompt_bank_metadata_uses_period_field():
    bank = build_prompt_bank(_corpus(), n=4, seed=0)
    for p in bank:
        assert p.metadata["period"] == "Late Baroque"
        assert p.metadata["musical_form"] == ["concerto"]
        assert p.metadata["part"] == "violino1"


def test_empty_corpus_raises():
    with pytest.raises(ValueError):
        build_prompt_bank([], n=1, seed=0)


def test_roundtrip_jsonl(tmp_path):
    bank = build_prompt_bank(_corpus(), n=5, seed=1)
    path = tmp_path / "bank.jsonl"
    write_prompt_bank(bank, path)
    loaded = load_prompt_bank(path)
    assert loaded == bank
    assert all(isinstance(p, Prompt) for p in loaded)


def test_bars_argument_produces_short_fragment_prompt():
    bank = build_prompt_bank(_corpus(), n=2, seed=0, bars=4)
    for p in bank:
        assert "approximately 4 bars" in p.user_prompt
