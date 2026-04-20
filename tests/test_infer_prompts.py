"""Smoke tests for the metadata-conditioned prompt bank additions in infer.py."""

from __future__ import annotations

import json
from pathlib import Path

from lilybench.infer import (
    _build_lora_preamble,
    _load_prompt_bank,
    _render_metadata_block,
    _strip_markdown_fences,
)


def test_render_metadata_block_roundtrips_all_fields():
    meta = {
        "composer": "Charpentier",
        "period": "Late Baroque",
        "musical_form": ["motet"],
        "ensemble": ["violin", "viola", "cello"],
        "part": "violino2",
    }
    block = _render_metadata_block(meta)
    assert block.startswith("%% === METADATA ===\n")
    assert block.endswith("%% === END METADATA ===\n")
    assert "%% composer: Charpentier" in block
    assert "%% period: Late Baroque" in block
    assert "%% musical_form: motet" in block
    assert "%% ensemble: violin, viola, cello" in block
    assert "%% part: violino2" in block


def test_render_metadata_block_none_yields_empty_block():
    block = _render_metadata_block(None)
    assert block == "%% === METADATA ===\n%% === END METADATA ===\n"


def test_render_metadata_block_skips_empty_fields():
    meta = {"composer": "Vivaldi", "period": None, "musical_form": [], "ensemble": "", "part": "full"}
    block = _render_metadata_block(meta)
    assert "%% composer: Vivaldi" in block
    assert "%% part: full" in block
    assert "%% period:" not in block
    assert "%% musical_form:" not in block
    assert "%% ensemble:" not in block


def test_build_lora_preamble_injects_metadata():
    pre = _build_lora_preamble(
        version="2.24.4",
        language="nederlands",
        metadata={"composer": "Vivaldi", "period": "Late Baroque"},
    )
    assert "%% composer: Vivaldi" in pre
    assert '\\version "2.24.4"' in pre
    assert '\\language "nederlands"' in pre
    assert pre.index("%% === METADATA ===") < pre.index("\\version")


def test_build_lora_preamble_empty_when_metadata_missing():
    pre = _build_lora_preamble(version="2.24.4", language="nederlands")
    assert pre.startswith("%% === METADATA ===\n%% === END METADATA ===\n")


def test_strip_markdown_fences_extracts_inner_block():
    text = '```lilypond\n\\version "2.24.0"\n\\relative c\' { c d e f }\n```'
    assert _strip_markdown_fences(text) == '\\version "2.24.0"\n\\relative c\' { c d e f }'


def test_strip_markdown_fences_handles_bare_fence():
    text = '```\n\\version "2.24.0"\n```'
    assert _strip_markdown_fences(text) == '\\version "2.24.0"'


def test_strip_markdown_fences_noop_on_plain_lilypond():
    text = '\\version "2.24.0"\n\\relative c\' { c d e f }'
    assert _strip_markdown_fences(text) == text


def test_strip_markdown_fences_unterminated_drops_opening_line():
    text = '```lilypond\n\\version "2.24.0"\n\\relative c\' { c d e f }'
    assert _strip_markdown_fences(text) == '\\version "2.24.0"\n\\relative c\' { c d e f }'


def test_strip_markdown_fences_picks_first_block_when_multiple():
    text = 'prose\n```lilypond\n\\version "A"\n```\nmore prose\n```lilypond\n\\version "B"\n```'
    assert _strip_markdown_fences(text) == '\\version "A"'


def test_load_prompt_bank_reads_jsonl(tmp_path: Path):
    bank = tmp_path / "bank.jsonl"
    bank.write_text(
        json.dumps({"metadata": {"composer": "Vivaldi"}, "user_prompt": "Go"}) + "\n"
        + json.dumps({"metadata": {"composer": "Bach"}, "user_prompt": "Stop"}) + "\n",
        encoding="utf-8",
    )
    records = _load_prompt_bank(bank)
    assert len(records) == 2
    assert records[0]["metadata"]["composer"] == "Vivaldi"
    assert records[1]["user_prompt"] == "Stop"
