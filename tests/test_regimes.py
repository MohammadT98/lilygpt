"""Tests for the generation regimes."""

from __future__ import annotations

import pytest

from lilybench.generation import (
    FewShot,
    Prompt,
    REGIME_REGISTRY,
    Regime,
    ZeroShot,
    register_regime,
)


class _FakeTokenizer:
    chat_template = None  # exercise the non-instruct fallback


_PROMPT = Prompt(
    id="p0",
    source_id="vivaldi_test",
    metadata={
        "composer": "Vivaldi", "period": "Late Baroque",
        "musical_form": ["concerto"], "ensemble": ["violin"], "part": "violino1",
    },
    user_prompt="Compose a fragment.",
)


def test_zero_shot_includes_metadata_block_and_prompt():
    text = ZeroShot().build_prompt(_PROMPT, tokenizer=_FakeTokenizer())
    assert "%% === METADATA ===" in text
    assert "%% composer: Vivaldi" in text
    assert "%% part: violino1" in text
    assert "Compose a fragment." in text


def test_few_shot_prepends_demonstrations():
    fs = FewShot(demonstrations="Example 1: c4 d4 e4 f4 |")
    text = fs.build_prompt(_PROMPT, tokenizer=_FakeTokenizer())
    assert "Example 1: c4 d4 e4 f4 |" in text
    assert "Compose a fragment." in text
    # demos come before the user message text
    assert text.find("Example 1") < text.find("Compose a fragment.")


def test_few_shot_from_file_named_class(tmp_path):
    path = tmp_path / "demos.txt"
    path.write_text("Demo body.", encoding="utf-8")
    fs = FewShot.from_file(path, name="few_ablation")
    assert fs.name == "few_ablation"
    text = fs.build_prompt(_PROMPT, tokenizer=_FakeTokenizer())
    assert "Demo body." in text


def test_registry_contains_paper_regimes():
    assert set(REGIME_REGISTRY) == {"zero", "few"}


def test_register_regime_inserts_and_collides():
    class Custom(Regime):
        name = "custom_regime"

        def build_prompt(self, prompt, *, tokenizer):
            return "X"

    register_regime(Custom)
    try:
        assert "custom_regime" in REGIME_REGISTRY
    finally:
        REGIME_REGISTRY.pop("custom_regime", None)

    class Dup(Regime):
        name = "zero"

        def build_prompt(self, prompt, *, tokenizer):
            return "X"

    with pytest.raises(KeyError):
        register_regime(Dup)
