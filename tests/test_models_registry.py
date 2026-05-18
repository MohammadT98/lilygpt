"""Tests for the backbone registry."""

from __future__ import annotations

import pytest

from lilybench.models import (
    MODEL_REGISTRY, ModelSpec, get_spec, list_model_ids, register_model,
)


def test_paper_backbones_are_registered():
    assert set(MODEL_REGISTRY) == {"phi4", "qwen-coder", "deepseek-coder", "codestral"}


def test_get_spec_returns_dataclass():
    spec = get_spec("phi4")
    assert isinstance(spec, ModelSpec)
    assert spec.hf_id == "microsoft/phi-4"


def test_get_spec_unknown_raises_with_known_ids():
    with pytest.raises(KeyError) as exc:
        get_spec("does-not-exist")
    msg = str(exc.value)
    for known in MODEL_REGISTRY:
        assert known in msg


def test_register_model_inserts_and_returns_with_get_spec():
    spec = ModelSpec(
        model_id="custom-test", hf_id="org/custom", dtype="bf16", family="general"
    )
    register_model(spec)
    try:
        assert "custom-test" in list_model_ids()
        assert get_spec("custom-test") is spec
    finally:
        MODEL_REGISTRY.pop("custom-test", None)


def test_register_model_collision_raises():
    with pytest.raises(KeyError):
        register_model(get_spec("phi4"))
