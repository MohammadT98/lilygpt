"""Smoke tests: Hydra configs compose without launching entry points."""

from __future__ import annotations

from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir  # noqa: E402

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _compose(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def test_train_config_composes() -> None:
    cfg = _compose("train")
    assert cfg.model.id == "phi4"
    assert cfg.lora.r == 8
    assert cfg.max_length == 1024


def test_train_config_model_override() -> None:
    cfg = _compose("train", overrides=["model=qwen-coder"])
    assert cfg.model.id == "qwen-coder"


def test_infer_config_composes() -> None:
    cfg = _compose("infer")
    assert cfg.model.id == "phi4"
    assert cfg.regime.name == "zero"
    assert cfg.num_samples == 100


def test_infer_config_regime_lora_override() -> None:
    cfg = _compose("infer", overrides=["regime=lora", "regime.path=/tmp/adapter"])
    assert cfg.regime.name == "lora"
    assert cfg.regime.path == "/tmp/adapter"


def test_evaluate_loss_config_composes() -> None:
    cfg = _compose("evaluate/loss", overrides=["lora_path=/tmp/adapter"])
    assert cfg.model.id == "phi4"
    assert cfg.max_length == 2048


def test_evaluate_text_midi_config_composes() -> None:
    cfg = _compose("evaluate/text_midi")
    assert cfg.expected_notation == "relative"


def test_evaluate_fmd_config_composes() -> None:
    cfg = _compose(
        "evaluate/fmd",
        overrides=[
            "generations_dir=/tmp/gen",
            "reference_path=/tmp/ref",
            "embedder_checkpoint=/tmp/lilybert",
        ],
    )
    assert cfg.reference_kind == "test"
    assert cfg.batch_size == 16
