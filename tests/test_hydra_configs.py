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
    assert cfg.wandb.project == "lilybench"
    assert cfg.wandb.run_name == "phi4_lora"
    assert cfg.tensorboard.enabled is True
    assert cfg.tensorboard.log_dir == f"{cfg.output_dir}/logs"
    assert cfg.hf.home is None
    assert cfg.hf.token is None
    assert cfg.hf.offline is False


def test_train_config_hf_override() -> None:
    cfg = _compose("train", overrides=["hf.home=/tmp/cache", "hf.token=dummy"])
    assert cfg.hf.home == "/tmp/cache"
    assert cfg.hf.token == "dummy"


def test_train_config_model_override() -> None:
    cfg = _compose("train", overrides=["model=qwen-coder"])
    assert cfg.model.id == "qwen-coder"
    assert cfg.wandb.run_name == "qwen-coder_lora"


def test_infer_config_composes() -> None:
    cfg = _compose("infer")
    assert cfg.model.id == "phi4"
    assert cfg.regime.name == "zero"
    assert cfg.num_samples == 100
    assert cfg.hf.home is None
    assert cfg.hf.offline is False


def test_infer_config_regime_lora_override() -> None:
    cfg = _compose("infer", overrides=["regime=lora", "regime.path=/tmp/adapter"])
    assert cfg.regime.name == "lora"
    assert cfg.regime.path == "/tmp/adapter"


def test_evaluate_loss_config_composes() -> None:
    cfg = _compose("evaluate/loss", overrides=["lora_path=/tmp/adapter"])
    assert cfg.model.id == "phi4"
    assert cfg.max_length == 2048
    assert cfg.hf.home is None


def test_evaluate_text_midi_config_composes() -> None:
    cfg = _compose("evaluate/text_midi")
    assert cfg.expected_notation == "relative"
    assert cfg.get("hf") is None
    assert cfg.reference_midi_dir is None
    assert cfg.reference_aggregate_path is None
    assert dict(cfg.reference_aggregate_paths) == {}


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
    assert cfg.hf.home is None
