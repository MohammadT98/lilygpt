"""Smoke tests: data submodule imports (requires torch; skipped if absent)."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")


def test_data_training_dataset_importable() -> None:
    mod = importlib.import_module("lilybench.data.training_dataset")
    assert hasattr(mod, "LilyStandardDataset")
    assert hasattr(mod, "collate_standard_batch")
