"""Smoke tests: evaluate submodule imports."""

from __future__ import annotations

import importlib

import pytest


def test_evaluate_extract_detokenized_importable() -> None:
    importlib.import_module("lilybench.evaluate.extract_detokenized")


def test_evaluate_text_midi_importable() -> None:
    pytest.importorskip("music21")
    pytest.importorskip("hydra")
    importlib.import_module("lilybench.evaluate.text_midi")


def test_evaluate_fmd_importable() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("torch")
    pytest.importorskip("hydra")
    importlib.import_module("lilybench.evaluate.fmd")


def test_evaluate_loss_importable() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("hydra")
    importlib.import_module("lilybench.evaluate.loss")
