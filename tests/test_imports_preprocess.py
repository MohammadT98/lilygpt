"""Smoke tests: preprocess submodules import without side-effects on torch."""

from __future__ import annotations

import importlib


def test_preprocess_augmentations_importable() -> None:
    importlib.import_module("lilybench.preprocess.augmentations")


def test_preprocess_prelude_importable() -> None:
    importlib.import_module("lilybench.preprocess.prelude")


def test_preprocess_metadata_header_importable() -> None:
    importlib.import_module("lilybench.preprocess.metadata_header")


def test_preprocess_build_dataset_importable() -> None:
    importlib.import_module("lilybench.preprocess.build_dataset")


def test_preprocess_build_splits_importable() -> None:
    importlib.import_module("lilybench.preprocess.build_splits")
