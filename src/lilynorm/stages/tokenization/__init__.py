"""Tokenization utilities (dataset exports are lazy to avoid circular imports)."""

from __future__ import annotations

from typing import Any

from .special_tokens import (
    DEFAULT_SPECIAL_TOKENS,
    add_structural_tokens,
    apply_special_tokens,
    build_special_tokens,
)

__all__ = [
    "DEFAULT_SPECIAL_TOKENS",
    "add_structural_tokens",
    "apply_special_tokens",
    "build_special_tokens",
    "LilyStandardDataset",
    "StandardSample",
    "collate_standard_batch",
]


def __getattr__(name: str) -> Any:
    if name in {"LilyStandardDataset", "StandardSample", "collate_standard_batch"}:
        from lilynorm.stages.dataset.training_dataset import (
            LilyStandardDataset,
            StandardSample,
            collate_standard_batch,
        )
        return {
            "LilyStandardDataset": LilyStandardDataset,
            "StandardSample": StandardSample,
            "collate_standard_batch": collate_standard_batch,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
