"""Tokenization stage: GPT tokenization and dataset loading."""

from . import tokenize_gpt
from .tokenize_gpt import DEFAULT_MODEL_NAME, DEFAULT_MAX_LENGTH

# Lazy-load torch-dependent imports (only needed for training/data loading)
def __getattr__(name):
    if name in ("LilyTokensDataset", "collate_batch", "create_dataloader", "JsonlSample"):
        from .dataset_jsonl import LilyTokensDataset, collate_batch, create_dataloader, JsonlSample
        globals()[name] = {"LilyTokensDataset": LilyTokensDataset, "collate_batch": collate_batch, "create_dataloader": create_dataloader, "JsonlSample": JsonlSample}[name]
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "tokenize_gpt",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MAX_LENGTH",
    "LilyTokensDataset",
    "collate_batch",
    "create_dataloader",
    "JsonlSample",
]
