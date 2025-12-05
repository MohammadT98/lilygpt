"""LilyGPT pipeline stages: preprocessing → tokenization → splitting → training."""

from lilynorm.stages.preprocessing import preparse, normalize, engrave_strip
from lilynorm.stages.tokenization import tokenize_gpt, DEFAULT_MODEL_NAME, DEFAULT_MAX_LENGTH
from lilynorm.stages.splitting import build_splits

# Lazy-load torch-dependent imports
def __getattr__(name):
    if name in ("LilyTokensDataset", "collate_batch", "create_dataloader", "JsonlSample"):
        from lilynorm.stages.tokenization.dataset_jsonl import LilyTokensDataset, collate_batch, create_dataloader, JsonlSample
        return {"LilyTokensDataset": LilyTokensDataset, "collate_batch": collate_batch, "create_dataloader": create_dataloader, "JsonlSample": JsonlSample}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Preprocessing
    "preparse",
    "normalize", 
    "engrave_strip",
    # Tokenization
    "tokenize_gpt",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MAX_LENGTH",
    "LilyTokensDataset",
    "collate_batch",
    "create_dataloader",
    "JsonlSample",
    # Splitting
    "build_splits",
]