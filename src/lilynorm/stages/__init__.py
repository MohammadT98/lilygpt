"""LilyGPT pipeline stages: normalization → tokenization → dataset → splitting → training."""

from lilynorm.stages.normalization import preprocess, normalize_syntax, engrave_strip
from lilynorm.stages.splitting import build_splits

__all__ = [
    # Normalization
    "preprocess",
    "normalize_syntax",
    "engrave_strip",
    # Splitting
    "build_splits",
]
