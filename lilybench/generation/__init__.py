"""Generation benchmark: prompts, regimes, and runner.

The benchmark pairs a fixed metadata-conditioned prompt bank with a
:class:`Regime` that defines how a backbone is prompted (zero-shot, few-
shot from the training distribution, or few-shot ablation with the
hand-written A-minor demonstrations from §3 of the paper).
"""

from lilybench.generation.metadata_block import render_metadata_block
from lilybench.generation.prompt_bank import (
    Prompt,
    build_prompt_bank,
    load_prompt_bank,
    write_prompt_bank,
)
from lilybench.generation.regimes import (
    REGIME_REGISTRY,
    Regime,
    ZeroShot,
    FewShot,
    register_regime,
)
from lilybench.generation.runner import GenerationConfig, generate

__all__ = [
    "Prompt",
    "Regime",
    "ZeroShot",
    "FewShot",
    "REGIME_REGISTRY",
    "register_regime",
    "GenerationConfig",
    "build_prompt_bank",
    "load_prompt_bank",
    "write_prompt_bank",
    "render_metadata_block",
    "generate",
]
