"""Registered backbones evaluated by LilyBench.

The registry pins the four open-weight LLMs used in the paper. Users can
register additional models with :func:`register_model`. ``load_backbone``
materialises a model + tokenizer pair, applying small compatibility shims
that some vendored checkpoints (notably DeepSeek-Coder-V2-Lite) still need
against current ``transformers`` releases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    """Per-backbone metadata used by the runner."""

    model_id: str
    hf_id: str
    dtype: str = "bf16"              # one of {"bf16","fp16","fp32"}
    family: str = "general"           # "general" | "code"
    trust_remote_code: bool = False
    gated: bool = False               # HF requires HF_TOKEN
    generation_end_marker: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "phi4": ModelSpec(
        model_id="phi4",
        hf_id="microsoft/phi-4",
        family="general",
    ),
    "qwen-coder": ModelSpec(
        model_id="qwen-coder",
        hf_id="Qwen/Qwen2.5-Coder-14B-Instruct",
        family="code",
    ),
    "deepseek-coder": ModelSpec(
        model_id="deepseek-coder",
        hf_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        family="code",
        trust_remote_code=True,
    ),
    "codestral": ModelSpec(
        model_id="codestral",
        hf_id="mistralai/Codestral-22B-v0.1",
        family="code",
        gated=True,
    ),
}


_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def register_model(spec: ModelSpec) -> None:
    """Add a new backbone to the registry. Raises if ``model_id`` collides."""
    if spec.model_id in MODEL_REGISTRY:
        raise KeyError(f"model id already registered: {spec.model_id!r}")
    MODEL_REGISTRY[spec.model_id] = spec


def get_spec(model_id: str) -> ModelSpec:
    """Look up the registered spec for ``model_id``."""
    if model_id not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"unknown model id {model_id!r}; known: {known}")
    return MODEL_REGISTRY[model_id]


def list_model_ids() -> list[str]:
    return sorted(MODEL_REGISTRY)


def _install_deepseek_cache_shims() -> None:
    """Backport ``transformers.cache_utils.DynamicCache`` accessors removed in
    transformers ≥ 4.50. DeepSeek-Coder-V2's vendored modeling code still
    queries ``seen_tokens`` / ``get_max_length`` / ``get_usable_length``.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())  # type: ignore[attr-defined]
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None  # type: ignore[attr-defined]
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (  # type: ignore[attr-defined]
            lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
        )
    try:
        import torch.fx  # noqa: F401
        import transformers.utils.import_utils as _imp
        if not hasattr(_imp, "is_torch_fx_available"):
            _imp.is_torch_fx_available = lambda: True  # type: ignore[attr-defined]
    except Exception:
        pass


def load_backbone(
    spec: ModelSpec,
    *,
    device_map: str = "auto",
    quantization: str | None = None,
) -> tuple[Any, Any]:
    """Load ``(model, tokenizer)`` for ``spec`` on the configured device.

    ``quantization`` accepts ``None`` (full precision per ``spec.dtype``),
    ``"int8"``, or ``"int4"`` and routes through ``bitsandbytes``.
    """
    if spec.trust_remote_code:
        _install_deepseek_cache_shims()

    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, use_fast=True, trust_remote_code=spec.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        **spec.extra_kwargs,
    }
    if quantization in {"int8", "int4"}:
        from transformers import BitsAndBytesConfig

        if quantization == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
    elif quantization not in (None, "none"):
        raise ValueError(
            f"unknown quantization {quantization!r}; expected None/'int8'/'int4'"
        )
    else:
        kwargs["torch_dtype"] = _DTYPE_MAP[spec.dtype]

    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
    model.eval()
    return model, tokenizer
