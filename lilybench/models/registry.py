from __future__ import annotations

"""Registry of models evaluated in LilyBench.

One source of truth for model HuggingFace ids, dtype, chat template kind,
and LoRA target selection.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    hf_id: str
    dtype: str  # "bf16" | "fp16" | "fp32"
    chat_template_kind: str  # "openai-harmony" | "chatml" | "phi" | "mistral" | "deepseek" | "gemma"
    max_seq_len: int
    lora_target_modules: str | tuple[str, ...]
    trust_remote_code: bool = False
    generation_end_marker: str | None = None
    family: str = "general"  # "general" | "code"
    gated: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gpt-oss": ModelSpec(
        model_id="gpt-oss",
        hf_id="openai/gpt-oss-20b",
        dtype="bf16",
        chat_template_kind="openai-harmony",
        max_seq_len=2048,
        lora_target_modules="all-linear",
        trust_remote_code=True,
        generation_end_marker="<|return|>",
        family="general",
    ),
    "phi4": ModelSpec(
        model_id="phi4",
        hf_id="microsoft/phi-4",
        dtype="bf16",
        chat_template_kind="phi",
        max_seq_len=2048,
        lora_target_modules="all-linear",
        family="general",
    ),
    "qwen-coder": ModelSpec(
        model_id="qwen-coder",
        hf_id="Qwen/Qwen2.5-Coder-14B-Instruct",
        dtype="bf16",
        chat_template_kind="chatml",
        max_seq_len=2048,
        lora_target_modules="all-linear",
        family="code",
    ),
    "deepseek-coder": ModelSpec(
        model_id="deepseek-coder",
        hf_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        dtype="bf16",
        chat_template_kind="deepseek",
        max_seq_len=2048,
        lora_target_modules="all-linear",
        trust_remote_code=True,
        family="code",
    ),
    "codestral": ModelSpec(
        model_id="codestral",
        hf_id="mistralai/Codestral-22B-v0.1",
        dtype="bf16",
        chat_template_kind="mistral",
        max_seq_len=2048,
        lora_target_modules="all-linear",
        family="code",
        gated=True,
    ),
    "gemma": ModelSpec(
        model_id="gemma",
        hf_id="google/gemma-4-31B",
        dtype="bf16",
        chat_template_kind="gemma",
        max_seq_len=2048,
        lora_target_modules="all-linear",
        family="general",
        gated=True,
    ),
}


def get_spec(model_id: str) -> ModelSpec:
    """Return the ModelSpec for a short id. Raises KeyError with a helpful message."""
    if model_id not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"unknown model id '{model_id}'. Known: {available}")
    return MODEL_REGISTRY[model_id]


def list_model_ids() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())
