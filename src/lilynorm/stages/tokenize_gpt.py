
from __future__ import annotations

import functools
from typing import Iterable, List, Optional

try:
    from transformers import AutoTokenizer 
except ImportError as exc: 
    raise RuntimeError(
        "transformers is required for GPT tokenization. "
        "Install it with `pip install transformers`."
    ) from exc

DEFAULT_MODEL_NAME = "EleutherAI/gpt-neox-20b"

@functools.lru_cache(maxsize=4)
def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def run(text: str, *, model_name: Optional[str] = None) -> List[int]:
    name = model_name or DEFAULT_MODEL_NAME
    tokenizer = _load_tokenizer(name)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return list(encoded["input_ids"])