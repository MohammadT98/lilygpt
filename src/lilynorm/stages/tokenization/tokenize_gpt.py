from __future__ import annotations

import functools
import re
from typing import Dict, List, Optional

try:
    from transformers import AutoTokenizer
except ImportError as exc:
    raise RuntimeError(
        "transformers is required for GPT tokenization. "
        "Install it with `pip install transformers`."
    ) from exc

# Use GPT-OSS-20B by default
DEFAULT_MODEL_NAME = "openai/gpt-oss-20b"

# Default max length for sequences (can override via run())
DEFAULT_MAX_LENGTH = 1024


@functools.lru_cache(maxsize=4)
def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Ensure pad token exists (for decoder-only GPT-style models)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def run(
    text: str,
    *,
    model_name: Optional[str] = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, object]:
    """
    Tokenize a LilyPond string using a Hugging Face tokenizer.

    Returns a dict similar to Matteo's _tokenize_text():

        {
            "input_ids": [...],
            "attention_mask": [...],
            "token_count": int,
            "truncated": bool,
        }
    """
    name = model_name or DEFAULT_MODEL_NAME
    tokenizer = _load_tokenizer(name)

    # Stage 1: basic text normalization
    text = _normalize_whitespace(text)

    # Stage 2: HF tokenizer
    encoded = tokenizer(
        text,
        add_special_tokens=False,   # or True, if you want BOS/EOS etc.
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    input_ids: List[int] = list(encoded["input_ids"])
    attention_mask: List[int] = list(encoded["attention_mask"])
    token_count = len(input_ids)
    truncated = token_count >= max_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_count": token_count,
        "truncated": truncated,
    }