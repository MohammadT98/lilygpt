from __future__ import annotations

import re
from typing import Iterable, List

from transformers import PreTrainedTokenizer

DEFAULT_SPECIAL_TOKENS: List[str] = [
    "<KEY:",
    "<TIME:",
    "<TEMPO:",
    "<VOICE>",
    "<BAR>",
]

KEY_TOKEN = "<KEY:"
TIME_TOKEN = "<TIME:"
TEMPO_TOKEN = "<TEMPO:"
BARE_VOICE_TOKEN = "<VOICE>"
BAR_TOKEN = "<BAR>"
CLOSE_TOKEN = ">"

RE_KEY_SIG = re.compile(
    r"\\key\s+((?:[a-g](?:is|es)?|do|re|mi|fa|sol|la|si)(?:d|b)?)\s*\\?(major|minor)\b",
    re.I,
)
RE_TIME_SIG = re.compile(r"\\time\s+(\d+)\s*/\s*(\d+)", re.I)
RE_TEMPO = re.compile(r"\\tempo\s+([^\n\\\\]+)", re.I)
RE_BAR_CMD = re.compile(r"\\bar\s+\"?[^\s\"}]+\"?", re.I)
RE_BAR_PIPE = re.compile(r"\|")


def add_structural_tokens(text: str) -> str:
    """Insert structural tokens for key, time, tempo, and bars."""
    text = RE_KEY_SIG.sub(
        lambda m: f"{KEY_TOKEN}{m.group(1)}_{m.group(2)}{CLOSE_TOKEN}",
        text,
    )
    text = RE_TIME_SIG.sub(
        lambda m: f"{TIME_TOKEN}{m.group(1)}/{m.group(2)}{CLOSE_TOKEN}",
        text,
    )
    text = RE_TEMPO.sub(
        lambda m: f"{TEMPO_TOKEN}{m.group(1).strip()}{CLOSE_TOKEN}",
        text,
    )
    text = RE_BAR_CMD.sub(f" {BAR_TOKEN} ", text)
    text = RE_BAR_PIPE.sub(f" {BAR_TOKEN} ", text)
    return text


def add_voice_label(text: str, var_name: str | None) -> str:
    """Prefix a block with a bare voice label."""
    if not text or not var_name:
        return text

    if var_name.lower().endswith("global"):
        return text

    if text.lstrip().startswith(BARE_VOICE_TOKEN):
        return text

    return f"{BARE_VOICE_TOKEN} {text}"


def build_special_tokens(extra_tokens: Iterable[str] | None = None) -> list[str]:
    """Return a de-duplicated list of special tokens."""
    tokens = list(DEFAULT_SPECIAL_TOKENS)
    if extra_tokens:
        tokens.extend(extra_tokens)

    seen = set()
    ordered: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def apply_special_tokens(
    tokenizer: PreTrainedTokenizer,
    extra_tokens: Iterable[str] | None = None,
) -> int:
    """Register special tokens on the tokenizer."""
    tokens = build_special_tokens(extra_tokens)
    if not tokens:
        return 0
    return tokenizer.add_special_tokens({"additional_special_tokens": tokens})
