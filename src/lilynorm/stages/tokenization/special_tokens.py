from __future__ import annotations

import re
from typing import Iterable, List

from transformers import PreTrainedTokenizer

DEFAULT_SPECIAL_TOKENS: List[str] = [
    "<KEY:",
    "<TIME:",
    "<TEMPO:",
    "<VOICE:",
    "<BAR>",
]

KEY_TOKEN = "<KEY:"
TIME_TOKEN = "<TIME:"
TEMPO_TOKEN = "<TEMPO:"
VOICE_TOKEN = "<VOICE:"
BAR_TOKEN = "<BAR>"
CLOSE_TOKEN = ">"

RE_KEY_SIG = re.compile(r"\\key\s+([a-g](?:is|es)?)\s+\\(major|minor)", re.I)
RE_TIME_SIG = re.compile(r"\\time\s+(\d+/\d+)", re.I)
RE_TEMPO = re.compile(r"\\tempo\s+([^\\]+)", re.I)
RE_VOICE = re.compile(r"\\voice(One|Two|Three|Four)\b")
RE_BAR = re.compile(r"\|")

VOICE_MAP = {
    "One": "1",
    "Two": "2",
    "Three": "3",
    "Four": "4",
}


def add_structural_tokens(text: str) -> str:
    text = RE_KEY_SIG.sub(
        lambda m: f"{KEY_TOKEN}{m.group(1)}_{m.group(2)}{CLOSE_TOKEN}",
        text,
    )
    text = RE_TIME_SIG.sub(
        lambda m: f"{TIME_TOKEN}{m.group(1)}{CLOSE_TOKEN}",
        text,
    )
    text = RE_TEMPO.sub(
        lambda m: f"{TEMPO_TOKEN}{m.group(1).strip()}{CLOSE_TOKEN}",
        text,
    )
    text = RE_VOICE.sub(
        lambda m: f"{VOICE_TOKEN}{VOICE_MAP.get(m.group(1), m.group(1))}{CLOSE_TOKEN}",
        text,
    )
    text = RE_BAR.sub(f" {BAR_TOKEN} ", text)
    return text


def build_special_tokens(extra_tokens: Iterable[str] | None = None) -> list[str]:
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
    tokens = build_special_tokens(extra_tokens)
    if not tokens:
        return 0
    return tokenizer.add_special_tokens({"additional_special_tokens": tokens})
