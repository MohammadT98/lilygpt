"""Generation regimes.

A :class:`Regime` describes how a :class:`~lilybench.generation.prompt_bank.Prompt`
is materialised into a chat prompt for a backbone tokenizer. Subclass
:class:`Regime` and register the result with :func:`register_regime` to
add a new regime without touching the runner.

The paper compares three regimes:

* ``zero``: a short instruction-tuned system prompt plus the bank's
  metadata block and user prompt.
* ``few``: three demonstrations sampled from the training distribution.
* ``few_ablation``: three hand-written A-minor demonstrations retained
  as an ablation (§3 of the paper).

``few`` and ``few_ablation`` share the same :class:`FewShot` class with a
different demonstration source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from lilybench.generation.metadata_block import render_metadata_block
from lilybench.generation.prompt_bank import Prompt


_SYSTEM_PROMPT = (
    "You are a LilyPond assistant. Output only valid LilyPond code, "
    "no prose, no markdown."
)


@dataclass(frozen=True)
class Regime:
    """Base class for generation regimes. Override :meth:`build_prompt`."""

    name: ClassVar[str] = "base"

    def build_prompt(self, prompt: Prompt, *, tokenizer: Any) -> str:
        raise NotImplementedError


def _apply_chat(messages: list[dict], tokenizer) -> str:
    if not getattr(tokenizer, "chat_template", None):
        # Fall back for non-instruction-tuned backbones.
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else _SYSTEM_PROMPT
        user = messages[-1]["content"]
        return f"### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n"
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@dataclass(frozen=True)
class ZeroShot(Regime):
    """Plain instruction + metadata block + user prompt."""

    name: ClassVar[str] = "zero"

    def build_prompt(self, prompt: Prompt, *, tokenizer: Any) -> str:
        user = render_metadata_block(prompt.metadata) + prompt.user_prompt
        return _apply_chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            tokenizer,
        )


@dataclass(frozen=True)
class FewShot(Regime):
    """Prepend a text block of demonstrations to every user message.

    The exact demonstrations live in a separate file so the regime is
    completely declarative — swap the file (and the regime name) to test
    a different ablation, no code changes required.
    """

    name: ClassVar[str] = "few"
    demonstrations: str = ""

    @classmethod
    def from_file(cls, demos_file: str | Path, *, name: str | None = None) -> "FewShot":
        text = Path(demos_file).read_text(encoding="utf-8")
        if name is None:
            return cls(demonstrations=text)
        # ``name`` is a ClassVar on the dataclass; we shadow it on the instance
        # by subclassing on the fly so the runner can dispatch by .name.
        new_cls = type(f"FewShot_{name}", (cls,), {"name": name})
        return new_cls(demonstrations=text)

    def build_prompt(self, prompt: Prompt, *, tokenizer: Any) -> str:
        user = render_metadata_block(prompt.metadata) + prompt.user_prompt
        if self.demonstrations:
            user = f"{self.demonstrations.strip()}\n\n{user}"
        return _apply_chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            tokenizer,
        )


REGIME_REGISTRY: dict[str, type[Regime]] = {
    ZeroShot.name: ZeroShot,
    FewShot.name: FewShot,
}


def register_regime(regime_cls: type[Regime]) -> None:
    """Add a new regime class to the dispatch registry."""
    if regime_cls.name in REGIME_REGISTRY:
        raise KeyError(f"regime {regime_cls.name!r} already registered")
    REGIME_REGISTRY[regime_cls.name] = regime_cls
