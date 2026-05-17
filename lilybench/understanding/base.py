"""Abstract base class + shared prompt templates for understanding tasks.

Each task implements two operations:

* :meth:`UnderstandingTask.build` — sample ``n`` records from a corpus,
  produce :class:`UnderstandingRecord` objects (prompt, options, gold).
* :meth:`UnderstandingTask.score` — given the bench records and the
  parsed model predictions, return a per-task score dict.

The two prompt-template constants come straight from ABC-Eval (Zhao et
al. 2026) — ``Input`` carries a LilyPond score in our adaptation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from lilybench.data.types import CorpusEntry


_MC_TEMPLATE = (
    "Input: {input_content}\n"
    "Task: {task_instruction}\n"
    "Options:\n"
    "0. {opt0}\n"
    "1. {opt1}\n"
    "2. {opt2}\n"
    "3. {opt3}\n\n"
    "Please only output the index of the correct option (0, 1, 2, or 3), "
    "do not output any additional content."
)

_STRUCTURED_TEMPLATE = (
    "Input: {input_content}\n"
    "Task: {task_instruction}\n"
    "Template: {structured_output_template}\n\n"
    "Please directly output the answer of the given task, "
    "without any explanation or additional content."
)

PROMPT_TEMPLATES = {
    "multiple_choice": _MC_TEMPLATE,
    "structured_output": _STRUCTURED_TEMPLATE,
}


def format_mc_prompt(
    *, input_content: str, task_instruction: str, options: Sequence[str]
) -> str:
    if len(options) != 4:
        raise ValueError(f"MC prompts require exactly 4 options, got {len(options)}")
    return _MC_TEMPLATE.format(
        input_content=input_content,
        task_instruction=task_instruction,
        opt0=options[0], opt1=options[1], opt2=options[2], opt3=options[3],
    )


def format_structured_prompt(
    *, input_content: str, task_instruction: str, structured_output_template: str
) -> str:
    return _STRUCTURED_TEMPLATE.format(
        input_content=input_content,
        task_instruction=task_instruction,
        structured_output_template=structured_output_template,
    )


@dataclass
class UnderstandingRecord:
    """One bench record, written to JSONL as the input for the runner."""

    task: str
    id: str
    source_file: str
    input_content: str
    task_instruction: str
    prompt: str
    template_kind: str
    gold: Any = None
    options: list[str] = field(default_factory=list)
    gold_index: int | None = None
    structured_output_template: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Hoist task-specific extras to the top level so downstream code
        # (scoring, analysis) does not need to know about the field.
        extras = d.pop("extras")
        d.update(extras)
        # Drop fields that are not used by the current task kind.
        if not d["options"]:
            d.pop("options")
            d.pop("gold_index", None)
        if d["structured_output_template"] is None:
            d.pop("structured_output_template", None)
        return d


class UnderstandingTask(ABC):
    """One ABC-Eval-style task adapted to LilyPond.

    Subclasses set ``name``, ``template_kind`` (``"multiple_choice"`` or
    ``"structured_output"``), ``task_instruction``, and optionally
    ``structured_output_template`` / ``default_n`` / ``corpus_kind``.
    """

    name: str = ""
    template_kind: str = "multiple_choice"
    task_instruction: str = ""
    structured_output_template: str = ""
    default_n: int = 60
    corpus_kind: str = "mutopia"  # one of {"mutopia", "emopia"}

    @abstractmethod
    def build(
        self, corpus: Sequence[CorpusEntry], *, n: int, seed: int
    ) -> list[UnderstandingRecord]:
        """Return up to ``n`` records sampled from ``corpus``."""

    @abstractmethod
    def score(
        self, bench: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Aggregate per-task metrics over aligned ``(record, prediction)`` pairs."""
