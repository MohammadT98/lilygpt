"""Task definitions, prompt templates, and option samplers.

Strings are kept verbatim against the prompt-template table in
arXiv-2509.23350v1, with one adaptation: ``Input`` carries a LilyPond score
rather than ABC notation.

The eight tasks here are a subset of the paper's ten: emotion recognition and
error detection are dropped because the Mutopia corpus carries no labels for
either.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class TaskSpec:
    """Per-task configuration consumed by the bench builder."""
    name: str
    n: int                                  # target sample count per the paper
    template_kind: str                       # "multiple_choice" | "structured_output"
    task_instruction: str
    structured_output_template: str = ""     # only used by structured tasks
    extra: dict = field(default_factory=dict)


# ---- Prompt templates (verbatim from the paper, modulo ``Input`` content) ----

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


def format_mc_prompt(
    *,
    input_content: str,
    task_instruction: str,
    options: Sequence[str],
) -> str:
    """Render the multiple-choice prompt template."""
    if len(options) != 4:
        raise ValueError(f"MC prompts require exactly 4 options, got {len(options)}")
    return _MC_TEMPLATE.format(
        input_content=input_content,
        task_instruction=task_instruction,
        opt0=options[0],
        opt1=options[1],
        opt2=options[2],
        opt3=options[3],
    )


def format_structured_prompt(
    *,
    input_content: str,
    task_instruction: str,
    structured_output_template: str,
) -> str:
    """Render the structured-output prompt template."""
    return _STRUCTURED_TEMPLATE.format(
        input_content=input_content,
        task_instruction=task_instruction,
        structured_output_template=structured_output_template,
    )


def sample_distractors(
    pool: Sequence[str],
    gold: str,
    rng: random.Random,
    k: int = 3,
) -> list[str]:
    """Sample ``k`` distinct distractors from ``pool``, excluding ``gold``.

    Deterministic given ``rng``. Raises ``ValueError`` if the pool has fewer
    than ``k`` candidates after removing the gold.
    """
    candidates = [x for x in pool if x != gold]
    candidates = list(dict.fromkeys(candidates))  # dedupe, preserve order
    if len(candidates) < k:
        raise ValueError(
            f"pool too small for {k} distractors (have {len(candidates)} after "
            f"removing gold={gold!r})"
        )
    return rng.sample(candidates, k)


# ---- Task registry --------------------------------------------------------

TASKS: dict[str, TaskSpec] = {
    "bar_count": TaskSpec(
        name="bar_count",
        n=100,
        template_kind="structured_output",
        task_instruction=(
            "Count the number of bars in the LilyPond score above. "
            "Output a single integer."
        ),
        structured_output_template="<integer>",
    ),
    "metadata_qa": TaskSpec(
        name="metadata_qa",
        n=60,
        template_kind="multiple_choice",
        task_instruction=(
            "Select the value of the queried metadata field from the four options. "
            "The relevant field is given in the question."
        ),
    ),
    "bar_sequencing": TaskSpec(
        name="bar_sequencing",
        n=119,
        template_kind="structured_output",
        task_instruction=(
            "The bars below are shuffled and labelled with indices 0..3. "
            "Output the indices in the correct sequential order as a single "
            "digit string (for example, 0312)."
        ),
        structured_output_template="<four-digit index permutation, no separator>",
    ),
    "next_bar_prediction": TaskSpec(
        name="next_bar_prediction",
        n=119,
        template_kind="multiple_choice",
        task_instruction=(
            "Given the opening bars of the score, select the option that is "
            "most likely to be the next bar."
        ),
    ),
    "metadata_prediction": TaskSpec(
        name="metadata_prediction",
        n=60,
        template_kind="multiple_choice",
        task_instruction=(
            "The score has one metadata field masked. Predict the value of "
            "that field from the four options based on the musical content."
        ),
    ),
    "music_captioning": TaskSpec(
        name="music_captioning",
        n=60,
        template_kind="multiple_choice",
        task_instruction=(
            "Select the most likely title for the score from the four options."
        ),
    ),
    "composer_recognition": TaskSpec(
        name="composer_recognition",
        n=96,
        template_kind="multiple_choice",
        task_instruction=(
            "Select the most likely composer of the score from the four options."
        ),
    ),
    "genre_recognition": TaskSpec(
        name="genre_recognition",
        n=132,
        template_kind="multiple_choice",
        task_instruction=(
            "Select the most likely genre of the score from the four options."
        ),
    ),
}
