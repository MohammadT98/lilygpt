from __future__ import annotations
from typing import Literal, Union

FormatType = Literal["none", "plain", "chatml"]

Formatted = Union[str, tuple[str, str]]

def format_example(
    text: str,
    instruction_format: FormatType = "none",
    instruction: str = "Generate LilyPond music notation.",
) -> Formatted:
    if not text or not text.strip():
        text = ""

    if instruction_format == "none":
        return text

    if instruction_format == "plain":
        return f"{instruction}\n{text}"

    if instruction_format == "chatml":
        prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
        answer = text
        return (prompt, answer)

    raise ValueError(f"Unknown instruction format: {instruction_format}")

def format_full_text(
    text: str,
    instruction_format: FormatType = "none",
    instruction: str = "Generate LilyPond music notation.",
) -> str:
    result = format_example(text, instruction_format, instruction)
    if isinstance(result, tuple):
        return result[0] + result[1]
    return result