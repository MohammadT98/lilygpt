"""
Dataset formatting utilities for instruction-following training.

Supports multiple prompt formats for different base models.
"""

from __future__ import annotations
from typing import Literal, Union

FormatType = Literal["none", "plain", "chatml"]

# Return type: either single string or (prompt, answer) tuple
Formatted = Union[str, tuple[str, str]]


def format_example(
    text: str,
    instruction_format: FormatType = "none",
    instruction: str = "Generate LilyPond music notation.",
) -> Formatted:
    """
    Format a normalized LilyPond text with instruction wrapping.

    Args:
        text: The normalized LilyPond music code
        instruction_format: The format style to use
            - "none": No wrapping, return text as-is
            - "plain": Simple text prefix "{instruction}\n{text}"
            - "chatml": ChatML format with <|user|>/<|assistant|> tokens,
              returns (prompt, answer) tuple for prompt/answer masking
        instruction: The user instruction to prepend

    Returns:
        - "none" / "plain": a single concatenated string
        - "chatml": (prompt_text, answer_text) tuple for label masking during training

    Example:
        >>> result = format_example(r"\\version \"2.24.4\"\\nIvlIn = ...", instruction_format="chatml")
        >>> prompt, answer = result
        >>> # Use prompt for masking in labels, answer is the music code
    """
    if not text or not text.strip():
        # Handle empty input gracefully
        text = ""

    if instruction_format == "none":
        return text

    if instruction_format == "plain":
        return f"{instruction}\n{text}"

    if instruction_format == "chatml":
        # ChatML format with separators for clean tokenization
        # Prevents model from learning glued tokens like notation.<|assistant|>
        prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
        answer = text
        return (prompt, answer)

    raise ValueError(f"Unknown instruction format: {instruction_format}")


def format_full_text(
    text: str,
    instruction_format: FormatType = "none",
    instruction: str = "Generate LilyPond music notation.",
) -> str:
    """
    Always returns a single concatenated string (prompt + answer).

    Use this when your training script does NOT support prompt/answer masking.
    """
    result = format_example(text, instruction_format, instruction)
    if isinstance(result, tuple):
        return result[0] + result[1]
    return result
