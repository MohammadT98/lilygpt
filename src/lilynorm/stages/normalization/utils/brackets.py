"""Balanced-bracket helpers for normalization."""

from __future__ import annotations


def grab_balanced(text: str, start: int, open_char: str = "{", close_char: str = "}") -> int:
    """Find the closing bracket position for balanced brackets.

    Args:
        text: The source string
        start: Position of the opening bracket
        open_char: Opening bracket character (default '{')
        close_char: Closing bracket character (default '}')

    Returns:
        Position of matching closing bracket, or -1 if not found
    """
    depth = 1
    for i in range(start + 1, len(text)):
        if text[i] == open_char:
            depth += 1
        elif text[i] == close_char:
            depth -= 1
            if depth == 0:
                return i
    return -1


def grab_angles(text: str, start: int) -> int:
    """Find the closing >> position for balanced << >> angle brackets.

    Args:
        text: The source string
        start: Position of the opening <<

    Returns:
        Position after the matching >>, or -1 if not found
    """
    depth = 1
    i = start + 2
    while i < len(text) and depth > 0:
        if text.startswith("<<", i):
            depth += 1
            i += 2
        elif text.startswith(">>", i):
            depth -= 1
            i += 2
        else:
            i += 1
    return i if depth == 0 else -1


__all__ = ["grab_balanced", "grab_angles"]
