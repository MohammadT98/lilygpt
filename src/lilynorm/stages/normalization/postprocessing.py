"""
Post-processing fixes for malformed LilyPond patterns and cleanup operations.
"""
from __future__ import annotations

import re
from typing import Tuple


def apply_postprocessing_fixes(text: str) -> str:
    """
    Fix common malformed patterns in baroque LilyPond scores.

    Handles edge cases like stray articulations, glued durations, unclosed braces, etc.
    """
    cleaned = text

    # Remove \version declarations (standardized later if needed)
    cleaned = re.sub(r'(^|\n)\\version\s+"[^"]+"\s*', "", cleaned)

    # Remove stray caret articulations (mi^ la → mi la)
    note_token = r"(?:do|re|mi|fa|sol|la|si|[a-gr])[a-z]*"
    cleaned = re.sub(
        rf"(\b{note_token}[',]*\d?)\^\s+(?={note_token})",
        r"\1 ",
        cleaned,
        flags=re.MULTILINE,
    )
    cleaned = re.sub(
        rf"(\b{note_token}[',]*\d?)\^(?=\s*{note_token})",
        r"\1",
        cleaned,
        flags=re.MULTILINE,
    )
    cleaned = re.sub(
        rf"(\b{note_token}[',]*\d?)\^\s*$",
        r"\1",
        cleaned,
        flags=re.MULTILINE,
    )

    # Fix glued durations (la168 → la16 8)
    cleaned = re.sub(
        rf"(\b{note_token}[',]*)(128|64|32|16)([1248])\b",
        r"\1\2 \3",
        cleaned,
    )

    # Remove malformed \tempo directives
    cleaned = re.sub(r"(?m)^\s*\\tempo\s+.*$", "", cleaned)

    # Drop bare \mark lines
    cleaned = re.sub(r'(?m)^\s*\\mark\s*"?[^"\n]*"?\s*$', "", cleaned)

    # Remove empty text blocks
    cleaned = re.sub(r'\{\s*"\s*"\s*\}', "", cleaned)

    # Remove broken polyphonic openings
    cleaned = re.sub(r"(?m)^\s*<<[^>]*$", "", cleaned)

    # Remove stray ornament tokens
    cleaned = re.sub(r"\[\s*tr\s*\]", "", cleaned)

    # Split glued solfege note names (faddod → fad dod)
    solfege = r"(?:dod|red|mid|fad|sold|lad|sid|do|re|mi|fa|sol|la|si)"
    cleaned = re.sub(
        rf"\b({solfege})([',]*?)({solfege})([',]*\d*)\b",
        r"\1\2 \3\4",
        cleaned,
    )

    # Close unclosed braces before assignments
    cleaned = close_unclosed_braces(cleaned)

    # Remove Staff context blocks from displayLilyMusic output
    cleaned = re.sub(r'Staff\s+\{[^}]*\}[^{<]*(\{|<<)', r'\1', cleaned, flags=re.DOTALL)

    return cleaned


def close_unclosed_braces(text: str) -> str:
    """
    Close any unclosed braces before top-level variable assignments.

    Prevents syntax errors from malformed nesting.
    """
    assign_re = re.compile(r"^[A-Za-z_][\w-]*\s*=")
    lines = text.splitlines()
    out: list[str] = []
    depth = 0

    for line in lines:
        if assign_re.match(line) and depth > 0:
            out.extend(["}"] * depth)
            depth = 0
        out.append(line)

        # Track brace depth (ignore braces in comments)
        line_no_comment = line.split("%", 1)[0]
        depth += line_no_comment.count("{") - line_no_comment.count("}")
        if depth < 0:
            depth = 0

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _grab_balanced(
    text: str,
    start: int,
    open_char: str = "{",
    close_char: str = "}",
) -> int:
    """
    Given an index `start` pointing at an opening brace (or other delimiter),
    return the index of the matching closing delimiter, handling nested pairs.

    Returns -1 if the text ends before a matching closing char is found.
    """
    depth = 1
    index = start + 1
    length = len(text)

    while index < length:
        char = text[index]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
        index += 1

    return -1


def remove_empty_variable_assignments(text: str) -> Tuple[str, int]:
    """Remove variable assignments that are empty (leftover from content removal).

    Handles both simple empty blocks (varname = { }) and nested empty blocks
    (varname = { {} }).

    Also removes all references to those empty variables to avoid undefined references.

    Returns (cleaned_text, removed_count).
    """
    removed = 0

    def _is_empty_content(content: str) -> bool:
        """Check if content is effectively empty (only whitespace, nested empty braces, or clef directives)."""
        # Remove clef directives first (before whitespace removal, since they contain spaces)
        stripped = re.sub(r'\\clef\s+\w+', '', content)
        # Remove all whitespace and newlines
        stripped = re.sub(r'[\s\n]+', '', stripped)
        # Remove all nested empty braces {} recursively
        while '{}' in stripped:
            stripped = stripped.replace('{}', '')
        # If nothing remains, it's empty
        return len(stripped.strip()) == 0

    # First pass: find all empty variable assignments and their positions
    empty_var_ranges = []  # List of (start_pos, end_pos, var_name)
    pattern = re.compile(r"(?m)^([A-Za-z_][\w-]*)\s*=\s*\{")
    index = 0

    while index < len(text):
        match = pattern.search(text, index)
        if not match:
            break

        var_name = match.group(1)
        assignment_start = match.start()
        brace_start = match.end() - 1  # Position of opening brace
        brace_end = _grab_balanced(text, brace_start, "{", "}")

        if brace_end != -1:
            # Extract content between braces
            content = text[brace_start + 1:brace_end]

            # Check if content is effectively empty
            if _is_empty_content(content):
                # Find the end of the assignment (including all trailing newlines)
                assignment_end = brace_end + 1
                # Include all trailing newlines
                while assignment_end < len(text) and text[assignment_end] == '\n':
                    assignment_end += 1
                empty_var_ranges.append((assignment_start, assignment_end, var_name))
                removed += 1

            index = brace_end + 1
        else:
            index = match.end()

    # Second pass: remove empty variable definitions (in reverse order to preserve indices)
    for start, end, var_name in reversed(empty_var_ranges):
        text = text[:start] + text[end:]
        # Remove all references to this variable (e.g., \Ivl)
        ref_pattern = re.compile(rf"\\{re.escape(var_name)}\b")
        text = ref_pattern.sub("", text)

    # Match variable assignment with no value (just whitespace after =)
    pattern2 = re.compile(r"(?m)^[A-Za-z_][\w-]*\s*=\s*$")
    text, count2 = pattern2.subn("", text)
    removed += count2

    # Clean up extra blank lines from removals
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, removed
