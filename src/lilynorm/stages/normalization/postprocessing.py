from __future__ import annotations

import re
from typing import Tuple


def apply_postprocessing_fixes(text: str) -> str:
    cleaned = text

    cleaned = re.sub(r'(^|\n)\\version\s+"[^"]+"\s*', "", cleaned)

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

    cleaned = re.sub(
        rf"(\b{note_token}[',]*)(128|64|32|16)([1248])\b",
        r"\1\2 \3",
        cleaned,
    )

    cleaned = re.sub(r"(?m)^\s*\\tempo\s+.*$", "", cleaned)

    cleaned = re.sub(r'(?m)^\s*\\mark\s*"?[^"\n]*"?\s*$', "", cleaned)

    cleaned = re.sub(r'\{\s*"\s*"\s*\}', "", cleaned)

    cleaned = re.sub(r"(?m)^\s*<<[^>]*$", "", cleaned)

    cleaned = re.sub(r"\[\s*tr\s*\]", "", cleaned)

    solfege = r"(?:dod|red|mid|fad|sold|lad|sid|do|re|mi|fa|sol|la|si)"
    cleaned = re.sub(
        rf"\b({solfege})([',]*?)({solfege})([',]*\d*)\b",
        r"\1\2 \3\4",
        cleaned,
    )

    cleaned = close_unclosed_braces(cleaned)

    cleaned = re.sub(r'Staff\s+\{[^}]*\}[^{<]*(\{|<<)', r'\1', cleaned, flags=re.DOTALL)

    return cleaned


def close_unclosed_braces(text: str) -> str:
    assign_re = re.compile(r"^[A-Za-z_][\w-]*\s*=")
    lines = text.splitlines()
    out: list[str] = []
    depth = 0

    for line in lines:
        if assign_re.match(line) and depth > 0:
            out.extend(["}"] * depth)
            depth = 0
        out.append(line)

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
    removed = 0

    def _is_empty_content(content: str) -> bool:
        stripped = re.sub(r'\\clef\s+\w+', '', content)
        stripped = re.sub(r'[\s\n]+', '', stripped)
        while '{}' in stripped:
            stripped = stripped.replace('{}', '')
        return len(stripped.strip()) == 0

    empty_var_ranges = []
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
            content = text[brace_start + 1:brace_end]

            if _is_empty_content(content):
                assignment_end = brace_end + 1
                while assignment_end < len(text) and text[assignment_end] == '\n':
                    assignment_end += 1
                empty_var_ranges.append((assignment_start, assignment_end, var_name))
                removed += 1

            index = brace_end + 1
        else:
            index = match.end()

    for start, end, var_name in reversed(empty_var_ranges):
        text = text[:start] + text[end:]
        ref_pattern = re.compile(rf"\\{re.escape(var_name)}\b")
        text = ref_pattern.sub("", text)

    pattern2 = re.compile(r"(?m)^[A-Za-z_][\w-]*\s*=\s*$")
    text, count2 = pattern2.subn("", text)
    removed += count2

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, removed
