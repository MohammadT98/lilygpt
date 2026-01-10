from __future__ import annotations

import re
from typing import Tuple

def apply_postprocessing_fixes(text: str) -> str:
    note = r"(?:do|re|mi|fa|sol|la|si|[a-gr])[a-z]*"
    solfege = r"(?:dod|red|mid|fad|sold|lad|sid|do|re|mi|fa|sol|la|si)"

    text = re.sub(r'(^|\n)\\version\s+"[^"]+"\s*', "", text)
    text = re.sub(r'(?m)^\s*#\(set-default-paper-size\s*\)\s*$', "", text)
    text = re.sub(rf"(\b{note}[',]*\d?)\^\s+(?={note})", r"\1 ", text, flags=re.MULTILINE)
    text = re.sub(rf"(\b{note}[',]*\d?)\^(?=\s*{note})", r"\1", text, flags=re.MULTILINE)
    text = re.sub(rf"(\b{note}[',]*\d?)\^\s*$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(rf"(\b{note}[',]*)(128|64|32|16)([1248])\b", r"\1\2 \3", text)
    text = re.sub(r'\\tempo\S*\s*=.*?(?=\s|$)', "", text)  # Remove malformed tempo with = syntax
    text = re.sub(r'(?m)^\s*\\time\s*$', "", text)  # Remove bare \time lines
    text = re.sub(r'\\time(?!\s*\d+/\d+)\b', "", text)  # Remove malformed \time tokens
    text = re.sub(r'(?m)^\s*Staff\s*=\s*(?:up|down)\s*$', "", text)
    text = re.sub(r'(?m)^\s*Voice\s*=\s*$', "", text)
    text = re.sub(r'(?m)^\s*\w+\s*=\s*-column\s*$', "", text)
    text = re.sub(r'(?m)^\s*[\w-]+\.[\w-]+\s*=\s*', "", text)
    text = re.sub(r'\bsuggestAccidentals\s*=\s*##[tf]\b', "", text)
    # Drop structure-only brace lines (artifact from stripping voices).
    struct_cmd = re.compile(r'\\(?:time|key|tempo|partial|repeat|alternative|bar)\b')
    note_token = re.compile(rf'\\b{note}[,\']*\\d*\\b|\\bR\\d*\\b|\\br\\d*\\b', re.I)
    filtered = []
    for line in text.splitlines():
        if struct_cmd.search(line) and line.strip().startswith("{") and "}" in line:
            if not note_token.search(line):
                continue
        filtered.append(line)
    text = "\n".join(filtered) + ("\n" if text.endswith("\n") else "")
    text = re.sub(r'(?m)^\s*\d+\s*$', "", text)  # Remove orphaned numbers (tempo fragments)
    text = re.sub(r'Rest\s+#\'.*?(?=\s|$)', "", text)  # Remove Rest with scheme properties
    text = re.sub(r'#\'[\s\w\-\(\.\-\)]+', "", text)  # Remove scheme code fragments and pairs
    text = re.sub(r'(?m)^\s*\\mark\s*"?[^"\n]*"?\s*$', "", text)
    text = re.sub(r'\{\s*"\s*"\s*\}', "", text)
    text = re.sub(r'Score\.measureLength\s*=\s*#\(.*?\)\s*', "", text)
    text = re.sub(r'R\d+\^', "", text)  # Remove malformed rests with articulation marks
    text = re.sub(r"(?m)^\s*<<[^>]*$", "", text)
    text = re.sub(r"\[\s*tr\s*\]", "", text)
    text = re.sub(rf"\b({solfege})([',]*?)({solfege})([',]*\d*)\b", r"\1\2 \3\4", text)
    text = close_unclosed_braces(text)
    text = re.sub(r'Staff\s+\{[^}]*\}[^{<]*(\{|<<)', r'\1', text, flags=re.DOTALL)

    return text


def close_unclosed_braces(text: str) -> str:
    assign_re = re.compile(r"^[A-Za-z_][\w-]*\s*=")
    lines = text.splitlines()
    out = []
    depth = 0

    for line in lines:
        if assign_re.match(line) and depth > 0:
            out.extend(["}"] * depth)
            depth = 0
        out.append(line)
        line_no_comment = line.split("%", 1)[0]
        depth += line_no_comment.count("{") - line_no_comment.count("}")
        depth = max(0, depth)

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _grab_balanced(text: str, start: int, open_char: str = "{", close_char: str = "}") -> int:
    depth = 1
    for i in range(start + 1, len(text)):
        if text[i] == open_char:
            depth += 1
        elif text[i] == close_char:
            depth -= 1
            if depth == 0:
                return i
    return -1


def remove_empty_variable_assignments(text: str) -> Tuple[str, int]:
    def _is_empty_content(content: str) -> bool:
        s = re.sub(r'\\clef\s+\w+', '', content)
        s = re.sub(r'\\(?:key|time|tempo|partial)\b[^\n]*', '', s)
        s = re.sub(r'(?m)^\s*Staff\s*=\s*(?:up|down)\s*$', '', s)
        s = re.sub(r'(?m)^\s*Voice\s*=\s*$', '', s)
        s = re.sub(r'\b(?:s\d*\.?)(?:\*\d+)?(?:\s*[-_^]+\s*)?', '', s)
        s = re.sub(r'[\s\n]+', '', s)
        while '{}' in s:
            s = s.replace('{}', '')
        return not s.strip()

    def _is_structure_only(content: str) -> bool:
        s = re.sub(r'\\(?:key|time|tempo|partial)\b[^\n]*', '', content)
        s = re.sub(r'(?m)^\s*Staff\s*=\s*(?:up|down)\s*$', '', s)
        s = re.sub(r'(?m)^\s*Voice\s*=\s*$', '', s)
        s = re.sub(r'\b(?:s\d*\.?)(?:\*\d+)?(?:\s*[-_^]+\s*)?', '', s)
        s = re.sub(r'[{}\s\n]+', '', s)
        return not s.strip()

    empty_vars = []
    pattern = re.compile(
        r"(?m)^([A-Za-z_][\w-]*)\s*=\s*(?:\\relative\b[^\n{]*\{|\\transpose\b[^\n{]*\{|\\absolute\b[^\n{]*\{|\\drummode\b[^\n{]*\{|\{)"
    )
    i = 0

    while i < len(text):
        match = pattern.search(text, i)
        if not match:
            break

        var_name = match.group(1)
        if re.search(r"cadenza", var_name, re.I):
            assignment_end = _grab_balanced(text, text.find("{", match.end() - 1))
            if assignment_end != -1:
                assignment_end += 1
                while assignment_end < len(text) and text[assignment_end] == '\n':
                    assignment_end += 1
                empty_vars.append((match.start(), assignment_end, var_name))
            i = match.end()
            continue
        brace_start = text.find("{", match.end() - 1)
        brace_end = _grab_balanced(text, brace_start)

        if brace_end != -1:
            body = text[brace_start + 1:brace_end]
            if _is_empty_content(body) or _is_structure_only(body):
                assignment_end = brace_end + 1
                while assignment_end < len(text) and text[assignment_end] == '\n':
                    assignment_end += 1
                empty_vars.append((match.start(), assignment_end, var_name))
            i = brace_end + 1
        else:
            i = match.end()

    for start, end, var_name in reversed(empty_vars):
        text = text[:start] + text[end:]
        text = re.sub(rf"\\{re.escape(var_name)}\b", "", text)

    text, count = re.subn(r"(?m)^[A-Za-z_][\w-]*\s*=\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text, len(empty_vars) + count
