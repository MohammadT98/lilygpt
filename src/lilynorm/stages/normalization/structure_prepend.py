from __future__ import annotations

import re

from .utils import grab_balanced

RE_ASSIGNMENT = re.compile(
    r"(?m)^\s*([A-Za-z_@][\w@-]*)\s*=\s*(?:\\relative\b[^\n{]*\{|\{)"
)
RE_ASSIGN_NAME = re.compile(r"(?m)^\s*([A-Za-z_@][\w@-]*)\s*=\s*\{")
RE_COMMAND = re.compile(r"\\(key|time|tempo|partial)\b", re.I)
RE_NOTE = re.compile(
    r"\b(?:do|re|mi|fa|sol|la|si|[a-g]|r|R)[isbfes']*\d*\b", re.I
)
KEY_MODES = {
    "major",
    "minor",
    "ionian",
    "dorian",
    "phrygian",
    "lydian",
    "mixolydian",
    "aeolian",
    "locrian",
}


def _extract_command(text: str, idx: int) -> tuple[str, int] | None:
    match = RE_COMMAND.match(text, idx)
    if not match:
        return None
    name = match.group(1).lower()
    end = match.end()
    tail = text[end:]
    if name == "key":
        m = re.match(r"\s*([^\s]+)(?:\s+([^\s]+))?", tail)
        if not m:
            return None
        note = m.group(1)
        mode = m.group(2)
        consume_full = True
        if mode is not None:
            raw_mode = mode.lstrip("\\").lower()
            if raw_mode not in KEY_MODES:
                mode = None
                consume_full = False
        if mode is None and "\\major" in note:
            note, mode = note.split("\\major", 1)
            note = note.strip()
            mode = "\\major"
        elif mode is None and "\\minor" in note:
            note, mode = note.split("\\minor", 1)
            note = note.strip()
            mode = "\\minor"
        if mode is None:
            return None
        cmd = f"\\key {note} {mode}"
        end_pos = end + (m.end() if consume_full else m.end(1))
        return cmd, end_pos
    if name == "time":
        m = re.match(r"\s*([0-9]+/[0-9]+)", tail)
        if not m:
            return None
        cmd = f"\\time {m.group(1)}"
        return cmd, end + m.end()
    if name == "partial":
        m = re.match(r"\s*([0-9.]+)", tail)
        if not m:
            return None
        cmd = f"\\partial {m.group(1)}"
        return cmd, end + m.end()
    if name == "tempo":
        line_end = text.find("\n", end)
        if line_end == -1:
            line_end = len(text)
        cmd = text[idx:line_end].strip()
        return cmd, line_end
    return None


def _extract_structure(text: str) -> str:
    structure: list[str] = []
    seen = set()
    i = 0
    while i < len(text):
        m = RE_COMMAND.search(text, i)
        if not m:
            break
        cmd_info = _extract_command(text, m.start())
        if cmd_info:
            cmd, end = cmd_info
            key = cmd.split(None, 1)[0].lower()
            if key not in seen:
                structure.append(cmd)
                seen.add(key)
            i = end
        else:
            i = m.end()
    return "\n".join(structure).strip()


def _extract_preferred_source(text: str) -> str:
    # Prefer forma blocks first, then any *global block that actually has structure.
    for match in RE_ASSIGN_NAME.finditer(text):
        if match.group(1).lower() != "forma":
            continue
        brace_start = text.find("{", match.end() - 1)
        if brace_start == -1:
            continue
        brace_end = grab_balanced(text, brace_start)
        if brace_end == -1:
            continue
        return text[brace_start + 1:brace_end]

    for match in RE_ASSIGN_NAME.finditer(text):
        if not match.group(1).lower().endswith("global"):
            continue
        brace_start = text.find("{", match.end() - 1)
        if brace_start == -1:
            continue
        brace_end = grab_balanced(text, brace_start)
        if brace_end == -1:
            continue
        block = text[brace_start + 1:brace_end]
        if RE_COMMAND.search(block):
            return block

    return text


def _should_prepend(block: str) -> bool:
    note_match = RE_NOTE.search(block)
    if not note_match:
        return False
    cmd_match = RE_COMMAND.search(block)
    if cmd_match and cmd_match.start() < note_match.start():
        return False
    return True


def run(text: str, _opts) -> str:
    source = _extract_preferred_source(text)
    structure = _extract_structure(source)
    if structure:
        structure = "\n".join(line for line in structure.splitlines() if line.strip())
        structure = re.sub(r"(?m)^\s*\\time\s*$", "", structure)
        structure = "\n".join(line for line in structure.splitlines() if line.strip())
    if not structure:
        return text

    out: list[str] = []
    cursor = 0
    for match in RE_ASSIGNMENT.finditer(text):
        var_name = match.group(1)
        lower_name = var_name.lower()
        if lower_name == "forma" or lower_name.endswith("global"):
            continue
        brace_start = text.find("{", match.end() - 1)
        if brace_start == -1:
            continue
        brace_end = grab_balanced(text, brace_start)
        if brace_end == -1:
            continue
        out.append(text[cursor:brace_start + 1])
        block = text[brace_start + 1:brace_end]
        if _should_prepend(block):
            prefix = "\n" if block.startswith("\n") else " "
            block = f"{prefix}{structure}\n{block.lstrip()}"
        out.append(block)
        cursor = brace_end
    out.append(text[cursor:])
    return "".join(out)
