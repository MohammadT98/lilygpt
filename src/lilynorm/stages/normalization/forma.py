"""Inject or inline forma structure into LilyPond assignment blocks."""

from __future__ import annotations

import re

from .utils import grab_balanced, grab_angles

RE_ASSIGNMENT = re.compile(
    r"(?m)^\s*([A-Za-z_@][\w@-]*)\s*=\s*(?:\\relative\b[^\n{]*\{|\{)"
)
RE_ASSIGN_NAME = re.compile(r"(?m)^\s*([A-Za-z_@][\w@-]*)\s*=\s*\{")
RE_COMMAND = re.compile(r"\\(key|time|tempo|partial)\b", re.I)
RE_NOTE = re.compile(r"\b(?:do|re|mi|fa|sol|la|si|[a-g]|r|R)[isbfes']*\d*\b", re.I)
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


FORMA_START = re.compile(r"(?m)^forma\s*=\s*\{")
SEMANTIC_CMD_RE = re.compile(
    r"""
    \\
    (?:
        key|time|tempo|partial|repeat|alternative|bar
    )
    \b
    """,
    re.X | re.I,
)
SKIP_RE = re.compile(r"\bs[0-9.']*(?:\*\d+)?\b")


def _find_brace_block(text: str, start: int) -> tuple[int, int] | None:
    brace_start = text.find("{", start)
    if brace_start == -1:
        return None
    brace_end = grab_balanced(text, brace_start)
    if brace_end == -1:
        return None
    return brace_start, brace_end


def _extract_command(text: str, idx: int) -> tuple[str, int] | None:
    """Extract a structural command and the next cursor position."""
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
    """Return the first occurrences of key/time/tempo/partial commands."""
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
    """Pick the most relevant block to extract structure from."""
    for match in RE_ASSIGN_NAME.finditer(text):
        if match.group(1).lower() != "forma":
            continue
        bounds = _find_brace_block(text, match.end() - 1)
        if not bounds:
            continue
        brace_start, brace_end = bounds
        return text[brace_start + 1 : brace_end]

    for match in RE_ASSIGN_NAME.finditer(text):
        if not match.group(1).lower().endswith("global"):
            continue
        bounds = _find_brace_block(text, match.end() - 1)
        if not bounds:
            continue
        brace_start, brace_end = bounds
        block = text[brace_start + 1 : brace_end]
        if RE_COMMAND.search(block):
            return block

    return text


def _should_prepend(block: str) -> bool:
    """Return True when a block starts with notes before structure."""
    note_match = RE_NOTE.search(block)
    if not note_match:
        return False
    cmd_match = RE_COMMAND.search(block)
    if cmd_match and cmd_match.start() < note_match.start():
        return False
    return True


def prepend_structure(text: str, _opts) -> str:
    """Prepend structure tokens into assignment blocks when needed."""
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
        bounds = _find_brace_block(text, match.end() - 1)
        if not bounds:
            continue
        brace_start, brace_end = bounds
        out.append(text[cursor : brace_start + 1])
        block = text[brace_start + 1 : brace_end]
        if _should_prepend(block):
            prefix = "\n" if block.startswith("\n") else " "
            block = f"{prefix}{structure}\n{block.lstrip()}"
        out.append(block)
        cursor = brace_end
    out.append(text[cursor:])
    return "".join(out)


def _extract_forma(text: str) -> tuple[str | None, tuple[int, int] | None]:
    """Return the forma body and its span in the source text."""
    match = FORMA_START.search(text)
    if not match:
        return None, None
    brace_start = match.end() - 1
    brace_end = grab_balanced(text, brace_start)
    if brace_end == -1:
        return None, None
    return text[brace_start + 1:brace_end], (match.start(), brace_end + 1)


def _strip_skips_and_layout(forma_body: str) -> str:
    """Keep only semantic commands, stripping skips and layout markers."""
    parts: list[str] = []
    tokens = forma_body.splitlines()
    for line in tokens:
        line = line.split("%", 1)[0]
        if not line.strip():
            continue
        if not SEMANTIC_CMD_RE.search(line):
            continue
        line = SKIP_RE.sub("", line)
        line = line.replace("\\break", "").replace("\\pageBreak", "")
        parts.append(line.strip())
    return " ".join(parts).strip()


def _split_simul_block(block: str) -> list[str]:
    """Split a simultaneous block into top-level chunks."""
    chunks: list[str] = []
    i = 0
    n = len(block)
    start = 0
    depth_brace = 0
    depth_angle = 0
    while i < n:
        if block.startswith("<<", i):
            depth_angle += 1
            i += 2
            continue
        if block.startswith(">>", i):
            depth_angle = max(0, depth_angle - 1)
            i += 2
            continue
        if block[i] == "{":
            depth_brace += 1
        elif block[i] == "}":
            depth_brace = max(0, depth_brace - 1)
        elif block[i].isspace() and depth_brace == 0 and depth_angle == 0:
            chunk = block[start:i].strip()
            if chunk:
                chunks.append(chunk)
            while i < n and block[i].isspace():
                i += 1
            start = i
            continue
        i += 1
    tail = block[start:].strip()
    if tail:
        chunks.append(tail)
    return chunks


def _inline_forma_in_simul(text: str, forma_semantics: str) -> str:
    """Inline forma semantics into simultaneous voice blocks."""
    if not forma_semantics:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("<<", i):
            end = grab_angles(text, i)
            if end == -1:
                out.append(text[i:])
                break
            block = text[i + 2:end - 2].strip()
            if "\\forma" not in block:
                out.append(text[i:end])
                i = end
                continue
            pieces = _split_simul_block(block)
            pieces = [p for p in pieces if p != "\\forma"]
            if len(pieces) == 1:
                new_block = "{ " + forma_semantics + " " + pieces[0] + " }"
            else:
                new_block = "{ " + forma_semantics + " << " + " ".join(pieces) + " >> }"
            out.append(new_block)
            i = end
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _remove_forma_assignment(text: str, forma_span: tuple[int, int] | None) -> str:
    """Remove the original forma assignment after inlining."""
    if not forma_span:
        return text
    start, end = forma_span
    line_start = text.rfind("\n", 0, start) + 1
    line_end = end
    if line_end < len(text) and text[line_end] == "\n":
        line_end += 1
    return text[:line_start] + text[line_end:]


def inline_forma(text: str, _opts) -> str:
    """Inline forma semantics and drop the original forma block."""
    forma_body, forma_span = _extract_forma(text)
    if not forma_body:
        return text
    forma_semantics = _strip_skips_and_layout(forma_body)
    if not forma_semantics:
        return text
    text = _inline_forma_in_simul(text, forma_semantics)
    text = _remove_forma_assignment(text, forma_span)
    return text
