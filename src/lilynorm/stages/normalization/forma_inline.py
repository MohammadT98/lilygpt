from __future__ import annotations

import re

from .utils import grab_balanced, grab_angles

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


def _extract_forma(text: str) -> tuple[str | None, tuple[int, int] | None]:
    match = FORMA_START.search(text)
    if not match:
        return None, None
    brace_start = match.end() - 1
    brace_end = grab_balanced(text, brace_start)
    if brace_end == -1:
        return None, None
    return text[brace_start + 1:brace_end], (match.start(), brace_end + 1)


def _strip_skips_and_layout(forma_body: str) -> str:
    # Keep only semantic commands; drop skips and layout tokens.
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
    # Split the inside of << >> into top-level chunks.
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
    if not forma_span:
        return text
    start, end = forma_span
    line_start = text.rfind("\n", 0, start) + 1
    line_end = end
    if line_end < len(text) and text[line_end] == "\n":
        line_end += 1
    return text[:line_start] + text[line_end:]


def run(text: str, _opts) -> str:
    forma_body, forma_span = _extract_forma(text)
    if not forma_body:
        return text
    forma_semantics = _strip_skips_and_layout(forma_body)
    if not forma_semantics:
        return text
    text = _inline_forma_in_simul(text, forma_semantics)
    text = _remove_forma_assignment(text, forma_span)
    return text
