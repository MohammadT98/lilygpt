from __future__ import annotations

"""Prelude boundary detection and variable-declaration extraction.

The prelude is the leading region of a preprocessed bmdataset LilyPond file that
contains boilerplate (``\\version``, ``\\header``, ``\\paper``, ``\\language``) and
the ``variabili.ly`` articulation library. Anti-memorization augmentations target
declarations in this region only; the music body that follows is left untouched.
"""

import re
from dataclasses import dataclass, field
from typing import Iterable

_VARIABILI_END_MARKER = "% === END INCLUDE: variabili.ly ==="

_ASSIGNMENT_START = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9]*)\s*=", re.MULTILINE)

_REL_REFERENCE = re.compile(r"\\relative\b")
_PITCH_WITH_DURATION = re.compile(
    r"\b[a-g](?:is|es|isis|eses)?[,']*(?:\d+)",
)
_CHORD_OPEN = re.compile(r"<[a-g]")
_REST_WITH_DURATION = re.compile(r"\b[Rr]\d+\b")
_VAR_REFERENCE = re.compile(r"\\([A-Za-z][A-Za-z0-9]*)")


@dataclass(frozen=True)
class VarDecl:
    """A single top-level ``name = <rhs>`` declaration within the prelude."""

    name: str
    full_start: int
    full_end: int
    body_start: int
    body_end: int
    body_text: str
    has_pitch_tokens: bool
    line_count: int
    references: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_short(self) -> bool:
        return self.line_count <= 10 and not self.has_pitch_tokens


@dataclass(frozen=True)
class PreludeInfo:
    """Result of :func:`identify_prelude`."""

    prelude_end: int
    body_start: int
    vars: tuple[VarDecl, ...]

    @property
    def char_range(self) -> tuple[int, int]:
        return (0, self.prelude_end)


def _body_has_pitch_tokens(body: str) -> bool:
    if _REL_REFERENCE.search(body):
        return True
    if _CHORD_OPEN.search(body):
        return True
    if _PITCH_WITH_DURATION.search(body):
        return True
    if _REST_WITH_DURATION.search(body):
        return True
    return False


def _scan_brace_body(text: str, start: int) -> int:
    """Given ``text[start] == '{'``, return the index just past the matching '}'."""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _scan_paren_body(text: str, start: int) -> int:
    """Given ``text[start] == '('``, return the index just past the matching ')'."""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _scan_simple_rhs(text: str, start: int) -> int:
    """RHS that isn't a top-level braced/parenthesized block: consume a logical line.

    Tracks ``{`` / ``(`` depth and ``"…"`` strings. A newline ends the RHS only
    when depth is zero and we are not inside a string. This correctly handles
    bodies like ``\\markup {\n  italic "hi"\n}`` and ``#(lambda (x) (+ x 1))``.
    """
    i = start
    n = len(text)
    brace = 0
    paren = 0
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            brace += 1
        elif c == "}":
            brace -= 1
        elif c == "(":
            paren += 1
        elif c == ")":
            paren -= 1
        elif c == "\n" and brace == 0 and paren == 0:
            return i
        i += 1
    return i


def _find_assignment_end(text: str, eq_end: int) -> int:
    """Return the char offset just past the last char of the RHS for an assignment.

    ``eq_end`` is the offset just after the ``=`` sign (whitespace, including
    newlines, may follow — LilyPond allows ``name =\n{ … }``).
    """
    i = eq_end
    n = len(text)
    while i < n and text[i] in " \t\r\n":
        i += 1
    if i >= n:
        return i
    c = text[i]
    if c == "{":
        end = _scan_brace_body(text, i)
        return end if end != -1 else n
    if c == "#" and i + 1 < n and text[i + 1] == "(":
        end = _scan_paren_body(text, i + 1)
        return end if end != -1 else n
    return _scan_simple_rhs(text, i)


def _extract_references(body: str) -> frozenset[str]:
    return frozenset(m.group(1) for m in _VAR_REFERENCE.finditer(body))


def iter_toplevel_assignments(
    text: str, start: int = 0, end: int | None = None
) -> Iterable[tuple[int, int]]:
    """Yield ``(full_start, full_end)`` spans for top-level ``name = …`` assignments.

    ``_ASSIGNMENT_START`` is anchored at line starts (``^`` in MULTILINE mode), so
    only declarations beginning in column 0 match — the same rule ``_iter_prelude_vars``
    applies. Use this to locate chunk boundaries in the music body after prelude
    detection.
    """
    if end is None:
        end = len(text)
    for match in _ASSIGNMENT_START.finditer(text, start, end):
        full_start = match.start()
        eq_pos = match.end()
        full_end = _find_assignment_end(text, eq_pos)
        if full_end <= eq_pos:
            continue
        yield (full_start, full_end)


def _iter_prelude_vars(text: str, prelude_end: int) -> Iterable[VarDecl]:
    for match in _ASSIGNMENT_START.finditer(text, 0, prelude_end):
        name = match.group("name")
        full_start = match.start()
        eq_pos = match.end()  # just past the '=' sign
        body_end = _find_assignment_end(text, eq_pos)
        if body_end <= eq_pos:
            continue
        # body_start is the first non-ws char after '=' (incl. newlines)
        body_start = eq_pos
        while body_start < body_end and text[body_start] in " \t\r\n":
            body_start += 1
        body_text = text[body_start:body_end]
        line_count = body_text.count("\n") + 1
        yield VarDecl(
            name=name,
            full_start=full_start,
            full_end=body_end,
            body_start=body_start,
            body_end=body_end,
            body_text=body_text,
            has_pitch_tokens=_body_has_pitch_tokens(body_text),
            line_count=line_count,
            references=_extract_references(body_text),
        )


def _find_music_body_start(text: str) -> int | None:
    """Fallback: first assignment whose body contains pitch tokens marks music-body start."""
    for match in _ASSIGNMENT_START.finditer(text):
        eq_pos = match.end()
        body_end = _find_assignment_end(text, eq_pos)
        if body_end <= eq_pos:
            continue
        body_start = eq_pos
        while body_start < body_end and text[body_start] in " \t\r\n":
            body_start += 1
        body = text[body_start:body_end]
        if _body_has_pitch_tokens(body):
            return match.start()
    return None


def _find_score_block_start(text: str) -> int | None:
    idx = text.find("\\score")
    return idx if idx != -1 else None


def identify_prelude(text: str) -> PreludeInfo:
    """Detect the prelude region in a preprocessed LilyPond file.

    Priority order:
    1. ``% === END INCLUDE: variabili.ly ===`` marker (present in most preprocessed
       files); prelude ends at the newline after that marker.
    2. First top-level assignment whose body contains pitch tokens.
    3. ``\\score { … }`` opening.
    4. Fallback: entire text is prelude (``prelude_end == len(text)``).
    """
    marker_idx = text.find(_VARIABILI_END_MARKER)
    if marker_idx != -1:
        newline_idx = text.find("\n", marker_idx)
        prelude_end = newline_idx + 1 if newline_idx != -1 else len(text)
    else:
        music_start = _find_music_body_start(text)
        if music_start is not None:
            prelude_end = music_start
        else:
            score_start = _find_score_block_start(text)
            prelude_end = score_start if score_start is not None else len(text)

    body_start = prelude_end
    # Skip the "BEGIN INCLUDE" marker line immediately after the prelude, if any.
    vars_tuple = tuple(_iter_prelude_vars(text, prelude_end))
    return PreludeInfo(prelude_end=prelude_end, body_start=body_start, vars=vars_tuple)


def find_version_language_header(text: str) -> str:
    r"""Extract the ``\version`` and ``\language`` lines for use in chunks #1..N.

    Returns a two-line string (or the lines we found, separated by ``\n``, with a
    trailing ``\n``). Missing lines are omitted silently.
    """
    lines: list[str] = []
    for key in (r"\version", r"\language"):
        m = re.search(rf"^{re.escape(key)}[^\n]*$", text, flags=re.MULTILINE)
        if m:
            lines.append(m.group(0))
    return "\n".join(lines) + ("\n" if lines else "")
