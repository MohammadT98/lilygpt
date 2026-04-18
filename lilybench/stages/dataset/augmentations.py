from __future__ import annotations

"""Syntax-preserving augmentations for anti-memorization training.

Each augmentation takes the original LilyPond text plus a :class:`PreludeInfo`
and returns a modified text. Augmentations target short, reference-free prelude
declarations only; music-body variables are never touched. Pass a pre-seeded
``random.Random`` for deterministic variant generation.
"""

import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .prelude import PreludeInfo, VarDecl

_CALL_SITE_CACHE: dict[str, re.Pattern] = {}


def _call_site_pattern(name: str) -> re.Pattern:
    if name not in _CALL_SITE_CACHE:
        _CALL_SITE_CACHE[name] = re.compile(rf"\\{re.escape(name)}\b")
    return _CALL_SITE_CACHE[name]


def _count_call_sites(text: str, name: str, decl: VarDecl) -> int:
    """Count ``\\name`` occurrences in ``text`` outside of ``decl``."""
    pattern = _call_site_pattern(name)
    count = 0
    for m in pattern.finditer(text):
        if decl.full_start <= m.start() < decl.full_end:
            continue
        count += 1
    return count


def _is_body_inlineable(decl: VarDecl) -> bool:
    """Return True if ``decl.body_text`` is a coherent single LilyPond expression.

    Accepts: brace-wrapped music, bare commands (``\\trill``), post-script markups
    (``_\\markup…``, ``^\\markup…``, ``-\\markup…``), and scheme blocks (``#(...)``).
    Rejects bodies that look like loose tokens (starting with a letter or digit)
    or that are empty. The final safety net is the ``check_brace_balance`` gate
    run on the full augmented output.
    """
    stripped = decl.body_text.strip()
    if not stripped:
        return False
    first = stripped[0]
    if first == "{":
        return stripped.rstrip().endswith("}")
    if first == "\\":
        return True
    if first in "_^-" and len(stripped) > 1 and stripped[1] == "\\":
        return True
    if first == "#" and len(stripped) > 1 and stripped[1] == "(":
        return True
    return False


def _shuffleable(decl: VarDecl, prelude_names: set[str]) -> bool:
    """A decl is shuffleable iff it is short and does not reference any other prelude var."""
    if not decl.is_short:
        return False
    cross_refs = decl.references & (prelude_names - {decl.name})
    return not cross_refs


def shuffle_prelude(
    text: str,
    info: PreludeInfo,
    rng: random.Random,
) -> str:
    """Permute the order of short, reference-free prelude declarations.

    Non-shuffleable declarations (long bodies, or those referencing other prelude
    vars) stay where they are; their text spans act as fixed anchors.
    """
    prelude_names = {v.name for v in info.vars}
    spans = [
        (v.full_start, v.full_end)
        for v in info.vars
        if _shuffleable(v, prelude_names)
    ]
    if len(spans) < 2:
        return text
    texts = [text[s:e] for s, e in spans]
    shuffled = texts.copy()
    rng.shuffle(shuffled)
    if shuffled == texts:
        return text

    pieces: list[str] = []
    cur = 0
    for (s, e), new_text in zip(spans, shuffled):
        pieces.append(text[cur:s])
        pieces.append(new_text)
        cur = e
    pieces.append(text[cur:])
    return "".join(pieces)


def drop_unused_prelude(
    text: str,
    info: PreludeInfo,
    rng: random.Random,
    p: float = 0.3,
) -> str:
    """Delete each short prelude declaration with zero call sites, with prob. ``p``.

    Never drops a declaration that is referenced anywhere in ``text`` (including
    from other prelude decls — we scan the whole file, not just the body region).
    """
    to_drop: list[VarDecl] = []
    for v in info.vars:
        if not v.is_short:
            continue
        if _count_call_sites(text, v.name, v) > 0:
            continue
        if rng.random() < p:
            to_drop.append(v)
    if not to_drop:
        return text
    to_drop.sort(key=lambda d: d.full_start)

    pieces: list[str] = []
    cur = 0
    for decl in to_drop:
        pieces.append(text[cur:decl.full_start])
        end = decl.full_end
        # Also consume the trailing newline(s) so we don't leave blank holes.
        while end < len(text) and text[end] in " \t":
            end += 1
        if end < len(text) and text[end] == "\n":
            end += 1
        cur = end
    pieces.append(text[cur:])
    return "".join(pieces)


def inline_variables(
    text: str,
    info: PreludeInfo,
    rng: random.Random,
    p: float = 0.5,
) -> str:
    """Inline each chosen prelude variable at all its call sites, then delete the decl.

    Safety: only inlines declarations whose body is a coherent single expression
    (see :func:`_is_body_inlineable`). Some call-site contexts may still produce
    invalid LilyPond (e.g., inlining a bare ``\\override`` command into a
    post-script position). The build pipeline's syntactic validator filters
    those variants out.
    """
    prelude_names = {v.name for v in info.vars}
    chosen: list[VarDecl] = []
    for v in info.vars:
        if not v.is_short:
            continue
        if not _is_body_inlineable(v):
            continue
        # Skip if the body references any prelude var — inlining could produce
        # unbound references if those vars were dropped by a sibling augmentation.
        if v.references & (prelude_names - {v.name}):
            continue
        if _count_call_sites(text, v.name, v) == 0:
            continue
        if rng.random() < p:
            chosen.append(v)
    if not chosen:
        return text

    # 1. Replace call sites. Do this from right to left in offset order so earlier
    #    offsets stay valid.
    replacements: list[tuple[int, int, str]] = []
    for decl in chosen:
        pattern = _call_site_pattern(decl.name)
        body = decl.body_text
        for m in pattern.finditer(text):
            if decl.full_start <= m.start() < decl.full_end:
                continue
            replacements.append((m.start(), m.end(), body))
    # 2. Mark declarations for deletion (plus trailing newline).
    for decl in chosen:
        end = decl.full_end
        while end < len(text) and text[end] in " \t":
            end += 1
        if end < len(text) and text[end] == "\n":
            end += 1
        replacements.append((decl.full_start, end, ""))

    replacements.sort(key=lambda r: r[0])
    pieces: list[str] = []
    cur = 0
    for start, end, replacement in replacements:
        if start < cur:
            # Overlapping replacement: skip (shouldn't happen in practice since
            # call sites are disjoint from the decl span).
            continue
        pieces.append(text[cur:start])
        pieces.append(replacement)
        cur = end
    pieces.append(text[cur:])
    return "".join(pieces)


def check_brace_balance(text: str) -> bool:
    """Fast syntactic sanity check: braces balanced and strings/block comments closed.

    Ignores characters inside ``%`` line comments, ``%{…}%`` block comments, and
    ``"…"`` strings. Parens are **not** checked: Scheme blocks (``#(...)``) embed
    quoted lists (``'(...)``), character literals (``#\\(``), and other constructs
    that a character-level scanner can't track without a full Scheme reader.
    Paren correctness is delegated to :func:`lilypond_parse_ok`.
    """
    n = len(text)
    i = 0
    brace = 0
    in_string = False
    in_line_comment = False
    while i < n:
        c = text[i]
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_string:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == "%":
            if i + 1 < n and text[i + 1] == "{":
                end = text.find("%}", i + 2)
                if end == -1:
                    return False
                i = end + 2
                continue
            in_line_comment = True
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "{":
            brace += 1
        elif c == "}":
            brace -= 1
            if brace < 0:
                return False
        i += 1
    return brace == 0 and not in_string


def lilypond_parse_ok(text: str, timeout: float = 10.0) -> bool:
    """Run ``lilypond`` on ``text`` and return True iff it exits cleanly.

    Heavy: invokes a subprocess and writes to a temp directory per call. Intended
    for sampled validation, not for every variant.
    """
    lily = shutil.which("lilypond")
    if lily is None:
        return False
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "variant.ly"
        src.write_text(text, encoding="utf-8")
        try:
            result = subprocess.run(
                [
                    lily,
                    "-dno-point-and-click",
                    "-loglevel=ERROR",
                    "-s",
                    "-o",
                    str(Path(td) / "out"),
                    str(src),
                ],
                capture_output=True,
                timeout=timeout,
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0
