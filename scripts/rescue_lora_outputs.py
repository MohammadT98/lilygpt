"""Aggressive rescue of LoRA-generated LilyPond samples.

LoRA models trained on full-file train.jsonl reproduce the corpus shape
(`name = \\relative { ... }` top-level assignments) without the surrounding
render structure (`\\score { ... \\midi {} }`). They also reference Mutopia/
bmdataset macros like `\\mbreak` that the training files defined inline but
the LoRA forgets to emit. Combined with token-budget truncation, the raw
output has a 0% compile rate.

This script does a structural rebuild:

  1. Parse top-level `name [\\prefix] { ... }` assignments via a depth-aware
     scanner that ignores strings and comments.
  2. Drop the trailing assignment if its bar count is < 50% of the average
     of the others (iteratively until stable, or only one assignment left).
  3. If a single surviving assignment has < 8 bars: drop the whole sample
     (return None — caller writes nothing).
  4. For each kept assignment: cut the body at the last `|` (whole bars
     only), then append `>>` / `}` to balance any open brackets.
  5. Reconstruct the file as:
        \\version "..."
        \\language "..."
        <compat preamble for any \\mbreak / \\noBreak / etc. that appears>
        <header block kept verbatim if present>
        \\score {
          \\new Staff { <body> }                      # single
          # or
          \\new StaffGroup << \\new Staff { b1 } ... >>  # multi
          \\layout {}
          \\midi {}
        }

The original assignment definitions are stripped from the rebuilt file —
the score block consumes the bodies inline. This avoids name-resolution
errors when the model references undefined assignments.

Usage:
    python scripts/rescue_lora_outputs.py \\
        --input-dir /nfsd/.../inference/phi4_lora/samples \\
        --output-dir /nfsd/.../inference/phi4_lora_rescued_v2/samples
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


# Custom commands the bmdataset/Mutopia training corpus defines inline but
# the LoRA forgets to emit. We add them to the rebuilt preamble only when
# they actually appear in the kept body, so the preamble stays minimal.
_COMPAT_MACROS: dict[str, str] = {
    "mbreak": r"{ \break }",
    "noBreak": r"{ \noBreak }",
}

_VERSION_RE = re.compile(r'\\version\s+"([^"]+)"')
_LANGUAGE_RE = re.compile(r'\\language\s+"([^"]+)"')
_HEADER_RE = re.compile(r'\\header\s*\{', re.MULTILINE)


@dataclass
class _Assignment:
    name: str
    body: str          # raw text between the outermost {}
    bars: int          # count of `|` in body, outside strings/comments
    body_start: int    # offset of the `{` in the original text (for debugging)


def _next_nonspace(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in " \t\r\n":
        i += 1
    return i


def _scan_advance(text: str, i: int, n: int) -> tuple[int, str]:
    """Single-step scanner that skips strings/comments transparently.

    Returns (new_i, mode) where mode in {'code', 'eof'}. When mode='code',
    text[new_i] is a non-string non-comment character. The caller then
    inspects/advances past it.
    """
    while i < n:
        c = text[i]
        if c == "%":
            if i + 1 < n and text[i + 1] == "{":
                # block comment
                j = text.find("%}", i + 2)
                if j == -1:
                    return n, "eof"
                i = j + 2
                continue
            # line comment
            j = text.find("\n", i + 1)
            i = n if j == -1 else j + 1
            continue
        if c == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
            i = j + 1 if j < n else n
            continue
        return i, "code"
    return n, "eof"


def _count_bars(body: str) -> int:
    """Count `|` characters in body, outside strings and comments."""
    count = 0
    i = 0
    n = len(body)
    while i < n:
        i, mode = _scan_advance(body, i, n)
        if mode == "eof":
            break
        c = body[i]
        if c == "|":
            count += 1
        i += 1
    return count


def _truncate_at_last_bar(body: str) -> str:
    """Cut after the last `|` that's outside strings and comments. If no `|`,
    return the body as-is."""
    last = -1
    i = 0
    n = len(body)
    while i < n:
        i, mode = _scan_advance(body, i, n)
        if mode == "eof":
            break
        if body[i] == "|":
            last = i + 1
        i += 1
    if last == -1:
        return body
    return body[:last]


def _balance_brackets(body: str) -> str:
    """Append `>>` and `}` to close any open `<<` or `{` in body."""
    depth_curly = 0
    depth_angle = 0
    i = 0
    n = len(body)
    while i < n:
        i, mode = _scan_advance(body, i, n)
        if mode == "eof":
            break
        c = body[i]
        if c == "<" and i + 1 < n and body[i + 1] == "<":
            depth_angle += 1
            i += 2
            continue
        if c == ">" and i + 1 < n and body[i + 1] == ">":
            depth_angle = max(0, depth_angle - 1)
            i += 2
            continue
        if c == "{":
            depth_curly += 1
        elif c == "}":
            depth_curly = max(0, depth_curly - 1)
        i += 1
    suffix = ">>" * depth_angle + "}" * depth_curly
    return body + ("\n" + suffix if suffix else "")


_NAME_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*)\s*=\s*", re.ASCII)


def _find_assignments(text: str) -> list[_Assignment]:
    """Walk text at depth 0 and capture top-level `name = [...] { ... }` blocks.

    Returns assignments in order of appearance. Bodies are the raw text between
    the matching outermost `{` and `}` (or end-of-file if unclosed).
    """
    assignments: list[_Assignment] = []
    n = len(text)
    i = 0
    depth = 0
    while i < n:
        i, mode = _scan_advance(text, i, n)
        if mode == "eof":
            break
        c = text[i]
        if c == "{":
            depth += 1
            i += 1
            continue
        if c == "}":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and (c.isalpha() or c == "_"):
            m = _NAME_RE.match(text, i)
            if m:
                # We must be at the start of an identifier — i.e., the
                # previous non-space char must have been a `}`, `>>`,
                # newline-after-end-of-block, or BOF.
                prev = i - 1
                while prev >= 0 and text[prev] in " \t\r\n":
                    prev -= 1
                if prev < 0 or text[prev] in "}\n>":
                    # Find the next `{` after the `=` (skipping any
                    # `\command args`).
                    j = m.end()
                    while j < n:
                        j, jmode = _scan_advance(text, j, n)
                        if jmode == "eof":
                            break
                        if text[j] == "{":
                            break
                        # Not yet at body — skip a token (e.g., `\relative`,
                        # `c''`, etc.) up to next whitespace.
                        while j < n and text[j] not in " \t\r\n{":
                            j += 1
                    if j < n and text[j] == "{":
                        body_start = j + 1
                        # Find matching `}` honoring nesting (and skipping
                        # strings/comments).
                        bd = 1
                        k = body_start
                        while k < n:
                            k, kmode = _scan_advance(text, k, n)
                            if kmode == "eof":
                                break
                            ch = text[k]
                            if ch == "{":
                                bd += 1
                            elif ch == "}":
                                bd -= 1
                                if bd == 0:
                                    break
                            k += 1
                        body_end = k if k < n else n
                        body = text[body_start:body_end]
                        assignments.append(
                            _Assignment(
                                name=m.group(1),
                                body=body,
                                bars=_count_bars(body),
                                body_start=body_start,
                            )
                        )
                        i = body_end + 1 if body_end < n else n
                        continue
        i += 1
    return assignments


def _drop_trailing_short(
    assignments: list[_Assignment], ratio: float
) -> list[_Assignment]:
    """Iteratively drop the trailing assignment if its bar count is below
    `ratio` * mean(bars of others). Stops when only one assignment remains or
    the trailing is no longer too short."""
    kept = list(assignments)
    while len(kept) >= 2:
        others = kept[:-1]
        avg = sum(a.bars for a in others) / len(others)
        if avg <= 0:
            break
        if kept[-1].bars < ratio * avg:
            kept.pop()
        else:
            break
    return kept


def _detect_compat(body: str) -> list[str]:
    """Return a list of macro names from _COMPAT_MACROS that appear in body
    (as `\\name` references)."""
    used: list[str] = []
    for name in _COMPAT_MACROS:
        if re.search(r"\\" + re.escape(name) + r"(?![A-Za-z0-9])", body):
            used.append(name)
    return used


def _extract_header_block(text: str) -> str | None:
    """Extract the first \\header { ... } block at top-level if present."""
    m = _HEADER_RE.search(text)
    if not m:
        return None
    start = m.end()
    n = len(text)
    depth = 1
    i = start
    while i < n:
        i, mode = _scan_advance(text, i, n)
        if mode == "eof":
            break
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[m.start() : i + 1]
        i += 1
    return None


def rescue_text(
    text: str,
    min_bars: int = 8,
    ratio_threshold: float = 0.5,
) -> tuple[str | None, str]:
    """Return (rescued_text, status).

    Status is one of: 'rescued', 'unchanged', 'dropped_short'.

    'rescued' = file rewritten with new \\score block.
    'unchanged' = file already had a balanced \\score block; left as-is.
    'dropped_short' = surviving music below threshold; caller should skip.
    """
    # Quick-out: if text already has a balanced `\score` block with `\midi`,
    # don't touch it.
    if "\\score" in text and "\\midi" in text:
        # cheap balance check
        depth_curly = 0
        i = 0
        n = len(text)
        while i < n:
            i, mode = _scan_advance(text, i, n)
            if mode == "eof":
                break
            c = text[i]
            if c == "{":
                depth_curly += 1
            elif c == "}":
                depth_curly -= 1
            i += 1
        if depth_curly == 0:
            return text, "unchanged"

    assignments = _find_assignments(text)
    # No assignments found — try treating the whole non-prelude tail as a
    # single anonymous body. Heuristic: take everything after the last
    # \version / \language line.
    if not assignments:
        anchor = 0
        for rx in (_VERSION_RE, _LANGUAGE_RE):
            m = rx.search(text)
            if m:
                anchor = max(anchor, m.end())
        body = text[anchor:].lstrip()
        if body:
            assignments = [_Assignment(name="anon", body=body, bars=_count_bars(body), body_start=anchor)]

    if not assignments:
        return None, "dropped_short"

    kept = _drop_trailing_short(assignments, ratio_threshold)
    if len(kept) == 1 and kept[0].bars < min_bars:
        return None, "dropped_short"

    # Cut + balance each kept body.
    rebuilt_bodies: list[str] = []
    for a in kept:
        cut = _truncate_at_last_bar(a.body)
        balanced = _balance_brackets(cut)
        rebuilt_bodies.append(balanced.strip())

    # Detect compat macros across all kept bodies.
    used_compat: list[str] = []
    seen = set()
    for body in rebuilt_bodies:
        for name in _detect_compat(body):
            if name not in seen:
                seen.add(name)
                used_compat.append(name)

    # Pick version + language from the original (defaults if missing).
    m_ver = _VERSION_RE.search(text)
    version_line = f'\\version "{m_ver.group(1)}"' if m_ver else '\\version "2.24.4"'
    m_lang = _LANGUAGE_RE.search(text)
    language_line = f'\\language "{m_lang.group(1)}"' if m_lang else '\\language "nederlands"'

    header_block = _extract_header_block(text)

    parts: list[str] = [version_line, language_line, ""]
    if used_compat:
        for name in used_compat:
            parts.append(f"{name} = {_COMPAT_MACROS[name]}")
        parts.append("")
    if header_block:
        parts.append(header_block)
        parts.append("")

    if len(rebuilt_bodies) == 1:
        score = (
            "\\score {\n"
            f"  \\new Staff {{\n{rebuilt_bodies[0]}\n}}\n"
            "  \\layout {}\n"
            "  \\midi {}\n"
            "}\n"
        )
    else:
        staves = "\n".join(
            f"    \\new Staff {{\n{b}\n}}" for b in rebuilt_bodies
        )
        score = (
            "\\score {\n"
            f"  \\new StaffGroup <<\n{staves}\n  >>\n"
            "  \\layout {}\n"
            "  \\midi {}\n"
            "}\n"
        )
    parts.append(score)

    return "\n".join(parts), "rescued"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-bars", type=int, default=8)
    p.add_argument("--ratio-threshold", type=float, default=0.5)
    p.add_argument("--stats-out", type=Path, default=None)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.ly"))
    if not files:
        print(f"[rescue] no .ly files in {args.input_dir}")
        return 2

    counts = {"unchanged": 0, "rescued": 0, "dropped_short": 0}
    for src in files:
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            counts["dropped_short"] += 1
            continue
        rescued, status = rescue_text(
            text, min_bars=args.min_bars, ratio_threshold=args.ratio_threshold
        )
        counts[status] = counts.get(status, 0) + 1
        if status == "dropped_short" or rescued is None:
            continue
        out = args.output_dir / src.name
        out.write_text(rescued, encoding="utf-8")

    total = sum(counts.values())
    written = counts["unchanged"] + counts["rescued"]
    print(
        f"[rescue] {total} files: "
        f"unchanged={counts['unchanged']}, "
        f"rescued={counts['rescued']}, "
        f"dropped_short={counts['dropped_short']} "
        f"(wrote {written} to {args.output_dir})"
    )

    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(
            json.dumps({"counts": counts, "total": total, "written": written}, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
