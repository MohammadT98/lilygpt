"""Preprocess LilyPond files by removing comments and normalizing whitespace."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Tuple

BOM = "\ufeff"
_RE_MARKUP_TOKEN = re.compile(r"\\markup(?![A-Za-z])")

@dataclass
class CleanOptions:
    """Options controlling preprocessing cleanup behavior."""
    keep_line_comments: bool = False
    keep_block_comments: bool = False
    max_blank_lines: int = 1
    trim_trailing_ws: bool = True
    ensure_final_newline: bool = True

@dataclass
class CleanStats:
    """Counters for removed comment types."""
    line_comments_removed: int = 0
    block_comments_removed: int = 0

class Context:
    __slots__ = ("in_string", "escaped", "scheme_paren_depth", "markup_brace_depth", "saw_markup_token")

    def __init__(self) -> None:
        self.in_string = False
        self.escaped = False
        self.scheme_paren_depth = 0
        self.markup_brace_depth = 0
        self.saw_markup_token = False

def _normalize_newlines(text: str) -> str:
    return text.lstrip(BOM).replace("\r\n", "\n").replace("\r", "\n")


def _split_inline_assignments(text: str) -> str:
    text = re.sub(r"}\s+(?=[A-Za-z_][\w-]*\s*=)", "}\n\n", text)
    text = re.sub(r"\)\s+(?=[A-Za-z_][\w-]*\s*=)", ")\n\n", text)
    return text

def _remove_comments(text: str, opts: CleanOptions, stats: CleanStats) -> str:
    ctx = Context()
    out_chars = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # Handle string literals
        if ctx.in_string:
            out_chars.append(ch)
            if ctx.escaped:
                ctx.escaped = False
            elif ch == "\\":
                ctx.escaped = True
            elif ch == '"':
                ctx.in_string = False
            i += 1
            continue

        # Enter string (outside Scheme and markup)
        if ch == '"' and ctx.scheme_paren_depth == 0 and ctx.markup_brace_depth == 0:
            ctx.in_string = True
            out_chars.append(ch)
            i += 1
            continue

        # Handle Scheme blocks
        if ctx.scheme_paren_depth > 0:
            out_chars.append(ch)
            ctx.scheme_paren_depth += (ch == "(") - (ch == ")")
            i += 1
            continue

        # Enter Scheme block
        if text.startswith("#(", i):
            out_chars.append("#(")
            i += 2
            ctx.scheme_paren_depth = 1
            continue

        # Handle markup blocks
        if ctx.markup_brace_depth > 0:
            out_chars.append(ch)
            ctx.markup_brace_depth += (ch == "{") - (ch == "}")
            i += 1
            continue

        # After seeing \markup token
        if ctx.saw_markup_token:
            out_chars.append(ch)
            if ch == "{":
                ctx.markup_brace_depth = 1
                ctx.saw_markup_token = False
            elif not ch.isspace():
                ctx.saw_markup_token = False
            i += 1
            continue

        # Detect \markup token
        if ch == "\\":
            markup_match = _RE_MARKUP_TOKEN.match(text, i)
            if markup_match:
                out_chars.append(markup_match.group(0))
                i = markup_match.end()
                ctx.saw_markup_token = True
                continue

        # Handle comments
        if ch == "%":
            if text.startswith("%{", i) and not opts.keep_block_comments:
                i, removed = _consume_block_comment(text, i)
                stats.block_comments_removed += removed
                continue
            if not opts.keep_line_comments:
                while i < n and text[i] != "\n":
                    i += 1
                stats.line_comments_removed += 1
                continue

        out_chars.append(ch)
        i += 1

    return "".join(out_chars)

def _consume_block_comment(text: str, i: int) -> Tuple[int, int]:
    depth = 0
    idx = i
    n = len(text)

    while idx < n:
        if text.startswith("%{", idx):
            depth += 1
            idx += 2
        elif text.startswith("%}", idx):
            depth -= 1
            idx += 2
            if depth == 0:
                return idx, 1
        else:
            idx += 1

    print("unterminated block comment", file=sys.stderr)
    return idx, 1

def _post_whitespace_cleanup(text: str, opts: CleanOptions) -> str:
    lines = text.split("\n")

    if opts.trim_trailing_ws:
        lines = [line.rstrip(" \t\x0b\x0c") for line in lines]

    if opts.max_blank_lines >= 0:
        collapsed = []
        blank_streak = 0
        for line in lines:
            if not line.strip():
                blank_streak += 1
                if blank_streak <= opts.max_blank_lines:
                    collapsed.append("")
            else:
                blank_streak = 0
                collapsed.append(line)
        lines = collapsed

    output = "\n".join(lines)
    if opts.ensure_final_newline and output and not output.endswith("\n"):
        output += "\n"

    return output

def clean_text(
    src: str,
    opts: CleanOptions | None = None,
    stats: CleanStats | None = None,
) -> Tuple[str, CleanStats]:
    """Return cleaned text and comment-removal statistics."""
    opts = opts or CleanOptions()
    stats = stats or CleanStats()
    text = _normalize_newlines(src)
    text = _remove_comments(text, opts, stats)
    text = _post_whitespace_cleanup(text, opts)
    text = _split_inline_assignments(text)
    return text, stats

try:
    from lilynorm.utils.options import NormOptions
except Exception:
    class NormOptions:
        strip_comments: bool = True
        normalize_whitespace: bool = True

def run(text: str, opts: "NormOptions") -> str:
    """Preprocess text using options compatible with NormOptions."""
    clean_opts = CleanOptions(
        keep_line_comments=not getattr(opts, "strip_comments", True),
        keep_block_comments=not getattr(opts, "strip_comments", True),
        trim_trailing_ws=getattr(opts, "normalize_whitespace", True),
    )
    cleaned, stats = clean_text(text, clean_opts)
    print(f"[preprocess] line_removed={stats.line_comments_removed} block_removed={stats.block_comments_removed}", file=sys.stderr)
    return cleaned
