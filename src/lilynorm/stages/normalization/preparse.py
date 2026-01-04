from __future__ import annotations

import io
import re
import sys
from dataclasses import dataclass
from typing import Tuple

BOM = "\ufeff"

_RE_MARKUP_TOKEN = re.compile(r"\\markup(?![A-Za-z])")


@dataclass
class CleanOptions:
    """
    Options controlling comment and whitespace handling during cleaning.
    """
    keep_line_comments: bool = False
    keep_block_comments: bool = False
    max_blank_lines: int = 1
    trim_trailing_ws: bool = True
    ensure_final_newline: bool = True


@dataclass
class CleanStats:
    """
    Statistics about removed comments during cleaning.
    """
    line_comments_removed: int = 0
    block_comments_removed: int = 0


class Context:
    """
    Stateful context for scanning LilyPond-like text.

    Tracks whether we are:
      - inside a string
      - inside a Scheme expression (#( ... ))
      - inside a markup { ... } block
      - right after a \\markup token (waiting to see if a brace follows)
    """
    __slots__ = (
        "in_string",
        "escaped",
        "scheme_paren_depth",
        "markup_brace_depth",
        "saw_markup_token",
    )

    def __init__(self) -> None:
        self.in_string: bool = False
        self.escaped: bool = False
        self.scheme_paren_depth: int = 0
        self.markup_brace_depth: int = 0
        self.saw_markup_token: bool = False  # True just after seeing \markup

    def in_opaque(self) -> bool:
        """
        Return True if we are inside a context where comments should not be parsed
        (string, Scheme expression, or markup block).
        """
        return (
            self.in_string
            or self.scheme_paren_depth > 0
            or self.markup_brace_depth > 0
        )


def _normalize_newlines(text: str) -> str:
    if text.startswith(BOM):
        text = text[len(BOM):]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _remove_comments(text: str, opts: CleanOptions, stats: CleanStats) -> str:
    r"""
    Remove line and block comments from text according to CleanOptions.

    The scanner is aware of:
      - double-quoted strings
      - Scheme blocks: #( ... )
      - markup blocks: \markup { ... }

    Comments are:
      - Line comments: % ... (until newline)
      - Block comments: %{ ... %}
    """
    ctx = Context()
    out_chars: list[str] = []
    i = 0
    n = len(text)

    def peek(offset: int = 0) -> str:
        index = i + offset
        return text[index] if 0 <= index < n else ""

    def startswith_at(prefix: str) -> bool:
        return text.startswith(prefix, i)

    while i < n:
        ch = text[i]

        # --------------------------------------------------
        # Inside string literal
        # --------------------------------------------------
        if ctx.in_string:
            out_chars.append(ch)

            if ctx.escaped:
                ctx.escaped = False
            else:
                if ch == "\\":
                    ctx.escaped = True
                elif ch == '"':
                    ctx.in_string = False

            i += 1
            continue

        # --------------------------------------------------
        # Starting a string (only when not in Scheme/markup)
        # --------------------------------------------------
        if (
            ch == '"'
            and ctx.scheme_paren_depth == 0
            and ctx.markup_brace_depth == 0
        ):
            ctx.in_string = True
            out_chars.append(ch)
            i += 1
            continue

        # --------------------------------------------------
        # Inside Scheme expression: #( ... )
        # --------------------------------------------------
        if ctx.scheme_paren_depth > 0:
            out_chars.append(ch)
            if ch == "(":
                ctx.scheme_paren_depth += 1
            elif ch == ")":
                ctx.scheme_paren_depth -= 1
            i += 1
            continue

        # Scheme start: "#("
        if startswith_at("#("):
            out_chars.append("#(")
            i += 2
            ctx.scheme_paren_depth = 1
            continue

        # --------------------------------------------------
        # Inside markup block: \markup { ... }
        # --------------------------------------------------
        if ctx.markup_brace_depth > 0:
            out_chars.append(ch)
            if ch == "{":
                ctx.markup_brace_depth += 1
            elif ch == "}":
                ctx.markup_brace_depth -= 1
            i += 1
            continue

        # Just saw \markup token and now we're looking for its brace
        if ctx.saw_markup_token:
            if ch.isspace():
                out_chars.append(ch)
                i += 1
                continue
            if ch == "{":
                out_chars.append(ch)
                ctx.markup_brace_depth = 1
                ctx.saw_markup_token = False
                i += 1
                continue

            # Any other non-space means no markup block follows
            ctx.saw_markup_token = False

        # --------------------------------------------------
        # Backslash: may be \markup or other command
        # --------------------------------------------------
        if ch == "\\":
            markup_match = _RE_MARKUP_TOKEN.match(text, i)
            if markup_match:
                out_chars.append(markup_match.group(0))
                i = markup_match.end()
                ctx.saw_markup_token = True
                continue

            out_chars.append(ch)
            i += 1
            continue

        # --------------------------------------------------
        # Comments: %, %{ ... %}, and line comments
        # --------------------------------------------------
        if ch == "%":
            # Block comment %{ ... %}
            if text.startswith("%{", i) and not opts.keep_block_comments:
                i, removed_blocks = _consume_block_comment(text, i)
                stats.block_comments_removed += removed_blocks
                continue

            # Single-line comment % ... \n
            if not opts.keep_line_comments:
                i += 1
                while i < n and text[i] != "\n":
                    i += 1
                stats.line_comments_removed += 1
                continue

        # --------------------------------------------------
        # Normal character
        # --------------------------------------------------
        out_chars.append(ch)
        i += 1

    return "".join(out_chars)


def _consume_block_comment(text: str, i: int) -> Tuple[int, int]:
    assert text.startswith("%{", i)
    n = len(text)
    depth = 0
    idx = i
    removed = 0

    while idx < n:
        if text.startswith("%{", idx):
            depth += 1
            idx += 2
            continue

        if text.startswith("%}", idx):
            depth -= 1
            idx += 2
            if depth == 0:
                removed += 1
                break
            continue

        idx += 1

    if depth > 0:
        print("unterminated block comment", file=sys.stderr)
        removed += 1
    return idx, removed


def _post_whitespace_cleanup(text: str, opts: CleanOptions) -> str:
    lines = text.split("\n")

    # Trim trailing whitespace
    if opts.trim_trailing_ws:
        lines = [line.rstrip(" \t\x0b\x0c") for line in lines]

    # Collapse blank lines
    if opts.max_blank_lines >= 0:
        collapsed: list[str] = []
        blank_streak = 0

        for line in lines:
            if line.strip() == "":
                blank_streak += 1
                if blank_streak <= opts.max_blank_lines:
                    collapsed.append("")
            else:
                blank_streak = 0
                collapsed.append(line)

        lines = collapsed

    output = "\n".join(lines)
    if opts.ensure_final_newline and not output.endswith("\n"):
        output += "\n"

    return output


def clean_text(
    src: str,
    opts: CleanOptions | None = None,
    stats: CleanStats | None = None,
) -> Tuple[str, CleanStats]:
    """
    Clean comments and whitespace from src according to CleanOptions.

    Returns (cleaned_text, CleanStats).
    """
    opts = opts or CleanOptions()
    stats = stats or CleanStats()

    text = _normalize_newlines(src)
    text = _remove_comments(text, opts, stats)
    text = _post_whitespace_cleanup(text, opts)

    return text, stats


try:
    from lilynorm.utils.options import NormOptions
except Exception:
    class NormOptions:  # type: ignore[override]
        """
        Fallback NormOptions when lilynorm.utils.options is unavailable.

        Only mirrors the attributes used by this module.
        """
        keep_engraving: bool
        strip_scheme_blocks: bool
        strip_comments: bool
        normalize_whitespace: bool
        expand_relative: bool


def run(text: str, opts: "NormOptions") -> str:
    """
    Entry point used by the lilynorm pipeline.

    Builds CleanOptions from NormOptions, runs clean_text, prints a brief
    report to stdout, and returns the cleaned text.
    """
    clean_opts = CleanOptions(
        keep_line_comments=not getattr(opts, "strip_comments", True),
        keep_block_comments=not getattr(opts, "strip_comments", True),
        max_blank_lines=1,
        trim_trailing_ws=getattr(opts, "normalize_whitespace", True),
        ensure_final_newline=True,
    )

    cleaned, stats = clean_text(text, clean_opts)

    print(
        f"[preparse] line_removed={stats.line_comments_removed} "
        f"block_removed={stats.block_comments_removed}",
        file=sys.stderr,
    )
    return cleaned
