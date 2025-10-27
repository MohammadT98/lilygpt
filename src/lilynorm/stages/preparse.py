#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
preparse_cleaner.py — LilyPond pre-parse lexical hygiene (safe regex/state-machine pass)

What it does (safely, without touching musical structure):
  • Normalize line endings to \n and strip BOM
  • Remove line comments starting with % (but NOT inside strings, Scheme #( ... ), or \markup { ... })
  • Remove block comments %{ ... %} with proper nesting (and only outside strings/Scheme/markup)
  • Trim trailing whitespace
  • Collapse multiple blank lines (configurable)
  • Ensure single trailing newline

What it does NOT do:
  • It never adds/removes braces {}, angle brackets <>, backslash commands, or anything structural
  • It treats strings, Scheme expressions, and markup blocks as opaque

Usage:
  python preparse_cleaner.py [-i INFILE] [-o OUTFILE] [--keep-line-comments] [--keep-block-comments]
                             [--max-blank-lines N] [--no-trim-trailing-ws] [--no-final-newline]
                             [--stats]

If -i/-o are omitted, reads stdin and writes stdout.

Notes:
  • This is a lexical pass: do your AST/structural work in a later phase.
  • Markup detection: we detect "\\markup" tokens and treat the following balanced { ... } as opaque.
  • Scheme detection: we detect "#(" and track paren balance until the matching ")".
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from dataclasses import dataclass
from typing import Tuple

BOM = "\ufeff"

# Simple token regexes
_RE_WS = re.compile(r"\s+")
_RE_MARKUP_TOKEN = re.compile(r"\\markup(?![A-Za-z])")

@dataclass
class CleanOptions:
    keep_line_comments: bool = False
    keep_block_comments: bool = False
    max_blank_lines: int = 1
    trim_trailing_ws: bool = True
    ensure_final_newline: bool = True
    # future: toggle removing editor modelines, duplicate \version, etc.

@dataclass
class CleanStats:
    line_comments_removed: int = 0
    block_comments_removed: int = 0

class Context:
    r"""Tracks whether we are inside string, Scheme #( ... ), or a markup { ... } block."""
    __slots__ = ("in_string", "escaped", "scheme_paren_depth", "markup_brace_depth", "saw_markup_token")
    def __init__(self) -> None:
        self.in_string: bool = False
        self.escaped: bool = False
        self.scheme_paren_depth: int = 0
        self.markup_brace_depth: int = 0
        self.saw_markup_token: bool = False  # token just seen; waiting for the opening '{'

    def in_opaque(self) -> bool:
        return self.in_string or self.scheme_paren_depth > 0 or self.markup_brace_depth > 0


def _normalize_newlines(text: str) -> str:
    # Strip BOM if present and normalize CRLF/CR to LF
    if text.startswith(BOM):
        text = text[len(BOM):]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _remove_comments(text: str, opts: CleanOptions, stats: CleanStats) -> str:
    r"""State-machine pass that removes % line comments and %{ %} block comments
    while skipping content inside strings, Scheme #( ... ), and markup { ... }.
    """
    ctx = Context()
    out_chars: list[str] = []
    i = 0
    n = len(text)

    def peek(offset: int = 0) -> str:
        j = i + offset
        return text[j] if 0 <= j < n else ""

    def startswith_at(s: str) -> bool:
        return text.startswith(s, i)

    while i < n:
        ch = text[i]

        # ── String handling (Lily strings use double quotes; support escapes) ──
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

        # Detect beginning of string when not already in opaque
        if ch == '"' and ctx.scheme_paren_depth == 0 and ctx.markup_brace_depth == 0:
            ctx.in_string = True
            out_chars.append(ch)
            i += 1
            continue

        # ── Scheme #( ... ) handling (track parens) ──
        if ctx.scheme_paren_depth > 0:
            out_chars.append(ch)
            if ch == '(':
                ctx.scheme_paren_depth += 1
            elif ch == ')':
                ctx.scheme_paren_depth -= 1
            i += 1
            continue

        # Detect start of Scheme expression
        if startswith_at("#("):
            out_chars.append("#(")
            i += 2
            ctx.scheme_paren_depth = 1
            continue

        # ── Markup handling: after a \markup token, the next non-space must be '{' ──
        if ctx.markup_brace_depth > 0:
            out_chars.append(ch)
            if ch == '{':
                # allow nesting within markup
                ctx.markup_brace_depth += 1
            elif ch == '}':
                ctx.markup_brace_depth -= 1
            i += 1
            continue

        if ctx.saw_markup_token:
            # Skip whitespace until we see '{' which starts the opaque block
            if ch.isspace():
                out_chars.append(ch)
                i += 1
                continue
            if ch == '{':
                out_chars.append(ch)
                ctx.markup_brace_depth = 1
                ctx.saw_markup_token = False
                i += 1
                continue
            # If it's not '{', then it's not a brace-based markup; cancel the flag
            ctx.saw_markup_token = False
            # fall through to normal processing

        # Detect a standalone \markup token
        if ch == '\\':
            # Copy the backslash and try to match the rest
            # We need to look ahead to see if it's the word 'markup'
            if _RE_MARKUP_TOKEN.match(text, i):
                m = _RE_MARKUP_TOKEN.match(text, i)
                assert m is not None
                out_chars.append(m.group(0))
                i = m.end()
                ctx.saw_markup_token = True
                continue
            else:
                out_chars.append(ch)
                i += 1
                continue

        # ── Comments (only when NOT in opaque regions) ──
        if ch == '%':
            # Block comment? %{ ... %}
            if text.startswith('%{', i) and not opts.keep_block_comments:
                i, removed_blocks = _consume_block_comment(text, i)
                stats.block_comments_removed += removed_blocks
                # Replace removed block with a single space if it was inline, else nothing
                # Safer choice: insert nothing; whitespace cleanup later will normalize
                continue
            # Line comment? % ... EOL
            if not opts.keep_line_comments:
                # Consume until the end of line (but keep the newline itself if present)
                i += 1
                while i < n and text[i] != '\n':
                    i += 1
                stats.line_comments_removed += 1
                continue
            # If keeping comments, just fall through to copy

        # Default: copy char
        out_chars.append(ch)
        i += 1

    return ''.join(out_chars)


def _consume_block_comment(text: str, i: int) -> Tuple[int, int]:
    r"""Given text and index at a '%{' sequence, consume a possibly nested block comment
    and return (new_index, removed_count). If unterminated, consume to end.
    """
    assert text.startswith('%{', i)
    n = len(text)
    depth = 0
    idx = i
    removed = 0
    while idx < n:
        if text.startswith('%{', idx):
            depth += 1
            idx += 2
            continue
        if text.startswith('%}', idx):
            depth -= 1
            idx += 2
            if depth == 0:
                removed += 1
                break
            continue
        # Otherwise, advance by one
        idx += 1
    return idx, removed


def _post_whitespace_cleanup(text: str, opts: CleanOptions) -> str:
    lines = text.split('\n')

    if opts.trim_trailing_ws:
        lines = [ln.rstrip(' \t\x0b\x0c') for ln in lines]

    # Collapse multiple blank lines
    if opts.max_blank_lines >= 0:
        collapsed: list[str] = []
        blank_streak = 0
        for ln in lines:
            if ln.strip() == '':
                blank_streak += 1
                if blank_streak <= opts.max_blank_lines:
                    collapsed.append('')
            else:
                blank_streak = 0
                collapsed.append(ln)
        lines = collapsed

    out = '\n'.join(lines)
    if opts.ensure_final_newline and (not out.endswith('\n')):
        out += '\n'
    return out


def clean_text(src: str, opts: CleanOptions | None = None, stats: CleanStats | None = None) -> Tuple[str, CleanStats]:
    opts = opts or CleanOptions()
    stats = stats or CleanStats()
    s = _normalize_newlines(src)
    s = _remove_comments(s, opts, stats)
    s = _post_whitespace_cleanup(s, opts)
    return s, stats


def read_text(path: str | None) -> str:
    if not path or path == '-':
        return sys.stdin.read()
    with io.open(path, 'r', encoding='utf-8', newline='') as f:
        return f.read()


def write_text(path: str | None, text: str) -> None:
    if not path or path == '-':
        sys.stdout.write(text)
        return
    with io.open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(text)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LilyPond pre-parse cleaner (lexical hygiene)")
    p.add_argument('-i', '--in', dest='infile', default='-', help='Input .ly file path (or - for stdin)')
    p.add_argument('-o', '--out', dest='outfile', default='-', help='Output file path (or - for stdout)')
    p.add_argument('--keep-line-comments', action='store_true', help='Do not remove % line comments')
    p.add_argument('--keep-block-comments', action='store_true', help='Do not remove %{ %} block comments')
    p.add_argument('--max-blank-lines', type=int, default=1, help='Max consecutive blank lines to keep (default: 1)')
    p.add_argument('--no-trim-trailing-ws', action='store_true', help='Do not trim trailing whitespace')
    p.add_argument('--no-final-newline', action='store_true', help='Do not enforce single trailing newline')
    p.add_argument('--stats', action='store_true', help='Print removal stats to stderr')
    p.add_argument('--verbose', action='store_true', help='Print what the cleaner is doing')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv or sys.argv[1:])
    opts = CleanOptions(
        keep_line_comments=ns.keep_line_comments,
        keep_block_comments=ns.keep_block_comments,
        max_blank_lines=ns.max_blank_lines,
        trim_trailing_ws=not ns.no_trim_trailing_ws,
        ensure_final_newline=not ns.no_final_newline,
    )

    raw = read_text(ns.infile)
    cleaned, stats = clean_text(raw, opts)
    write_text(ns.outfile, cleaned)

    if ns.stats:
        sys.stderr.write(
            f"line_comments_removed={stats.line_comments_removed} block_comments_removed={stats.block_comments_removed}\n"
        )
    return 0

# ─────────────────────────────────────────────────────────────
# Adapter for pipeline: run(text, opts) -> str
# ─────────────────────────────────────────────────────────────
try:
    # Import the pipeline options type for type hints; not required at runtime
    from lilynorm.utils.options import NormOptions
except Exception:
    class NormOptions:  # fallback typing stub
        keep_engraving: bool
        strip_scheme_blocks: bool
        strip_comments: bool
        normalize_whitespace: bool
        expand_relative: bool

def run(text: str, opts: "NormOptions") -> str:
    """
    Pipeline adapter: consume a LilyPond string and return the cleaned string.
    Maps pipeline flags (NormOptions) to local CleanOptions.
    """
    # NormOptions.strip_comments=True means "REMOVE comments"
    # CleanOptions.keep_* mean "KEEP", so we invert.
    co = CleanOptions(
        keep_line_comments = not getattr(opts, "strip_comments", True),
        keep_block_comments = not getattr(opts, "strip_comments", True),
        max_blank_lines = 1,  # keep your default, or expose a flag later
        trim_trailing_ws = getattr(opts, "normalize_whitespace", True),
        ensure_final_newline = True,
    )
    cleaned, stats = clean_text(text, co)

    # Light diagnostics (goes to stdout; fine for now)
    print(f"[preparse] line_removed={stats.line_comments_removed} block_removed={stats.block_comments_removed}")
    return cleaned

if __name__ == '__main__':
    # Hardcoded input file; output goes to console (stdout)
    INPUT_PATH = r"C:\Users\Navid\Desktop\13.ly"

    opts = CleanOptions()
    raw = read_text(INPUT_PATH)
    cleaned, stats = clean_text(raw, opts)

    # Print cleaned LilyPond text directly in console
    print(cleaned)

    # Optional: show stats at the end
    #print(f"\n--- Cleaned {INPUT_PATH} ---")
    #print(f"Line comments removed: {stats.line_comments_removed}")
    #print(f"Block comments removed: {stats.block_comments_removed}")