from __future__ import annotations

import io
import os
import re
import sys
from dataclasses import dataclass
from typing import Tuple

BOM = "\ufeff"

_RE_WS = re.compile(r"\s+")
_RE_MARKUP_TOKEN = re.compile(r"\\markup(?![A-Za-z])")

@dataclass
class CleanOptions:
    keep_line_comments: bool = False
    keep_block_comments: bool = False
    max_blank_lines: int = 1
    trim_trailing_ws: bool = True
    ensure_final_newline: bool = True

@dataclass
class CleanStats:
    line_comments_removed: int = 0
    block_comments_removed: int = 0

class Context:
    __slots__ = ("in_string", "escaped", "scheme_paren_depth", "markup_brace_depth", "saw_markup_token")
    def __init__(self) -> None:
        self.in_string: bool = False
        self.escaped: bool = False
        self.scheme_paren_depth: int = 0
        self.markup_brace_depth: int = 0
        self.saw_markup_token: bool = False  

    def in_opaque(self) -> bool:
        return self.in_string or self.scheme_paren_depth > 0 or self.markup_brace_depth > 0


def _normalize_newlines(text: str) -> str:
    if text.startswith(BOM):
        text = text[len(BOM):]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _remove_comments(text: str, opts: CleanOptions, stats: CleanStats) -> str:
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

        if ch == '"' and ctx.scheme_paren_depth == 0 and ctx.markup_brace_depth == 0:
            ctx.in_string = True
            out_chars.append(ch)
            i += 1
            continue

        if ctx.scheme_paren_depth > 0:
            out_chars.append(ch)
            if ch == '(':
                ctx.scheme_paren_depth += 1
            elif ch == ')':
                ctx.scheme_paren_depth -= 1
            i += 1
            continue

        if startswith_at("#("):
            out_chars.append("#(")
            i += 2
            ctx.scheme_paren_depth = 1
            continue

        if ctx.markup_brace_depth > 0:
            out_chars.append(ch)
            if ch == '{':
                ctx.markup_brace_depth += 1
            elif ch == '}':
                ctx.markup_brace_depth -= 1
            i += 1
            continue

        if ctx.saw_markup_token:
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
            
            ctx.saw_markup_token = False
  

     
        if ch == '\\':
      
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

     
        if ch == '%':
            
            if text.startswith('%{', i) and not opts.keep_block_comments:
                i, removed_blocks = _consume_block_comment(text, i)
                stats.block_comments_removed += removed_blocks
                
                continue
           
            if not opts.keep_line_comments:
                
                i += 1
                while i < n and text[i] != '\n':
                    i += 1
                stats.line_comments_removed += 1
                continue
           

        out_chars.append(ch)
        i += 1

    return ''.join(out_chars)


def _consume_block_comment(text: str, i: int) -> Tuple[int, int]:
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
     
        idx += 1
    return idx, removed


def _post_whitespace_cleanup(text: str, opts: CleanOptions) -> str:
    lines = text.split('\n')

    if opts.trim_trailing_ws:
        lines = [ln.rstrip(' \t\x0b\x0c') for ln in lines]

   
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


try:
   
    from lilynorm.utils.options import NormOptions
except Exception:
    class NormOptions:  
        keep_engraving: bool
        strip_scheme_blocks: bool
        strip_comments: bool
        normalize_whitespace: bool
        expand_relative: bool

def run(text: str, opts: "NormOptions") -> str:
    co = CleanOptions(
        keep_line_comments = not getattr(opts, "strip_comments", True),
        keep_block_comments = not getattr(opts, "strip_comments", True),
        max_blank_lines = 1, 
        trim_trailing_ws = getattr(opts, "normalize_whitespace", True),
        ensure_final_newline = True,
    )
    cleaned, stats = clean_text(text, co)

 
    print(f"[preparse] line_removed={stats.line_comments_removed} block_removed={stats.block_comments_removed}")
    return cleaned