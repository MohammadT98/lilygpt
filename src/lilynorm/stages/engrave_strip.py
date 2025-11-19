#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
engrave_strip.py — Remove engravings/markups/dynamics from LilyPond text (order-safe).
Assumes an earlier preparse stage already removed comments and normalized newlines.

You can choose whitespace behavior:
  --space-mode safe   (default) token-aware compacting, keeps tricky spacing edges
  --space-mode simple collapse any run of spaces/tabs to one space per line

Other safety:
  • Eats attached operators with their targets (mi^\f → mi, ^"solo" →)
  • Consumes \markup / \mark WITH their argument (block, string, bare token),
    including attached forms like ^\markup …
  • Non-greedy, single-line overrides (override/tweak/shape/omit/once)
  • Cleans up stray -/_/^ and orphan \once
  • Collapses whitespace-only assignment blocks:  Iglobal = { … } → Iglobal = {}
  • Optional pruning of spacer-only \\{ s… } subvoices

Defaults strip everything; customize with --keep / --remove / --keep-all.
"""
from __future__ import annotations
import re
import sys
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict, Iterable

# ─────────────────────────────────────────────────────────────
# Configuration toggles (safe defaults)
# ─────────────────────────────────────────────────────────────
DROP_EMPTY_ASSIGNMENTS = False      # keep `Name = {}` by default
PRUNE_SPACER_SUBVOICES = True       # remove \\{ s2 s4 … } subvoices
DEFAULT_SPACE_MODE = "safe"         # 'safe' or 'simple'

# Hardcoded input path for CLI mode (pipeline use calls run(text, opts))
INPUT_PATH = r"C:\Users\Navid\Desktop\00.ly"

# ─────────────────────────────────────────────────────────────
# Balanced-block helpers (for { … } scans)
# ─────────────────────────────────────────────────────────────
def _grab_balanced(text: str, start: int, open_char: str = '{', close_char: str = '}') -> int:
    depth = 1
    i = start + 1
    L = len(text)
    while i < L:
        c = text[i]
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1  # unbalanced

def _remove_block_directive(src: str, directive: str) -> Tuple[str, int]:
    pat = re.compile(rf"\\{directive}\s*\{{", re.M)
    removed = 0
    out, i = [], 0
    while True:
        m = pat.search(src, i)
        if not m:
            out.append(src[i:]); break
        out.append(src[i:m.start()])
        open_idx = m.end() - 1
        close_idx = _grab_balanced(src, open_idx, '{', '}')
        if close_idx == -1:   # malformed; keep token
            out.append(src[m.start():m.end()])
            i = m.end()
            continue
        removed += 1
        i = close_idx + 1
    return ("".join(out), removed)

def _remove_with_blocks(src: str) -> Tuple[str, int]:
    pat = re.compile(r"\\with\s*\{", re.M)
    removed = 0
    out, i = [], 0
    while True:
        m = pat.search(src, i)
        if not m:
            out.append(src[i:]); break
        out.append(src[i:m.start()])
        open_idx = m.end() - 1
        close_idx = _grab_balanced(src, open_idx, '{', '}')
        if close_idx == -1:
            out.append(src[m.start():m.end()])
            i = m.end()
            continue
        removed += 1
        i = close_idx + 1
    return ("".join(out), removed)

def _strip_lyricmode_assignments(text: str) -> Tuple[str, int]:
    """
    Replace `Name = \\lyricmode { ... }` blocks with empty assignments.
    Keeps the left-hand side so references remain valid.
    """
    removed = 0
    while True:
        m = RE_LYRIC_ASSIGN.search(text)
        if not m:
            break
        open_idx = m.end() - 1
        close_idx = _grab_balanced(text, open_idx, '{', '}')
        if close_idx == -1:
            break
        prefix = m.group(1)
        text = text[:m.start()] + prefix + "{}" + text[close_idx + 1 :]
        removed += 1
    return text, removed

def _strip_inline_lyricmode(text: str) -> Tuple[str, int]:
    """
    Remove standalone `\\lyricmode { ... }` blocks (e.g., under \\new Lyrics).
    """
    removed = 0
    i = 0
    out: list[str] = []
    while True:
        m = RE_LYRIC_INLINE.search(text, i)
        if not m:
            out.append(text[i:])
            break
        start = m.start()
        open_idx = m.end() - 1
        close_idx = _grab_balanced(text, open_idx, '{', '}')
        if close_idx == -1:
            out.append(text[i:m.end()])
            i = m.end()
            continue
        prefix = text[i:start]
        trimmed = prefix.rstrip(" \t")
        out.append(trimmed)
        out.append("{}")
        removed += 1
        i = close_idx + 1
    return "".join(out), removed

# ─────────────────────────────────────────────────────────────
# Patterns (single-line, non-greedy)
# ─────────────────────────────────────────────────────────────
RE_OVERRIDES = [
    re.compile(r"(?:\\once\s+)?\\override\b[^\n\r{}]*", re.I),
    re.compile(r"(?:\\once\s+)?\\revert\b[^\s{}]+", re.I),
    re.compile(r"(?:\\once\s+)?(?:[-_^]\s*)?\\tweak\b[^\n\r{}]*", re.I),
    re.compile(r"(?:\\once\s+)?\\shape\b[^\n\r{}]*", re.I),
    re.compile(r"(?:\\once\s+)?\\(?:undo\s+)?omit\b[^\s{}]+", re.I),
    re.compile(r"(?:\\once\s+)?\\(?:hideNotes|magnifyStaff|teeny|tiny|small|large|huge)\b(?:[^\S\n][^\n\r{}]*)?"),
]

RE_MARKUP = re.compile(r"(?:[-_^]\s*)?\\markup\b", re.I)
RE_MARK   = re.compile(r"(?:[-_^]\s*)?\\mark\b",   re.I)

DYNAMICS = ("ppppp|pppp|ppp|pp|p|mp|mf|f|ff|fff|ffff|fffff|fp|sf|sfz|sffz|rfz|fz|sfp|sff|sfpp|sfzp")
RE_DYNAMICS = re.compile(rf"(?:[-_^]\s*)?\\(?:{DYNAMICS})\b", re.I)

RE_HAIRPINS = re.compile(r"\\[<>!]|\\(?:cresc|decresc|decr|dim|crescendo|diminuendo)\b", re.I)
RE_ATTACHED_QUOTES = re.compile(r"(?:[-_^]\s*)\"[^\"]*\"")
RE_LYRIC_ASSIGN = re.compile(r"(?m)(^\s*[A-Za-z_@][\w@]*\s*=\s*)\\lyricmode\s*\{")
RE_LYRIC_INLINE = re.compile(r"\\lyricmode\s*\{", re.I)

# ─────────────────────────────────────────────────────────────
# Whitespace + cleanup (safe mode)
# ─────────────────────────────────────────────────────────────
HSPACE = re.compile(r"[ \t]+")
RE_STRAY_ATTACH = re.compile(
    r"(?m)([-_^])(?=\s*(?:$|[\r\n]|[,;:|)}\]]|(?!(?:\\|\"|\{|\<|\>|\!|[a-gris][',]*))))"
)
RE_LONE_ONCE = re.compile(r"(?m)\\once\b(?:[ \t]+(?=$|[\r\n}])|[ \t]*(?!\\))")
RE_SPACE_BEFORE_CLOSER = re.compile(r"[ \t]+(?=[)\]}])")
RE_SPACE_AFTER_OPENER  = re.compile(r"(?<=[({\[])[ \t]+")
RE_SPACE_BEFORE_PUNCT  = re.compile(r"[ \t]+(?=[,;:|>])")
RE_NOTE_OCTAVE_SPACE   = re.compile(r"(?i)(?<=\b[a-gr])\s+(?=[',])")
RE_MULTI_BLANKS        = re.compile(r"\n{3,}")

# Detect assignments with a balanced block body we can inspect/collapse
RE_ASSIGN_OPEN = re.compile(r"(?m)^(\s*\w+\s*=\s*)\{\s*$")  # captures "Name = {" at line start
RE_EMPTY_BLOCK_LINE = re.compile(r"(?m)^\s*\{\s*\}\s*$")
RE_EMPTY_ASSIGNMENT_LINE = re.compile(r"(?m)^\s*[A-Za-z_@][\w@]*\s*=\s*(?:\{\s*\})?\s*$")
RE_INLINE_EMPTY_BRACES = re.compile(r"(?<=\s)\{\s*\}(?=\s)")
RE_INCLUDE_TAG = re.compile(r"(?m)^\s*<<\s*\\\s*@\w+\b.*?>>\s*$")
RE_REPEATED_INCLUDE = re.compile(r"(<<\s*\\\s*@\w+\b.*?>>\s*)+", re.S)
RE_EMPTY_SCORES = re.compile(r"(?ms)^\\score\s*\{\s*\{\s*\}\s*(?:\\layout\s*\{.*?\}\s*)?\}\s*$")
RE_EMPTY_LAYOUT_BLOCK = re.compile(r"(?ms)^\\layout\s*\{\s*\}$")

def _collapse_empty_assignment_blocks(text: str) -> str:
    i = 0
    out = []
    L = len(text)
    while i < L:
        m = RE_ASSIGN_OPEN.search(text, i)
        if not m:
            out.append(text[i:]); break
        out.append(text[i:m.start()])
        prefix = m.group(1)  # "Name = "
        brace_open_pos = m.end() - 1
        close_idx = _grab_balanced(text, brace_open_pos, '{', '}')
        if close_idx == -1:
            line_end = text.find("\n", m.end())
            if line_end == -1: line_end = L
            out.append(text[m.start():line_end])
            i = line_end
            continue
        inner = text[brace_open_pos+1:close_idx]
        if inner.strip() == "":
            out.append(f"{prefix}{{}}\n")
            j = close_idx + 1
            if j < L and text[j] == "\n":
                j += 1
            i = j
        else:
            out.append(text[m.start():close_idx+1])
            i = close_idx + 1
    return "".join(out)

# OPTIONAL: remove subvoices that contain only spacer rests
RE_SPACER_ONLY_SUBVOICE = re.compile(
    r"(?sx)"
    r"(\\\\\{)"                         # opening \\{
    r"\s*(?:s[0-9.']*(?:\s+|$))+"       # one or more spacers with optional duration
    r"\s*(\})"                          # closing }
)

def _prune_spacer_only_subvoices(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = RE_SPACER_ONLY_SUBVOICE.sub("", text)
    return text

# ─────────────────────────────────────────────────────────────
# Space compaction modes
# ─────────────────────────────────────────────────────────────
def _compact_spaces_safe(text: str) -> str:
    lines = text.splitlines()
    text = "\n".join(HSPACE.sub(" ", ln).strip() for ln in lines)
    text = RE_SPACE_BEFORE_CLOSER.sub("", text)
    text = RE_SPACE_AFTER_OPENER.sub("", text)
    text = RE_SPACE_BEFORE_PUNCT.sub("", text)
    text = RE_NOTE_OCTAVE_SPACE.sub("", text)
    text = RE_MULTI_BLANKS.sub("\n\n", text).strip() + "\n"
    return text

def _compact_spaces_simple(text: str) -> str:
    """Collapse any run of spaces/tabs to one space per line; preserve newlines."""
    lines = text.splitlines()
    lines = [re.sub(r"[ \t]{2,}", " ", ln).rstrip() for ln in lines]
    return "\n".join(lines).strip() + "\n"

# ─────────────────────────────────────────────────────────────
# Options + core stripping
# ─────────────────────────────────────────────────────────────
CATEGORIES = ("overrides", "markups", "marks", "dynamics", "hairpins", "quotes")

@dataclass
class StripOptions:
    remove_overrides: bool = True
    remove_markups:   bool = True
    remove_marks:     bool = True
    remove_dynamics:  bool = True
    remove_hairpins:  bool = True
    remove_quotes:    bool = True
    space_mode:       str  = DEFAULT_SPACE_MODE  # 'safe' or 'simple'

    @classmethod
    def from_sets(cls, remove: Iterable[str], keep: Iterable[str], *, space_mode: str = DEFAULT_SPACE_MODE) -> "StripOptions":
        flags = {k: True for k in CATEGORIES}
        for k in remove:
            if k in flags:
                flags[k] = True
        for k in keep:
            if k in flags:
                flags[k] = False
        return cls(
            remove_overrides=flags["overrides"],
            remove_markups=flags["markups"],
            remove_marks=flags["marks"],
            remove_dynamics=flags["dynamics"],
            remove_hairpins=flags["hairpins"],
            remove_quotes=flags["quotes"],
            space_mode=space_mode,
        )

def _skip_markup_expression(s: str, idx: int) -> int:
    """Skip a LilyPond markup expression beginning at idx."""
    i = idx
    L = len(s)
    while i < L:
        ch = s[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch == '{':
            end = _grab_balanced(s, i, '{', '}')
            return end + 1 if end != -1 else L
        if ch == '"':
            i += 1
            escaped = False
            while i < L:
                curr = s[i]
                i += 1
                if curr == '"' and not escaped:
                    break
                escaped = (curr == "\\") and not escaped
            continue
        if ch == '\\':
            i += 1
            while i < L and (s[i].isalnum() or s[i] in "_-"):
                i += 1
            continue
        if ch == '#':
            i += 1
            while i < L and not s[i].isspace():
                i += 1
            continue
        # bare token
        start = i
        while i < L and not s[i].isspace() and s[i] not in '{}"':
            i += 1
        if i == start:
            i += 1
    return i


def _eat_after_keyword(s: str, kw_re: re.Pattern, *, deep_markup: bool = False) -> Tuple[str, int]:
    removed = 0
    i = 0
    out = []
    while True:
        m = kw_re.search(s, i)
        if not m:
            out.append(s[i:]); break
        out.append(s[i:m.start()])
        j = m.end()

        # skip only spaces/tabs (preserve newlines)
        while j < len(s) and s[j] in " \t":
            j += 1

        if deep_markup:
            i = _skip_markup_expression(s, j)
            removed += 1
            continue

        # {block}
        if j < len(s) and s[j] == '{':
            end = _grab_balanced(s, j, '{', '}')
            if end != -1:
                i = end + 1; removed += 1; continue

        # "string"
        if j < len(s) and s[j] == '"':
            j2 = j + 1
            while j2 < len(s) and s[j2] != '"':
                if s[j2] == '\\' and j2 + 1 < len(s):
                    j2 += 2
                else:
                    j2 += 1
            i = (j2 + 1) if j2 < len(s) else len(s)
            removed += 1; continue

        # bare token (until space/newline/brace/quote)
        j2 = j
        while j2 < len(s) and not s[j2].isspace() and s[j2] not in '{}"':
            j2 += 1

        # optional trailing "string"
        k = j2
        while k < len(s) and s[k] in " \t":
            k += 1
        if k < len(s) and s[k] == '"':
            k2 = k + 1
            while k2 < len(s) and s[k2] != '"':
                if s[k2] == '\\' and k2 + 1 < len(s):
                    k2 += 2
                else:
                    k2 += 1
            j2 = (k2 + 1) if k2 < len(s) else len(s)

        i = j2
        removed += 1
    return ("".join(out), removed)

def _strip_inline_patterns(text: str, opts: StripOptions) -> Tuple[str, Dict[str, int]]:
    counts = {"overrides":0, "markups":0, "marks":0, "dynamics":0, "hairpins":0, "quotes":0}

    # 1) Attached quotes (^"solo")
    if opts.remove_quotes:
        text, nq = RE_ATTACHED_QUOTES.subn("", text)
        counts["quotes"] += nq

    # 2) Markups / Marks — eat their argument; handle attached forms too
    if opts.remove_markups:
        text, nmk = _eat_after_keyword(text, RE_MARKUP, deep_markup=True)
        counts["markups"] += nmk
    if opts.remove_marks:
        text, nmark = _eat_after_keyword(text, RE_MARK)
        counts["marks"] += nmark

    # 3) Overrides: blocks first, then inline
    if opts.remove_overrides:
        t2, n_with = _remove_with_blocks(text)
        if n_with: text, counts["overrides"] = t2, counts["overrides"] + n_with
        for directive in ("layout", "paper", "header"):
            t2, nblk = _remove_block_directive(text, directive)
            if nblk: text, counts["overrides"] = t2, counts["overrides"] + nblk
        for rgx in RE_OVERRIDES:
            t2, n = rgx.subn("", text)
            if n: text, counts["overrides"] = t2, counts["overrides"] + n

    # 4) Dynamics and Hairpins (eat attached operators)
    if opts.remove_dynamics:
        text, ndyn = RE_DYNAMICS.subn("", text)
        counts["dynamics"] += ndyn
    if opts.remove_hairpins:
        text, nhp = RE_HAIRPINS.subn("", text)
        counts["hairpins"] += nhp

    # 5) Clean up stray attachments and orphan \once
    text, _ = RE_STRAY_ATTACH.subn("", text)
    text, _ = RE_LONE_ONCE.subn("", text)

    # 6) Drop lyricmode blocks (assignments + inline Lyrics contexts)
    text, _ = _strip_lyricmode_assignments(text)
    text, _ = _strip_inline_lyricmode(text)

    # 7) Collapse whitespace-only assignment blocks robustly
    text = _collapse_empty_assignment_blocks(text)
    text = RE_EMPTY_BLOCK_LINE.sub("", text)
    text = RE_EMPTY_ASSIGNMENT_LINE.sub("", text)
    text = RE_INLINE_EMPTY_BRACES.sub(" ", text)
    text = RE_INCLUDE_TAG.sub("", text)
    text = RE_REPEATED_INCLUDE.sub("", text)
    text = RE_EMPTY_SCORES.sub("", text)
    text = RE_EMPTY_LAYOUT_BLOCK.sub("", text)

    # 8) Optionally drop whole-line empty assignments
    if DROP_EMPTY_ASSIGNMENTS:
        text = re.sub(r"(?m)^\s*\w+\s*=\s*\{\s*\}\s*$", "", text)

    # 9) Optionally remove spacer-only subvoices (\\{ s2 s4 … })
    if PRUNE_SPACER_SUBVOICES:
        text = _prune_spacer_only_subvoices(text)

    # 10) Whitespace compaction (mode)
    if opts.space_mode == "simple":
        text = _compact_spaces_simple(text)
    else:
        text = _compact_spaces_safe(text)

    return text, counts

def clean_lilypond(src: str, opts: StripOptions) -> Tuple[str, Dict[str,int]]:
    text, counts = _strip_inline_patterns(src, opts)
    return text, counts

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optionally strip engraving/markup/dynamics from LilyPond.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--keep-all",   action="store_true", help="Keep everything (strip nothing).")
    grp.add_argument("--remove-all", action="store_true", help="Remove all categories (default).")

    p.add_argument("--keep",   nargs="+", choices=CATEGORIES, default=[],
                   help="Keep these categories (override default removal).")
    p.add_argument("--remove", nargs="+", choices=CATEGORIES, default=[],
                   help="Remove ONLY these categories (others are kept).")
    p.add_argument("--show-report", action="store_true", help="Print removal counts to stderr.")
    p.add_argument("--space-mode", choices=("safe","simple"), default=DEFAULT_SPACE_MODE,
                   help="Whitespace compaction: 'safe' (default) or 'simple' (collapse runs to one space).")
    p.add_argument("--drop-empty-assignments", action="store_true",
                   help="Remove lines like 'Name = {}' after collapsing.")
    p.add_argument("--keep-spacer-subvoices", action="store_true",
                   help="Do not prune \\{ s… } spacer-only subvoices.")
    return p.parse_args(argv)

def _opts_from_args(ns: argparse.Namespace) -> StripOptions:
    global DROP_EMPTY_ASSIGNMENTS, PRUNE_SPACER_SUBVOICES
    if ns.drop_empty_assignments:
        DROP_EMPTY_ASSIGNMENTS = True
    if ns.keep_spacer_subvoices:
        PRUNE_SPACER_SUBVOICES = False

    if ns.keep_all:
        return StripOptions.from_sets(remove=[], keep=CATEGORIES, space_mode=ns.space_mode)
    if ns.remove:
        keep_set = [c for c in CATEGORIES if c not in ns.remove]
        return StripOptions.from_sets(remove=ns.remove, keep=keep_set, space_mode=ns.space_mode)
    if ns.keep:
        return StripOptions.from_sets(remove=CATEGORIES, keep=ns.keep, space_mode=ns.space_mode)
    return StripOptions(space_mode=ns.space_mode)  # default: remove all

def main(argv: List[str] | None = None) -> int:
    ns = parse_args(argv)
    opts = _opts_from_args(ns)
    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except FileNotFoundError:
        sys.stderr.write(f"File not found: {INPUT_PATH}\n")
        return 1

    cleaned, counts = clean_lilypond(raw, opts)
    sys.stdout.write(cleaned)

    if ns.show_report:
        report = (
            f"--- Post-parser report ---\n"
            f"Overrides removed: {counts['overrides']}\n"
            f"Markups removed:   {counts['markups']}\n"
            f"Marks removed:     {counts['marks']}\n"
            f"Dynamics removed:  {counts['dynamics']}\n"
            f"Hairpins removed:  {counts['hairpins']}\n"
            f'Quotes removed:    {counts["quotes"]}\n'
        )
        sys.stderr.write(report)
    return 0

# ─────────────────────────────────────────────────────────────
# Pipeline adapter: run(text, opts) -> str
# ─────────────────────────────────────────────────────────────
try:
    from lilynorm.utils.options import NormOptions
except Exception:
    class NormOptions:
        keep_engraving: bool = True  # default: keep engravings

def run(text: str, opts: "NormOptions") -> str:
    """
    Stage-3 entrypoint for CLI pipeline.
    If opts.keep_engraving=True, return text unchanged.
    Otherwise strip engravings using default StripOptions.
    """
    if getattr(opts, "keep_engraving", True):
        print("[engrave_strip] keeping engravings", file=sys.stderr)
        return text

    global DROP_EMPTY_ASSIGNMENTS
    DROP_EMPTY_ASSIGNMENTS = True

    strip_opts = StripOptions(
        remove_overrides=True,
        remove_markups=True,
        remove_marks=True,
        remove_dynamics=True,
        remove_hairpins=True,
        remove_quotes=True,
        space_mode=DEFAULT_SPACE_MODE,
    )

    cleaned, counts = clean_lilypond(text, strip_opts)
    print(
        f"[engrave_strip] overrides:{counts['overrides']} "
        f"markups:{counts['markups']} marks:{counts['marks']} "
        f"dynamics:{counts['dynamics']} hairpins:{counts['hairpins']} "
        f"quotes:{counts['quotes']}",
        file=sys.stderr,
    )
    return cleaned

if __name__ == "__main__":
    raise SystemExit(main())
