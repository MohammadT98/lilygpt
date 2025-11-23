from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from typing import Tuple, List, Dict, Iterable

DROP_EMPTY_ASSIGNMENTS = False     
PRUNE_SPACER_SUBVOICES = True      
DEFAULT_SPACE_MODE = "safe"

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
    return -1

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
        if close_idx == -1:
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


RE_ASSIGN_OPEN = re.compile(r"(?m)^(\s*\w+\s*=\s*)\{\s*$") 
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


RE_SPACER_ONLY_SUBVOICE = re.compile(
    r"(?sx)"
    r"(\\\\\{)"                        
    r"\s*(?:s[0-9.']*(?:\s+|$))+"       
    r"\s*(\})"                         
)

def _prune_spacer_only_subvoices(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = RE_SPACER_ONLY_SUBVOICE.sub("", text)
    return text


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

CATEGORIES = ("overrides", "markups", "marks", "dynamics", "hairpins", "quotes")

@dataclass
class StripOptions:
    remove_overrides: bool = True
    remove_markups:   bool = True
    remove_marks:     bool = True
    remove_dynamics:  bool = True
    remove_hairpins:  bool = True
    remove_quotes:    bool = True
    space_mode:       str  = DEFAULT_SPACE_MODE 

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

        while j < len(s) and s[j] in " \t":
            j += 1

        if deep_markup:
            i = _skip_markup_expression(s, j)
            removed += 1
            continue

        if j < len(s) and s[j] == '{':
            end = _grab_balanced(s, j, '{', '}')
            if end != -1:
                i = end + 1; removed += 1; continue

        if j < len(s) and s[j] == '"':
            j2 = j + 1
            while j2 < len(s) and s[j2] != '"':
                if s[j2] == '\\' and j2 + 1 < len(s):
                    j2 += 2
                else:
                    j2 += 1
            i = (j2 + 1) if j2 < len(s) else len(s)
            removed += 1; continue

        j2 = j
        while j2 < len(s) and not s[j2].isspace() and s[j2] not in '{}"':
            j2 += 1

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

    if opts.remove_quotes:
        text, nq = RE_ATTACHED_QUOTES.subn("", text)
        counts["quotes"] += nq

    if opts.remove_markups:
        text, nmk = _eat_after_keyword(text, RE_MARKUP, deep_markup=True)
        counts["markups"] += nmk
    if opts.remove_marks:
        text, nmark = _eat_after_keyword(text, RE_MARK)
        counts["marks"] += nmark

    if opts.remove_overrides:
        t2, n_with = _remove_with_blocks(text)
        if n_with: text, counts["overrides"] = t2, counts["overrides"] + n_with
        for directive in ("layout", "paper", "header"):
            t2, nblk = _remove_block_directive(text, directive)
            if nblk: text, counts["overrides"] = t2, counts["overrides"] + nblk
        for rgx in RE_OVERRIDES:
            t2, n = rgx.subn("", text)
            if n: text, counts["overrides"] = t2, counts["overrides"] + n

    if opts.remove_dynamics:
        text, ndyn = RE_DYNAMICS.subn("", text)
        counts["dynamics"] += ndyn
    if opts.remove_hairpins:
        text, nhp = RE_HAIRPINS.subn("", text)
        counts["hairpins"] += nhp


    text, _ = RE_STRAY_ATTACH.subn("", text)
    text, _ = RE_LONE_ONCE.subn("", text)


    text, _ = _strip_lyricmode_assignments(text)
    text, _ = _strip_inline_lyricmode(text)


    text = _collapse_empty_assignment_blocks(text)
    text = RE_EMPTY_BLOCK_LINE.sub("", text)
    text = RE_EMPTY_ASSIGNMENT_LINE.sub("", text)
    text = RE_INLINE_EMPTY_BRACES.sub(" ", text)
    text = RE_INCLUDE_TAG.sub("", text)
    text = RE_REPEATED_INCLUDE.sub("", text)
    text = RE_EMPTY_SCORES.sub("", text)
    text = RE_EMPTY_LAYOUT_BLOCK.sub("", text)


    if DROP_EMPTY_ASSIGNMENTS:
        text = re.sub(r"(?m)^\s*\w+\s*=\s*\{\s*\}\s*$", "", text)


    if PRUNE_SPACER_SUBVOICES:
        text = _prune_spacer_only_subvoices(text)
    if opts.space_mode == "simple":
        text = _compact_spaces_simple(text)
    else:
        text = _compact_spaces_safe(text)

    return text, counts
