import re
import sys
import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set
from dataclasses import dataclass, asdict, field, fields
import argparse
DEFAULT_LILYPOND_PATH = r"C:\lilypond-2.24.4-mingw-x86_64\lilypond-2.24.4\bin\lilypond.exe"
DEBUG = False
def debug_print(label: str, content: str, separator: bool = True):
    if not DEBUG:
        return
    if separator:
        print("\n" + "=" * 80, file=sys.stderr)
    print(f"DEBUG [{label}]:", file=sys.stderr)
    print(content, file=sys.stderr)
    if separator:
        print("=" * 80 + "\n", file=sys.stderr)
@dataclass
class ParseOptions:
    expand_relative: bool = True
    inline_variables: bool = True              
    expand_music_functions: bool = True
    resolve_transpose: bool = True
    expand_repeat_unfold: bool = True          
    normalize_tuplets: bool = True             
    normalize_drums: bool = True               
    normalize_whitespace: bool = False
    preserve_linebreaks: bool = True
    canonicalize_chord_brackets: bool = True

_DEFAULT_PARSE_OPTIONS = ParseOptions()
@dataclass
class ParseReport:
    relative_blocks: int = 0
    variables_inlined: int = 0
    transpose_blocks: int = 0
    repeats_unfolded: int = 0
    tuplets_normalized: int = 0
    drum_blocks_normalized: int = 0
    lily_failures: int = 0
    notes: List[str] = field(default_factory=list)

def _grab_braces(s: str, i: int) -> int:
    depth, j = 1, i + 1
    while j < len(s) and depth > 0:
        ch = s[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        j += 1
    return j if depth == 0 else len(s)
def _grab_angles(s: str, i: int) -> int:
    depth, j = 1, i + 2
    while j < len(s) and depth > 0:
        if s.startswith("<<", j):
            depth += 1
            j += 2
        elif s.startswith(">>", j):
            depth -= 1
            j += 2
        else:
            j += 1
    return j if depth == 0 else len(s)

def _ok(cmd: str) -> bool:
    try:
        r = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False
def lily_available(lily_cmd: str) -> bool:
    return _ok(lily_cmd)
def resolve_lily_cmd() -> str:
    wh = shutil.which("lilypond")
    if wh and _ok(wh):
        return wh
    env = os.environ.get("LILYPOND_BIN")
    if env and _ok(env):
        return env
    if os.path.isfile(DEFAULT_LILYPOND_PATH) and _ok(DEFAULT_LILYPOND_PATH):
        return DEFAULT_LILYPOND_PATH
    return "lilypond"

RE_RELATIVE_BLK = re.compile(
    r"\\relative\b(?:\s+[^\s{}%]+)?(?:\s*(?:%[^\n]*\n|\s))*\{",
    re.I,
)
RE_RELATIVE_TOKEN = re.compile(r"\\relative\b\s+([^\s{}%]+)", re.I)
RE_LANGUAGE_DECL = re.compile(r"\\language\s+\"([^\"]+)\"", re.I)
ITALIAN_SOLFEGE = ("do", "re", "mi", "fa", "sol", "la", "si")

def _detect_note_language(source: str) -> Optional[str]:
    r"""
    Try to infer the active note language from the source.
    Preference order:
      1. Explicit \language "..." declaration.
      2. Heuristic: \relative followed by Italian solfege.
    Returns None if no guess can be made (defaults to Lily's english).
    """
    m = RE_LANGUAGE_DECL.search(source)
    if m:
        return m.group(1)
    for m in RE_RELATIVE_TOKEN.finditer(source):
        token = m.group(1).strip().lower()
        token = token.strip(",;'\"")  
        for sol in ITALIAN_SOLFEGE:
            if token.startswith(sol):
                return "italiano"
    return None
def _find_relative_blocks(source: str) -> List[Tuple[int, int, str]]:
    blocks = []
    i = 0
    while True:
        m = RE_RELATIVE_BLK.search(source, i)
        if not m:
            break
        lb = m.end() - 1
        rb = _grab_braces(source, lb)
        blocks.append((m.start(), rb, source[m.start():rb]))
        i = rb
    return blocks

RE_ASSIGN = re.compile(r"(^|[^\w-])([A-Za-z][\w-]*)\s*=\s*", re.M)
def _collect_named_music(source: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    i = 0
    n = len(source)
    while i < n:
        m = RE_ASSIGN.search(source, i)
        if not m:
            break
        j = m.end()
        while j < n and source[j].isspace():
            j += 1
        name = m.group(2)
        if source.startswith("\\relative", j):
            mrel = RE_RELATIVE_BLK.search(source, j)
            if not mrel:
                i = j + 1
                continue
            lb = mrel.end() - 1
            rb = _grab_braces(source, lb)
            env[name] = source[j:rb]
            i = rb
        elif j < n and source[j] == "{":
            rb = _grab_braces(source, j)
            env[name] = source[j:rb]
            i = rb
        elif source.startswith("<<", j):
            rb = _grab_angles(source, j)
            env[name] = source[j:rb]
            i = rb
        else:
            k = j
            while k < n and not source[k].isspace():
                if source[k] in "{}<>":
                    break
                k += 1
            env[name] = source[j:k]
            i = k
    return env
def _inline_once(source: str, env: Dict[str, str]) -> Tuple[str, int]:
    if not env:
        return source, 0
    count = 0
    names = sorted(env.keys(), key=len, reverse=True)
    pat = r"\\(" + "|".join(re.escape(n) for n in names) + r")\b"
    def repl(match: re.Match) -> str:
        nonlocal count
        nm = match.group(1)
        if nm in env:
            count += 1
            return env[nm]
        return match.group(0)
    out = re.sub(pat, repl, source)
    return out, count
def _inline_named_music_recursive(source: str, env: Dict[str, str], *, max_passes: int = 8) -> Tuple[str, int]:
    total = 0
    seen_hashes: Set[int] = set()
    s = source
    for _ in range(max_passes):
        h = hash(s)
        if h in seen_hashes:
            break
        seen_hashes.add(h)
        s2, cnt = _inline_once(s, env)
        total += cnt
        if cnt == 0:
            return s2, total
        s = s2
    return s, total

RE_DMF_HEADER = re.compile(
    r"(^|[\n;])\s*([A-Za-z][\w-]*)\s*=\s*#\(\s*define-music-function\b",
    re.S
)
def _collect_music_function_names(source: str) -> List[str]:
    names = []
    for m in RE_DMF_HEADER.finditer(source):
        names.append(m.group(2))
    return sorted(set(names), key=len, reverse=True)
def _extract_music_function_snippets(source: str) -> str:
    preamble_parts: List[str] = []
    for m in RE_DMF_HEADER.finditer(source):
        name_start = m.start(2)
        eq = source.find("=", name_start)
        if eq == -1:
            continue
        search_from = eq + 1
        open_hash = source.find("#{", search_from)
        if open_hash == -1:
            continue
        close_hash = source.find("#}", open_hash + 2)
        if close_hash == -1:
            continue
        close_paren = source.find(")", close_hash + 2)
        if close_paren == -1:
            continue
        snippet = source[name_start:close_paren + 1]
        preamble_parts.append(snippet)
    return "\n".join(preamble_parts)
def _find_function_calls(source: str, func_names: List[str]) -> List[Tuple[int,int,str]]:
    if not func_names:
        return []
    name_pat = r"\\(?:" + "|".join(re.escape(n) for n in func_names) + r")\b"
    calls: List[Tuple[int,int,str]] = []
    i = 0
    n = len(source)
    regex = re.compile(name_pat)
    while True:
        m = regex.search(source, i)
        if not m:
            break
        start = m.start()
        j = m.end()
        while j < n and source[j].isspace():
            j += 1
        if j >= n:
            i = j
            continue
        if source.startswith("{", j):
            rb = _grab_braces(source, j)
            end = rb
        elif source.startswith("<<", j):
            rb = _grab_angles(source, j)
            end = rb
        elif source.startswith("<", j):
            k = j + 1
            while k < n and source[k] != ">":
                k += 1
            end = k + 1 if k < n else n
        else:
            k = j
            while k < n and not source[k].isspace() and source[k] not in "{}<>":
                k += 1
            end = k
        calls.append((start, end, source[start:end]))
        i = end
    return calls

def _normalize_line_keep_newlines(seg: str) -> str:
    seg = seg.replace("\r\n", "\n").replace("\r", "\n")
    lines = seg.split("\n")
    norm = []
    for ln in lines:
        x = re.sub(r"[ \t]+", " ", ln.strip())
        x = re.sub(r"[ ]*\{[ ]*", " { ", x)
        x = re.sub(r"[ ]*\}[ ]*", " } ", x)
        x = re.sub(r"[ ]*<<[ ]*", " << ", x)
        x = re.sub(r"[ ]*>>[ ]*", " >> ", x)
        x = re.sub(r"[ ]{2,}", " ", x)
        norm.append(x)
    return "\n".join(norm).strip()
_CHORD_RE = re.compile(r"<([^>]*)>")
def _canonicalize_chord_brackets(s: str) -> str:
    def _fix(m: re.Match) -> str:
        inner = m.group(1)
        inner = " ".join(inner.split())
        return f"<{inner}>"
    s = _CHORD_RE.sub(_fix, s)
    s = re.sub(r"<\s+", "<", s)
    s = re.sub(r"\s+>", ">", s)
    return s

def _unwrap_absolute_layers(seg: str) -> str:
    seg = seg.strip()
    while seg.startswith("\\absolute"):
        i = seg.find("{")
        if i == -1:
            break
        j = _grab_braces(seg, i)
        seg = seg[i+1:j-1].strip()
    return seg
def _run_lily_batch(blocks: List[str], lily_cmd: str, *, preserve_linebreaks: bool, preamble: str = "") -> List[Optional[str]]:
    parts = ['\version "2.24.4"']
    if preamble:
        parts.append(preamble)
    def var_name(idx: int) -> str:

        def _letters(n: int) -> str:
            if n < 0:
                return "a"
            parts = []
            while True:
                parts.append(chr(ord('a') + (n % 26)))
                n = n // 26 - 1
                if n < 0:
                    break
            return "".join(reversed(parts))

        return f"music{_letters(idx)}"
    var_names = [var_name(i) for i in range(len(blocks))]
    for idx, blk in enumerate(blocks):
        parts.append(f"{var_names[idx]} = \\absolute {{ {blk} }}")
    for idx in range(len(blocks)):
        parts.append(f"#(display \"===BEGIN_{idx}===\\n\")")
        parts.append(f"\\displayLilyMusic \\{var_names[idx]}")
        parts.append(f"#(display \"===END_{idx}===\\n\")")
    ly_source = "\n".join(parts) + "\n"
    out = ""
    with tempfile.TemporaryDirectory() as tdir:
        tmp = Path(tdir, "input.ly")
        tmp.write_text(ly_source, encoding="utf-8")
        try:
            proc = subprocess.run(
                [lily_cmd, "-dno-print-pages", "-dbackend=null",
                 "-o", str(Path(tdir, "dump")), str(tmp)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=120, encoding="utf-8"
            )
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        except Exception as e:
            print(f"Error running LilyPond: {e}", file=sys.stderr)
            return [None] * len(blocks)
    out = out.replace("\r\n", "\n")
    results = [None] * len(blocks)
    for idx in range(len(blocks)):
        m = re.search(rf"===BEGIN_{idx}===\n(.*?)===END_{idx}===\n?", out, re.S)
        if not m:
            continue
        seg = m.group(1)
        seg = _unwrap_absolute_layers(seg)
        k = 0
        while k < len(seg) and seg[k].isspace():
            k += 1
        if k >= len(seg):
            results[idx] = ""
            continue
        if seg.startswith("<<", k):
            j = _grab_angles(seg, k)
            block_text = seg[k:j] if j > k else ""
        elif seg[k] == "{":
            j = _grab_braces(seg, k)
            block_text = seg[k:j] if j > k else ""
        else:
            block_text = seg.strip()
        if preserve_linebreaks:
            block_text = _normalize_line_keep_newlines(block_text)
        else:
            block_text = re.sub(r"[ \t]*\r?\n[ \t]*", " ", block_text)
            block_text = re.sub(r"[ \t]+", " ", block_text).strip()
        results[idx] = block_text
    return results

def expand_relative_with_lily_batched(source: str, lily_cmd: str, *, preserve_linebreaks: bool) -> Tuple[str, int]:
    blocks = _find_relative_blocks(source)
    if not blocks:
        return source, 0
    language = _detect_note_language(source)
    preamble = f'\\language "{language}"' if language else ""
    expansions = _run_lily_batch(
        [blk for (_, _, blk) in blocks],
        lily_cmd=lily_cmd,
        preserve_linebreaks=preserve_linebreaks,
        preamble=preamble,
    )
    out, i = [], 0
    failures = 0
    for (start, end, orig), expanded in zip(blocks, expansions):
        out.append(source[i:start])
        if expanded is None:
            out.append(orig)
            failures += 1
        else:
            out.append(expanded)
        i = end
    out.append(source[i:])
    return "".join(out), failures

RE_TRANSPOSE = re.compile(r"\\transpose\s+([^\s{}]+)\s+([^\s{}]+)\s*\{", re.I)
def _find_transpose_blocks(source: str) -> List[Tuple[int, int, str]]:
    blocks = []
    i = 0
    while True:
        m = RE_TRANSPOSE.search(source, i)
        if not m:
            break
        lb = m.end() - 1
        rb = _grab_braces(source, lb)
        blocks.append((m.start(), rb, source[m.start():rb]))
        i = rb
    return blocks
def resolve_transpose_with_lily_batched(source: str, lily_cmd: str, *, preserve_linebreaks: bool) -> Tuple[str, int, int]:
    blocks = _find_transpose_blocks(source)
    if not blocks:
        return source, 0, 0
    expansions = _run_lily_batch([blk for (_, _, blk) in blocks], lily_cmd=lily_cmd, preserve_linebreaks=preserve_linebreaks)
    out, i = [], 0
    ok, fail = 0, 0
    for (start, end, orig), expanded in zip(blocks, expansions):
        out.append(source[i:start])
        if expanded is not None:
            out.append(expanded)
            ok += 1
        else:
            out.append(orig)
            fail += 1
        i = end
    out.append(source[i:])
    return "".join(out), ok, fail

RE_REPEAT_UNFOLD = re.compile(r"\\repeat\s+unfold\s+(\d+)\s*\{", re.I)
def _expand_repeat_unfold_once(source: str) -> Tuple[str, int]:
    i = 0
    n = 0
    out = []
    last = 0
    while True:
        m = RE_REPEAT_UNFOLD.search(source, i)
        if not m:
            break
        start = m.start()
        lb = m.end() - 1
        rb = _grab_braces(source, lb)
        body = source[lb+1:rb-1]
        times = int(m.group(1))
        out.append(source[last:start])
        repeated = []
        for _ in range(times):
            repeated.append(body.strip())
        out.append(" ".join(repeated))
        n += 1
        i = rb
        last = rb
    out.append(source[last:])
    return "".join(out), n
def expand_repeat_unfold(source: str, *, max_passes: int = 8) -> Tuple[str, int]:
    total = 0
    s = source
    for _ in range(max_passes):
        s2, cnt = _expand_repeat_unfold_once(s)
        total += cnt
        if cnt == 0:
            return s2, total
        s = s2
    return s, total

RE_TIMES = re.compile(r"\\times\s+(\d+)\s*/\s*(\d+)\s*\{", re.I)
RE_TUPLET = re.compile(r"\\tuplet\s+(\d+)\s*/\s*(\d+)(?:\s+\d+)?\s*\{", re.I)
def _normalize_tuplet_spacing_block(s: str) -> str:
    s = re.sub(r"\\tuplet\s+(\d+)\s*/\s*(\d+)\s*", r"\\tuplet \1/\2 ", s)
    s = re.sub(r"\\tuplet\s+(\d+/\d+)\s*\{", r"\\tuplet \1 {", s)
    return s
def _dedupe_nested_tuplets_once(s: str) -> Tuple[str, int]:
    """
    Remove nested \\tuplet blocks that repeat the same ratio directly inside each other.
    Operates conservatively so only the transformed tuplets have their braces adjusted.
    """
    nested_pat = re.compile(
        r"(\\tuplet\s+(\d+)\s*/\s*(\d+)\s*\{)\s*(\\tuplet\s+\2\s*/\s*\3\s*\{)",
        re.I,
    )
    changed = 0
    while True:
        m = nested_pat.search(s)
        if not m:
            break
        inner_start = m.start(4)
        inner_lb = s.find("{", inner_start)
        if inner_lb == -1:
            break
        inner_rb = _grab_braces(s, inner_lb)
        if inner_rb <= inner_lb or inner_rb > len(s):
            break
        body = s[inner_lb + 1:inner_rb - 1]
        s = s[:inner_start] + body + s[inner_rb:]
        changed += 1
    return s, changed
def normalize_tuplets(source: str) -> Tuple[str, int]:
    changed = 0
    s = source
    i = 0
    out = []
    last = 0
    while True:
        m = RE_TIMES.search(s, i)
        if not m:
            break
        start = m.start()
        lb = m.end() - 1
        rb = _grab_braces(s, lb)
        body = s[lb:rb]  
        ratio = f"{m.group(1)}/{m.group(2)}"
        out.append(s[last:start])
        out.append(f"\\tuplet {ratio} {body}")
        changed += 1
        i = rb
        last = rb
    out.append(s[last:])
    s = "".join(out)
    def _kill_opt_dur(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        return f"\\tuplet {a}/{b} {{"
    s2 = re.sub(r"\\tuplet\s+(\d+)\s*/\s*(\d+)\s+\d+\s*\{", _kill_opt_dur, s)
    if s2 != s:
        diff_cnt = len(list(re.finditer(r"\\tuplet\s+\d+\s*/\s*\d+\s+\d+\s*\{", s)))
        changed += diff_cnt
        s = s2
    for _ in range(4):
        s, dedup_cnt = _dedupe_nested_tuplets_once(s)
        changed += dedup_cnt
        if dedup_cnt == 0:
            break
    s = _normalize_tuplet_spacing_block(s)
    return s, changed
RE_DRUMMODE = re.compile(r"\\drummode\s*\{", re.I)
DRUM_MAP = {
    "bd": "bd", "bassdrum": "bd", "kick": "bd",
    "sn": "sn", "snare": "sn", "snaredrum": "sn",
    "tom": "tom", "tomh": "tom", "toml": "tom", "tomhi": "tom", "tomlo": "tom",
    "ft": "ft", "floortom": "ft",
    "hh": "hh", "hihat": "hh", "hhc": "hhc", "hhclosed": "hhc", "hho": "hho", "hhopen": "hho",
    "ride": "ride", "rd": "ride",
    "crash": "crash", "cr": "crash",
    "rim": "rim", "rimshot": "rim",
    "clave": "clave",
    "cowb": "cowb", "cowbell": "cowb",
    "tamb": "tamb", "tambourine": "tamb",
    "tri": "tri", "triangle": "tri",
    "guiro": "guiro",
    "wood": "wood", "woodblock": "wood",
    "cymc": "cymc", "china": "cymc",
    "cymr": "cymr", "splash": "cymr",
}
DRUM_TOKEN = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]*)\b")
def _normalize_drums_in_block(block: str) -> Tuple[str, int]:
    def repl(m: re.Match) -> str:
        tok = m.group(1)
        low = tok.lower()
        if low in DRUM_MAP:
            return DRUM_MAP[low]
        return tok
    new = DRUM_TOKEN.sub(repl, block)
    before_tokens = DRUM_TOKEN.findall(block)
    after_tokens = DRUM_TOKEN.findall(new)
    cnt = sum(1 for a, b in zip(before_tokens, after_tokens) if a != b)
    return new, cnt
def normalize_drummode(source: str) -> Tuple[str, int]:
    i = 0
    out = []
    last = 0
    changed_blocks = 0
    while True:
        m = RE_DRUMMODE.search(source, i)
        if not m:
            break
        start = m.start()
        lb = m.end() - 1
        rb = _grab_braces(source, lb)
        body = source[lb:rb]  
        norm_body, _cnt = _normalize_drums_in_block(body)
        if norm_body != body:
            changed_blocks += 1
        out.append(source[last:start])
        out.append("\\drummode " + norm_body)
        i = rb
        last = rb
    out.append(source[last:])
    return "".join(out), changed_blocks

def expand_music_functions_with_lily(source: str, lily_cmd: str, *, preserve_linebreaks: bool) -> Tuple[str, int, int]:
    func_names = _collect_music_function_names(source)
    if not func_names:
        return source, 0, 0
    preamble = _extract_music_function_snippets(source)
    calls = _find_function_calls(source, func_names)
    if not calls:
        return source, 0, 0
    blocks = [call for (_, _, call) in calls]
    results = _run_lily_batch(blocks, lily_cmd=lily_cmd, preserve_linebreaks=preserve_linebreaks, preamble=preamble)
    out, i = [], 0
    ok, fail = 0, 0
    for (start, end, orig), expanded in zip(calls, results):
        out.append(source[i:start])
        if expanded is not None:
            out.append(expanded)
            ok += 1
        else:
            out.append(orig)
            fail += 1
        i = end
    out.append(source[i:])
    return "".join(out), ok, fail

def normalize_whitespace(source: str) -> str:
    s = source.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in lines]
    return "\n".join(lines).strip()

def process_string(src: str, lily_cmd: str, opts: ParseOptions) -> Tuple[str, ParseReport]:
    report = ParseReport(notes=[])
    s = src
    if opts.inline_variables:
        env = _collect_named_music(s)
        s2, cnt = _inline_named_music_recursive(s, env)
        report.variables_inlined = cnt
        s = s2
        debug_print("Variables inlined (recursive)", f"expansions={cnt}")
    if opts.expand_music_functions:
        s2, ok, fail = expand_music_functions_with_lily(
            s, lily_cmd=lily_cmd, preserve_linebreaks=opts.preserve_linebreaks
        )
        if ok or fail:
            s = s2
            report.lily_failures += fail
            debug_print("Music functions expanded", f"ok={ok}, fail={fail}")
    if opts.expand_relative:
        rel_count = len(_find_relative_blocks(s))
        if rel_count:
            s, rel_fail = expand_relative_with_lily_batched(s, lily_cmd=lily_cmd, preserve_linebreaks=opts.preserve_linebreaks)
            report.relative_blocks = rel_count
            report.lily_failures += rel_fail
            debug_print("Relative expanded", f"blocks={rel_count}")
    if opts.resolve_transpose:
        s2, ok, fail = resolve_transpose_with_lily_batched(s, lily_cmd=lily_cmd, preserve_linebreaks=opts.preserve_linebreaks)
        if ok or fail:
            s = s2
            report.transpose_blocks = ok
            report.lily_failures += fail
            debug_print("Transpose resolved", f"ok={ok}, fail={fail}")
    if opts.expand_repeat_unfold:
        s2, cnt = expand_repeat_unfold(s)
        if cnt:
            s = s2
            report.repeats_unfolded = cnt
            debug_print("Repeat unfold expanded", f"blocks={cnt}")
    if opts.normalize_tuplets:
        s2, cnt = normalize_tuplets(s)
        if cnt:
            s = s2
            report.tuplets_normalized = cnt
            debug_print("Tuplets normalized", f"changes~={cnt}")
    if opts.normalize_drums:
        s2, cnt_blocks = normalize_drummode(s)
        if cnt_blocks:
            s = s2
            report.drum_blocks_normalized = cnt_blocks
            debug_print("Drummode normalized", f"blocks={cnt_blocks}")
    if opts.normalize_whitespace:
        s = normalize_whitespace(s)
    if opts.canonicalize_chord_brackets:
        s = _canonicalize_chord_brackets(s)
    return s, report

try:
    from lilynorm.utils.options import NormOptions  # type: ignore
except Exception:  # pragma: no cover
    class NormOptions:  # fallback typing stub
        keep_engraving: bool = False
        strip_scheme_blocks: bool = True
        strip_comments: bool = True
        normalize_whitespace: bool = False
        expand_relative: bool = True
        inline_variables: bool = True
        expand_music_functions: bool = True
        resolve_transpose: bool = True 
        expand_repeat_unfold: bool = True
        normalize_tuplets: bool = True  
        normalize_drums: bool = True
        preserve_linebreaks: bool = True
        canonicalize_chord_brackets: bool = True

def _map_options(o: "NormOptions") -> ParseOptions:
    defaults = _DEFAULT_PARSE_OPTIONS
    values = {f.name: getattr(o, f.name, getattr(defaults, f.name)) for f in fields(ParseOptions)}
    return ParseOptions(**values)

def run(text: str, opts: "NormOptions") -> str:
    lily_cmd = resolve_lily_cmd()
    parse_opts = _map_options(opts)

    if not lily_available(lily_cmd):
        if parse_opts.expand_relative or parse_opts.expand_music_functions or parse_opts.resolve_transpose:
            print("[normalize] LilyPond not found – skipping relative/music-functions/transpose.", file=sys.stderr)
        parse_opts.expand_relative = False
        parse_opts.expand_music_functions = False
        parse_opts.resolve_transpose = False

    out, report = process_string(text, lily_cmd=lily_cmd, opts=parse_opts)

    print(
        f"[normalize] rel:{report.relative_blocks} "
        f"vars:{report.variables_inlined} "
        f"transpose_ok:{report.transpose_blocks} "
        f"repeat:{report.repeats_unfolded} "
        f"tuplets:{report.tuplets_normalized} "
        f"drums:{report.drum_blocks_normalized} "
        f"lily_fail:{report.lily_failures}"
    )
    return out

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone LilyPond lossless normalizer.")
    p.add_argument("--in", dest="inp", required=True, help="Input .ly file path")
    p.add_argument("--out", dest="out", default=None, help="Output file path (if omitted, prints to stdout)")
    p.add_argument("--lily", dest="lily", default=None, help="Path to lilypond executable (auto-detected if omitted)")
    p.add_argument("--debug", action="store_true", help="Enable debug prints to stderr")
    p.add_argument("--report", action="store_true", help="Print a JSON summary report to stderr")
    def onoff(flag: str, dest: str, default: bool):
        if default:
            p.add_argument(f"--no-{flag}", dest=dest, action="store_false", help=f"Disable {flag.replace('-', ' ')}")
        else:
            p.add_argument(f"--{flag}", dest=dest, action="store_true", help=f"Enable {flag.replace('-', ' ')}")
    onoff("expand-relative", "expand_relative", True)
    onoff("inline-variables", "inline_variables", True)
    onoff("expand-music-functions", "expand_music_functions", True)
    onoff("resolve-transpose", "resolve_transpose", True)
    onoff("expand-repeat-unfold", "expand_repeat_unfold", True)
    onoff("normalize-tuplets", "normalize_tuplets", True)
    onoff("normalize-drums", "normalize_drums", True)
    onoff("normalize-whitespace", "normalize_whitespace", False)
    onoff("preserve-linebreaks", "preserve_linebreaks", True)
    onoff("canonicalize-chord-brackets", "canonicalize_chord_brackets", True)
    return p
def main() -> int:
    global DEBUG
    ap = build_arg_parser()
    args = ap.parse_args()
    DEBUG = bool(args.debug)
    in_path = Path(args.inp)
    if not in_path.exists():
        print(f"File not found: {in_path}", file=sys.stderr)
        return 1
    lily_cmd = args.lily or resolve_lily_cmd()
    if not lily_available(lily_cmd):
        print(f"Error: LilyPond not found or not runnable: {lily_cmd}", file=sys.stderr)
        return 2
    src = in_path.read_text(encoding="utf-8", errors="ignore")
    opts = ParseOptions(
        expand_relative = args.expand_relative,
        inline_variables = args.inline_variables,
        expand_music_functions = args.expand_music_functions,
        resolve_transpose = args.resolve_transpose,
        expand_repeat_unfold = args.expand_repeat_unfold,
        normalize_tuplets = args.normalize_tuplets,
        normalize_drums = args.normalize_drums,
        normalize_whitespace = args.normalize_whitespace,
        preserve_linebreaks = args.preserve_linebreaks,
        canonicalize_chord_brackets = args.canonicalize_chord_brackets,
    )
    expanded, rep = process_string(src, lily_cmd=lily_cmd, opts=opts)
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(expanded, encoding="utf-8")
        print(f"Wrote: {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(expanded)
    if args.report:
        try:
            print("\n--- REPORT ---", file=sys.stderr)
            print(json.dumps(asdict(rep), ensure_ascii=False, indent=2), file=sys.stderr)
        except Exception as e:
            print(f"Failed to print report JSON: {e}", file=sys.stderr)
    return 0
if __name__ == "__main__":
    sys.exit(main())
