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


# Configuration

DEFAULT_LILYPOND_PATH = (
    r"C:\lilypond-2.24.4-mingw-x86_64\lilypond-2.24.4\bin\lilypond.exe"
)



# Core option types

@dataclass
class ParseOptions:
    """
    Options controlling which normalization steps are applied.
    """
    expand_relative: bool = True
    inline_variables: bool = False  # Disabled - causes syntax corruption in nested variables
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
    """
    Counters and notes describing what the normalizer did.
    """
    relative_blocks: int = 0
    variables_inlined: int = 0
    transpose_blocks: int = 0
    repeats_unfolded: int = 0
    tuplets_normalized: int = 0
    drum_blocks_normalized: int = 0
    lily_failures: int = 0
    notes: List[str] = field(default_factory=list)


# Small structural helpers (braces / angles)

def _grab_braces(s: str, start_index: int) -> int:
    depth = 1
    index = start_index + 1
    length = len(s)

    while index < length and depth > 0:
        char = s[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1

    return index if depth == 0 else length


def _grab_angles(s: str, start_index: int) -> int:
    depth = 1
    index = start_index + 2  # skip initial '<<'
    length = len(s)

    while index < length and depth > 0:
        if s.startswith("<<", index):
            depth += 1
            index += 2
        elif s.startswith(">>", index):
            depth -= 1
            index += 2
        else:
            index += 1

    return index if depth == 0 else length


# LilyPond executable resolution

def _ok(cmd: str) -> bool:
    try:
        result = subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def lily_available(lily_cmd: str) -> bool:
    """
    Check whether the given lilypond command is available.
    """
    return _ok(lily_cmd)


def resolve_lily_cmd() -> str:
    """
    Resolve a usable lilypond command.

    Search order:
      1. PATH ("lilypond").
      2. LILYPOND_BIN environment variable.
      3. DEFAULT_LILYPOND_PATH.
      4. Fallback to string "lilypond".
    """
    from_path = shutil.which("lilypond")
    if from_path and _ok(from_path):
        return from_path

    from_env = os.environ.get("LILYPOND_BIN")
    if from_env and _ok(from_env):
        return from_env

    if os.path.isfile(DEFAULT_LILYPOND_PATH) and _ok(DEFAULT_LILYPOND_PATH):
        return DEFAULT_LILYPOND_PATH

    return "lilypond"


# Relative blocks & note language detection

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

    Returns None if no guess can be made (defaults to Lily's English).
    """
    language_match = RE_LANGUAGE_DECL.search(source)
    if language_match:
        return language_match.group(1)

    for match in RE_RELATIVE_TOKEN.finditer(source):
        token = match.group(1).strip().lower()
        token = token.strip(",;'\"")
        for solfege in ITALIAN_SOLFEGE:
            if token.startswith(solfege):
                return "italiano"

    return None


def _find_relative_blocks(source: str) -> List[Tuple[int, int, str]]:
    blocks: List[Tuple[int, int, str]] = []
    search_start = 0

    while True:
        match = RE_RELATIVE_BLK.search(source, search_start)
        if not match:
            break

        brace_open_index = match.end() - 1
        brace_close_index = _grab_braces(source, brace_open_index)
        blocks.append((match.start(), brace_close_index, source[match.start():brace_close_index]))
        search_start = brace_close_index

    return blocks


# Variable assignment discovery and inline expansion

RE_ASSIGN = re.compile(r"(^|[^\w-])([A-Za-z][\w-]*)\s*=\s*", re.M)


def _collect_named_music(source: str) -> Dict[str, str]:
    environment: Dict[str, str] = {}
    search_start = 0
    length = len(source)

    while search_start < length:
        match = RE_ASSIGN.search(source, search_start)
        if not match:
            break

        rhs_start = match.end()
        while rhs_start < length and source[rhs_start].isspace():
            rhs_start += 1

        name = match.group(2)

        # Skip markup variables (start with _ or ^)
        if rhs_start < length and source[rhs_start] in ("_", "^"):
            # Skip to end of line or next assignment
            search_start = rhs_start + 1
            continue

        # Skip setting/override commands that shouldn't be inlined
        # (these are context-dependent and break when inlined into music blocks)
        skip_prefixes = ("\\override", "\\set", "\\tupletSpan", "\\revert", "\\unset")
        is_setting = any(source.startswith(prefix, rhs_start) for prefix in skip_prefixes)
        if is_setting:
            search_start = rhs_start + 1
            continue

        # name = \relative ...
        if source.startswith("\\relative", rhs_start):
            relative_match = RE_RELATIVE_BLK.search(source, rhs_start)
            if not relative_match:
                search_start = rhs_start + 1
                continue
            brace_open_index = relative_match.end() - 1
            brace_close_index = _grab_braces(source, brace_open_index)
            environment[name] = source[rhs_start:brace_close_index]
            search_start = brace_close_index

        # name = \transpose ...
        elif source.startswith("\\transpose", rhs_start):
            transpose_match = RE_TRANSPOSE.search(source, rhs_start)
            if not transpose_match:
                search_start = rhs_start + 1
                continue
            brace_open_index = transpose_match.end() - 1
            brace_close_index = _grab_braces(source, brace_open_index)
            environment[name] = source[rhs_start:brace_close_index]
            search_start = brace_close_index

        # name = { ... }
        elif rhs_start < length and source[rhs_start] == "{":
            brace_close_index = _grab_braces(source, rhs_start)
            environment[name] = source[rhs_start:brace_close_index]
            search_start = brace_close_index

        # name = << ... >>
        elif source.startswith("<<", rhs_start):
            angle_close_index = _grab_angles(source, rhs_start)
            environment[name] = source[rhs_start:angle_close_index]
            search_start = angle_close_index

        # name = simple_token (but not markup)
        else:
            token_end = rhs_start
            while token_end < length and not source[token_end].isspace():
                if source[token_end] in "{}<>":
                    break
                token_end += 1
            environment[name] = source[rhs_start:token_end]
            search_start = token_end

    return environment


def _inline_once(source: str, env: Dict[str, str]) -> Tuple[str, int]:
    if not env:
        return source, 0

    inlined_count = 0
    names = sorted(env.keys(), key=len, reverse=True)
    pattern = r"\\(" + "|".join(re.escape(name) for name in names) + r")\b"

    def replace(match: re.Match) -> str:
        nonlocal inlined_count
        name = match.group(1)
        if name in env:
            inlined_count += 1
            return env[name]
        return match.group(0)

    new_source = re.sub(pattern, replace, source)
    return new_source, inlined_count


def _inline_named_music_recursive(
    source: str,
    env: Dict[str, str],
    *,
    max_passes: int = 8,
) -> Tuple[str, int]:
    total_inlined = 0
    seen_hashes: Set[int] = set()
    current = source

    for _ in range(max_passes):
        current_hash = hash(current)
        if current_hash in seen_hashes:
            break
        seen_hashes.add(current_hash)

        new_source, count = _inline_once(current, env)
        total_inlined += count

        if count == 0:
            return new_source, total_inlined

        current = new_source

    return current, total_inlined


# Variable expansion via LilyPond (parser-based)

def _inline_named_music_with_lilypond(
    source: str,
    env: Dict[str, str],
    *,
    lily_cmd: str,
    preserve_linebreaks: bool,
) -> Tuple[str, int]:
    if not env:
        return source, 0

    # Build a preamble that defines all collected music variables.
    preamble_lines = []
    for name, rhs in env.items():
        preamble_lines.append(f"{name} = {rhs}")
    preamble = "\n".join(preamble_lines)

    # Ask LilyPond to expand each variable reference.
    names = list(env.keys())
    blocks = [f"\\{name}" for name in names]
    expanded = _run_lily_batch(
        blocks,
        lily_cmd,
        preserve_linebreaks=preserve_linebreaks,
        preamble=preamble,
    )

    # Build replacement map for successful expansions.
    repl: Dict[str, str] = {}
    for name, value in zip(names, expanded):
        if value and _is_safe_music_expansion(value):
            repl[name] = value

    if not repl:
        return source, 0

    # Replace occurrences of \name with expanded music.
    names_sorted = sorted(repl.keys(), key=len, reverse=True)
    pattern = r"\\(" + "|".join(re.escape(name) for name in names_sorted) + r")\b"
    count = 0

    def _replace(match: re.Match) -> str:
        nonlocal count
        name = match.group(1)
        if name in repl:
            count += 1
            return repl[name]
        return match.group(0)

    updated = re.sub(pattern, _replace, source)
    return updated, count


def _is_safe_music_expansion(text: str) -> bool:
    if not text:
        return False

    forbidden = (
        "\\markup",
        "\\paper",
        "\\header",
        "\\layout",
        "\\score",
        "\\context",
        "\\set",
        "\\override",
        "\\revert",
        "\\lyricmode",
        "#(",
    )
    if any(tok in text for tok in forbidden):
        return False

    # Require at least one note/rest token.
    note_re = re.compile(r"\b(?:do|re|mi|fa|sol|la|si|[a-g]|r)[',#isbf]*\d", re.I)
    if not note_re.search(text):
        return False

    # Reject empty brace placeholders that can appear in broken output.
    if "{}" in text or "{ }" in text:
        return False

    return True


# Music functions (define-music-function, calls, and snippets)

RE_DMF_HEADER = re.compile(
    r"(^|[\n;])\s*([A-Za-z][\w-]*)\s*=\s*#\(\s*define-music-function\b",
    re.S,
)


def _collect_music_function_names(source: str) -> List[str]:
    names: List[str] = []
    for match in RE_DMF_HEADER.finditer(source):
        names.append(match.group(2))
    return sorted(set(names), key=len, reverse=True)


def _extract_music_function_snippets(source: str) -> str:
    preamble_parts: List[str] = []

    for match in RE_DMF_HEADER.finditer(source):
        name_start = match.start(2)
        eq_index = source.find("=", name_start)
        if eq_index == -1:
            continue

        search_from = eq_index + 1
        open_hash_index = source.find("#{", search_from)
        if open_hash_index == -1:
            continue

        close_hash_index = source.find("#}", open_hash_index + 2)
        if close_hash_index == -1:
            continue

        close_paren_index = source.find(")", close_hash_index + 2)
        if close_paren_index == -1:
            continue

        snippet = source[name_start:close_paren_index + 1]
        preamble_parts.append(snippet)

    return "\n".join(preamble_parts)


def _find_function_calls(
    source: str,
    func_names: List[str],
) -> List[Tuple[int, int, str]]:
    if not func_names:
        return []

    name_pattern = r"\\(?:" + "|".join(re.escape(name) for name in func_names) + r")\b"
    call_regex = re.compile(name_pattern)

    calls: List[Tuple[int, int, str]] = []
    search_start = 0
    length = len(source)

    while True:
        match = call_regex.search(source, search_start)
        if not match:
            break

        start_index = match.start()
        token_end = match.end()

        while token_end < length and source[token_end].isspace():
            token_end += 1

        if token_end >= length:
            search_start = token_end
            continue

        if source.startswith("{", token_end):
            end_index = _grab_braces(source, token_end)
        elif source.startswith("<<", token_end):
            end_index = _grab_angles(source, token_end)
        elif source.startswith("<", token_end):
            # Simple < ... > chord
            chord_index = token_end + 1
            while chord_index < length and source[chord_index] != ">":
                chord_index += 1
            end_index = chord_index + 1 if chord_index < length else length
        else:
            # Single token call
            token_scan = token_end
            while (
                token_scan < length
                and not source[token_scan].isspace()
                and source[token_scan] not in "{}<>"
            ):
                token_scan += 1
            end_index = token_scan

        calls.append((start_index, end_index, source[start_index:end_index]))
        search_start = end_index

    return calls


# Formatting helpers (whitespace, chords, absolute wrappers)

def _normalize_line_keep_newlines(segment: str) -> str:
    segment = segment.replace("\r\n", "\n").replace("\r", "\n")
    lines = segment.split("\n")
    normalized_lines: List[str] = []

    for line in lines:
        line = re.sub(r"[ \t]+", " ", line.strip())
        line = re.sub(r"[ ]*\{[ ]*", " { ", line)
        line = re.sub(r"[ ]*\}[ ]*", " } ", line)
        line = re.sub(r"[ ]*<<[ ]*", " << ", line)
        line = re.sub(r"[ ]*>>[ ]*", " >> ", line)
        line = re.sub(r"[ ]{2,}", " ", line)
        normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


_CHORD_RE = re.compile(r"<([^>]*)>")


def _canonicalize_chord_brackets(text: str) -> str:
    def _fix(match: re.Match) -> str:
        inner = match.group(1)
        inner = " ".join(inner.split())
        return f"<{inner}>"

    text = _CHORD_RE.sub(_fix, text)
    text = re.sub(r"<\s+", "<", text)
    text = re.sub(r"\s+>", ">", text)
    return text


def _unwrap_absolute_layers(segment: str) -> str:
    segment = segment.strip()

    while segment.startswith("\\absolute"):
        brace_index = segment.find("{")
        if brace_index == -1:
            break
        brace_close_index = _grab_braces(segment, brace_index)
        segment = segment[brace_index + 1:brace_close_index - 1].strip()

    return segment


# LilyPond batch runner for multiple blocks

def _run_lily_batch(
    blocks: List[str],
    lily_cmd: str,
    *,
    preserve_linebreaks: bool,
    preamble: str = "",
) -> List[Optional[str]]:
    parts: List[str] = []
    if preamble:
        parts.append(preamble)

    def var_name(idx: int) -> str:
        """
        Produce a deterministic variable name: music[a..z, aa..zz, ...].
        """

        def _letters(n: int) -> str:
            if n < 0:
                return "a"
            chars: List[str] = []
            while True:
                chars.append(chr(ord("a") + (n % 26)))
                n = n // 26 - 1
                if n < 0:
                    break
            return "".join(reversed(chars))

        return f"music{_letters(idx)}"

    var_names = [var_name(i) for i in range(len(blocks))]

    # Assign blocks
    for idx, block in enumerate(blocks):
        parts.append(f"{var_names[idx]} = \\absolute {{ {block} }}")

    # Dump normalized Lily music
    for idx in range(len(blocks)):
        parts.append(f"#(display \"===BEGIN_{idx}===\\n\")")
        parts.append(f"\\displayLilyMusic \\{var_names[idx]}")
        parts.append(f"#(display \"===END_{idx}===\\n\")")

    ly_source = "\n".join(parts) + "\n"
    result_output = ""

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir, "input.ly")
        tmp_path.write_text(ly_source, encoding="utf-8")

        try:
            proc = subprocess.run(
                [
                    lily_cmd,
                    "-dno-print-pages",
                    "-dbackend=null",
                    "-o",
                    str(Path(temp_dir, "dump")),
                    str(tmp_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                encoding="utf-8",
            )
            result_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        except Exception as exc:
            print(f"Error running LilyPond: {exc}", file=sys.stderr)
            return [None] * len(blocks)

    result_output = result_output.replace("\r\n", "\n")
    results: List[Optional[str]] = [None] * len(blocks)

    for idx in range(len(blocks)):
        match = re.search(
            rf"===BEGIN_{idx}===\n(.*?)===END_{idx}===\n?",
            result_output,
            re.S,
        )
        if not match:
            continue

        segment = match.group(1)
        segment = _unwrap_absolute_layers(segment)

        segment_start = 0
        while segment_start < len(segment) and segment[segment_start].isspace():
            segment_start += 1

        if segment_start >= len(segment):
            results[idx] = ""
            continue

        if segment.startswith("<<", segment_start):
            block_end = _grab_angles(segment, segment_start)
            block_text = segment[segment_start:block_end] if block_end > segment_start else ""
        elif segment[segment_start] == "{":
            block_end = _grab_braces(segment, segment_start)
            block_text = segment[segment_start:block_end] if block_end > segment_start else ""
        else:
            block_text = segment.strip()

        if preserve_linebreaks:
            block_text = _normalize_line_keep_newlines(block_text)
        else:
            block_text = re.sub(r"[ \t]*\r?\n[ \t]*", " ", block_text)
            block_text = re.sub(r"[ \t]+", " ", block_text).strip()

        # Check for invalid/empty output
        if block_text in ("", "## { # }"):
            results[idx] = None
        else:
            results[idx] = block_text

    return results


# Relative expansion via LilyPond

def expand_relative_with_lily_batched(
    source: str,
    lily_cmd: str,
    *,
    preserve_linebreaks: bool,
) -> Tuple[str, int]:
    """
    Expand all \\relative blocks in source using LilyPond (batch mode).

    Returns (new_source, lily_failures_count).
    """
    blocks = _find_relative_blocks(source)
    if not blocks:
        return source, 0

    language = _detect_note_language(source)
    preamble = f'\\language "{language}"' if language else ""

    expansions = _run_lily_batch(
        [block_text for (_, _, block_text) in blocks],
        lily_cmd=lily_cmd,
        preserve_linebreaks=preserve_linebreaks,
        preamble=preamble,
    )

    output_parts: List[str] = []
    cursor = 0
    failures = 0

    for (start, end, original), expanded in zip(blocks, expansions):
        output_parts.append(source[cursor:start])
        if expanded is None:
            output_parts.append(original)
            failures += 1
        else:
            output_parts.append(expanded)
        cursor = end

    output_parts.append(source[cursor:])
    return "".join(output_parts), failures


# Transpose block resolution via LilyPond

RE_TRANSPOSE = re.compile(
    r"\\transpose\s+([^\s{}]+)\s+([^\s{}]+)\s*\{",
    re.I,
)


def _find_transpose_blocks(source: str) -> List[Tuple[int, int, str]]:
    blocks: List[Tuple[int, int, str]] = []
    search_start = 0

    while True:
        match = RE_TRANSPOSE.search(source, search_start)
        if not match:
            break

        brace_open_index = match.end() - 1
        brace_close_index = _grab_braces(source, brace_open_index)
        blocks.append((match.start(), brace_close_index, source[match.start():brace_close_index]))
        search_start = brace_close_index

    return blocks


def resolve_transpose_with_lily_batched(
    source: str,
    lily_cmd: str,
    *,
    preserve_linebreaks: bool,
) -> Tuple[str, int, int]:
    """
    Resolve \\transpose blocks by letting LilyPond expand them.

    Returns (new_source, ok_count, fail_count).
    """
    blocks = _find_transpose_blocks(source)
    if not blocks:
        return source, 0, 0

    language = _detect_note_language(source)
    preamble = f'\\language "{language}"' if language else ""

    expansions = _run_lily_batch(
        [block_text for (_, _, block_text) in blocks],
        lily_cmd=lily_cmd,
        preserve_linebreaks=preserve_linebreaks,
        preamble=preamble,
    )

    output_parts: List[str] = []
    cursor = 0
    ok_count = 0
    fail_count = 0

    for (start, end, original), expanded in zip(blocks, expansions):
        output_parts.append(source[cursor:start])
        if expanded is not None:
            output_parts.append(expanded)
            ok_count += 1
        else:
            output_parts.append(original)
            fail_count += 1
        cursor = end

    output_parts.append(source[cursor:])
    return "".join(output_parts), ok_count, fail_count


# \\repeat unfold expansion (pure string manipulation)

RE_REPEAT_UNFOLD = re.compile(r"\\repeat\s+unfold\s+(\d+)\s*\{", re.I)


def _expand_repeat_unfold_once(source: str) -> Tuple[str, int]:
    search_start = 0
    expanded_blocks = 0
    output_parts: List[str] = []
    last_index = 0

    while True:
        match = RE_REPEAT_UNFOLD.search(source, search_start)
        if not match:
            break

        start = match.start()
        brace_open_index = match.end() - 1
        brace_close_index = _grab_braces(source, brace_open_index)
        body = source[brace_open_index + 1:brace_close_index - 1]
        times = int(match.group(1))

        output_parts.append(source[last_index:start])
        repeated_body: List[str] = []
        for _ in range(times):
            repeated_body.append(body.strip())

        output_parts.append(" ".join(repeated_body))
        expanded_blocks += 1

        search_start = brace_close_index
        last_index = brace_close_index

    output_parts.append(source[last_index:])
    return "".join(output_parts), expanded_blocks


def expand_repeat_unfold(
    source: str,
    *,
    max_passes: int = 8,
) -> Tuple[str, int]:
    """
    Recursively expand \\repeat unfold blocks up to max_passes.

    Returns (new_source, total_blocks_expanded).
    """
    total_expanded = 0
    current = source

    for _ in range(max_passes):
        new_source, count = _expand_repeat_unfold_once(current)
        total_expanded += count
        if count == 0:
            return new_source, total_expanded
        current = new_source

    return current, total_expanded


# Tuplet normalization (\\times -> \\tuplet, spacing, dedupe)

RE_TIMES = re.compile(r"\\times\s+(\d+)\s*/\s*(\d+)\s*\{", re.I)
RE_TUPLET = re.compile(r"\\tuplet\s+(\d+)\s*/\s*(\d+)(?:\s+\d+)?\s*\{", re.I)


def _normalize_tuplet_spacing_block(text: str) -> str:
    text = re.sub(r"\\tuplet\s+(\d+)\s*/\s*(\d+)\s*", r"\\tuplet \1/\2 ", text)
    text = re.sub(r"\\tuplet\s+(\d+/\d+)\s*\{", r"\\tuplet \1 {", text)
    return text


def _dedupe_nested_tuplets_once(text: str) -> Tuple[str, int]:
    nested_pattern = re.compile(
        r"(\\tuplet\s+(\d+)\s*/\s*(\d+)\s*\{)\s*(\\tuplet\s+\2\s*/\s*\3\s*\{)",
        re.I,
    )
    changed = 0

    while True:
        match = nested_pattern.search(text)
        if not match:
            break

        inner_start = match.start(4)
        inner_lb = text.find("{", inner_start)
        if inner_lb == -1:
            break

        inner_rb = _grab_braces(text, inner_lb)
        if inner_rb <= inner_lb or inner_rb > len(text):
            break

        body = text[inner_lb + 1:inner_rb - 1]
        text = text[:inner_start] + body + text[inner_rb:]
        changed += 1

    return text, changed


def normalize_tuplets(source: str) -> Tuple[str, int]:
    """
    Normalize tuplets:

      - Convert \\times ratios to \\tuplet.
      - Remove optional tuplet duration, when possible.
      - Remove directly nested redundant tuplets.
      - Normalize tuplet spacing.

    Returns (new_source, estimated_change_count).
    """
    total_changes = 0
    text = source

    # Convert \\times -> \\tuplet
    search_start = 0
    output_parts: List[str] = []
    last_index = 0

    while True:
        match = RE_TIMES.search(text, search_start)
        if not match:
            break

        start = match.start()
        brace_open_index = match.end() - 1
        brace_close_index = _grab_braces(text, brace_open_index)
        body = text[brace_open_index:brace_close_index]
        ratio = f"{match.group(1)}/{match.group(2)}"

        output_parts.append(text[last_index:start])
        output_parts.append(f"\\tuplet {ratio} {body}")
        total_changes += 1

        search_start = brace_close_index
        last_index = brace_close_index

    output_parts.append(text[last_index:])
    text = "".join(output_parts)

    # Remove optional tuplet duration: \\tuplet 3/2 8 { ... } -> \\tuplet 3/2 {
    def _kill_opt_dur(match: re.Match) -> str:
        a, b = match.group(1), match.group(2)
        return f"\\tuplet {a}/{b} {{"

    text2 = re.sub(r"\\tuplet\s+(\d+)\s*/\s*(\d+)\s+\d+\s*\{", _kill_opt_dur, text)
    if text2 != text:
        diff_count = len(list(re.finditer(r"\\tuplet\s+\d+\s*/\s*\d+\s+\d+\s*\{", text)))
        total_changes += diff_count
        text = text2

    # Collapse nested repeated tuplets
    for _ in range(4):
        text, dedup_count = _dedupe_nested_tuplets_once(text)
        total_changes += dedup_count
        if dedup_count == 0:
            break

    text = _normalize_tuplet_spacing_block(text)
    return text, total_changes


# Drummode normalization

RE_DRUMMODE = re.compile(r"\\drummode\s*\{", re.I)

DRUM_MAP: Dict[str, str] = {
    "bd": "bd",
    "bassdrum": "bd",
    "kick": "bd",
    "sn": "sn",
    "snare": "sn",
    "snaredrum": "sn",
    "tom": "tom",
    "tomh": "tom",
    "toml": "tom",
    "tomhi": "tom",
    "tomlo": "tom",
    "ft": "ft",
    "floortom": "ft",
    "hh": "hh",
    "hihat": "hh",
    "hhc": "hhc",
    "hhclosed": "hhc",
    "hho": "hho",
    "hhopen": "hho",
    "ride": "ride",
    "rd": "ride",
    "crash": "crash",
    "cr": "crash",
    "rim": "rim",
    "rimshot": "rim",
    "clave": "clave",
    "cowb": "cowb",
    "cowbell": "cowb",
    "tamb": "tamb",
    "tambourine": "tamb",
    "tri": "tri",
    "triangle": "tri",
    "guiro": "guiro",
    "wood": "wood",
    "woodblock": "wood",
    "cymc": "cymc",
    "china": "cymc",
    "cymr": "cymr",
    "splash": "cymr",
}

DRUM_TOKEN = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]*)\b")


def _normalize_drums_in_block(block: str) -> Tuple[str, int]:
    def repl(match: re.Match) -> str:
        token = match.group(1)
        lower = token.lower()
        if lower in DRUM_MAP:
            return DRUM_MAP[lower]
        return token

    new_block = DRUM_TOKEN.sub(repl, block)

    before_tokens = DRUM_TOKEN.findall(block)
    after_tokens = DRUM_TOKEN.findall(new_block)
    changes = sum(1 for before, after in zip(before_tokens, after_tokens) if before != after)

    return new_block, changes


def normalize_drummode(source: str) -> Tuple[str, int]:
    """
    Normalize \\drummode {...} notation to the canonical drum token set.

    Returns (new_source, changed_block_count).
    """
    search_start = 0
    output_parts: List[str] = []
    last_index = 0
    changed_blocks = 0

    while True:
        match = RE_DRUMMODE.search(source, search_start)
        if not match:
            break

        start = match.start()
        brace_open_index = match.end() - 1
        brace_close_index = _grab_braces(source, brace_open_index)
        body = source[brace_open_index:brace_close_index]

        normalized_body, changes = _normalize_drums_in_block(body)
        if normalized_body != body:
            changed_blocks += 1

        output_parts.append(source[last_index:start])
        output_parts.append("\\drummode " + normalized_body)

        search_start = brace_close_index
        last_index = brace_close_index

    output_parts.append(source[last_index:])
    return "".join(output_parts), changed_blocks


# Music function expansion via LilyPond

def expand_music_functions_with_lily(
    source: str,
    lily_cmd: str,
    *,
    preserve_linebreaks: bool,
) -> Tuple[str, int, int]:
    """
    Expand calls to define-music-function-defined functions via LilyPond.

    Returns (new_source, ok_count, fail_count).
    """
    func_names = _collect_music_function_names(source)
    if not func_names:
        return source, 0, 0

    preamble = _extract_music_function_snippets(source)
    calls = _find_function_calls(source, func_names)
    if not calls:
        return source, 0, 0

    block_texts = [call_text for (_, _, call_text) in calls]
    results = _run_lily_batch(
        block_texts,
        lily_cmd=lily_cmd,
        preserve_linebreaks=preserve_linebreaks,
        preamble=preamble,
    )

    output_parts: List[str] = []
    cursor = 0
    ok_count = 0
    fail_count = 0

    for (start, end, original), expanded in zip(calls, results):
        output_parts.append(source[cursor:start])
        if expanded is not None:
            output_parts.append(expanded)
            ok_count += 1
        else:
            output_parts.append(original)
            fail_count += 1
        cursor = end

    output_parts.append(source[cursor:])
    return "".join(output_parts), ok_count, fail_count


# Whitespace normalization

def normalize_whitespace(source: str) -> str:
    """
    Normalize whitespace:

      - Convert CRLF / CR to LF.
      - Collapse runs of spaces/tabs to single spaces.
      - Trim leading/trailing whitespace on each line.
      - Strip leading/trailing whitespace overall.
    """
    text = source.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    return "\n".join(lines).strip()


# Main processing pipeline

def process_string(
    src: str,
    lily_cmd: str,
    opts: ParseOptions,
) -> Tuple[str, ParseReport]:
    """
    Process one LilyPond string according to ParseOptions.

    Returns (normalized_string, ParseReport).
    """
    report = ParseReport(notes=[])
    text = src

    # Inline variables
    if opts.inline_variables:
        env = _collect_named_music(text)
        if lily_available(lily_cmd):
            text_after_inline, count = _inline_named_music_with_lilypond(
                text,
                env,
                lily_cmd=lily_cmd,
                preserve_linebreaks=opts.preserve_linebreaks,
            )
        else:
            text_after_inline, count = _inline_named_music_recursive(text, env)
        report.variables_inlined = count
        text = text_after_inline

    # Expand music functions
    if opts.expand_music_functions:
        text_after_functions, ok_count, fail_count = expand_music_functions_with_lily(
            text,
            lily_cmd=lily_cmd,
            preserve_linebreaks=opts.preserve_linebreaks,
        )
        if ok_count or fail_count:
            text = text_after_functions
            report.lily_failures += fail_count

    # Expand relative
    if opts.expand_relative:
        relative_count = len(_find_relative_blocks(text))
        if relative_count:
            text, rel_failures = expand_relative_with_lily_batched(
                text,
                lily_cmd=lily_cmd,
                preserve_linebreaks=opts.preserve_linebreaks,
            )
            report.relative_blocks = relative_count
            report.lily_failures += rel_failures

    # Resolve transpose
    if opts.resolve_transpose:
        text_after_transpose, ok_count, fail_count = resolve_transpose_with_lily_batched(
            text,
            lily_cmd=lily_cmd,
            preserve_linebreaks=opts.preserve_linebreaks,
        )
        if ok_count or fail_count:
            text = text_after_transpose
            report.transpose_blocks = ok_count
            report.lily_failures += fail_count

    # Expand repeat unfold
    if opts.expand_repeat_unfold:
        text_after_repeat, count = expand_repeat_unfold(text)
        if count:
            text = text_after_repeat
            report.repeats_unfolded = count

    # Normalize tuplets
    if opts.normalize_tuplets:
        text_after_tuplets, count = normalize_tuplets(text)
        if count:
            text = text_after_tuplets
            report.tuplets_normalized = count

    # Normalize drummode
    if opts.normalize_drums:
        text_after_drums, changed_blocks = normalize_drummode(text)
        if changed_blocks:
            text = text_after_drums
            report.drum_blocks_normalized = changed_blocks

    # Whitespace & chord bracket canonicalization
    if opts.normalize_whitespace:
        text = normalize_whitespace(text)

    if opts.canonicalize_chord_brackets:
        text = _canonicalize_chord_brackets(text)

    return text, report


# Integration with lilynorm NormOptions

try:
    from lilynorm.utils.options import NormOptions
except Exception:
    class NormOptions:  # type: ignore[override]
        """
        Fallback NormOptions when lilynorm.utils.options is unavailable.

        Only mirrors the attributes used in this module.
        """
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


def _map_options(norm_opts: "NormOptions") -> ParseOptions:
    defaults = _DEFAULT_PARSE_OPTIONS
    values = {
        field_def.name: getattr(norm_opts, field_def.name, getattr(defaults, field_def.name))
        for field_def in fields(ParseOptions)
    }

    # DISABLED: inline_variables causes issues with markup and setting variables
    # The selective filtering in _collect_named_music helps but doesn't solve all cases
    # Better to let expansion fail gracefully than corrupt the output
    parse_opts = ParseOptions(**values)

    return parse_opts


def run(text: str, opts: "NormOptions") -> str:
    """
    Entry point used by the lilynorm pipeline.
    """
    lily_cmd = resolve_lily_cmd()
    parse_opts = _map_options(opts)

    if not lily_available(lily_cmd):
        if (
            parse_opts.expand_relative
            or parse_opts.expand_music_functions
            or parse_opts.resolve_transpose
        ):
            print(
                "[normalize] LilyPond not found – skipping relative/music-functions/transpose.",
                file=sys.stderr,
            )
        parse_opts.expand_relative = False
        parse_opts.expand_music_functions = False
        parse_opts.resolve_transpose = False

    output, report = process_string(text, lily_cmd=lily_cmd, opts=parse_opts)

    print(
        f"[normalize] rel:{report.relative_blocks} "
        f"vars:{report.variables_inlined} "
        f"transpose_ok:{report.transpose_blocks} "
        f"repeat:{report.repeats_unfolded} "
        f"tuplets:{report.tuplets_normalized} "
        f"drums:{report.drum_blocks_normalized} "
        f"lily_fail:{report.lily_failures}"
    )
    return output


# CLI

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for the standalone normalizer.
    """
    parser = argparse.ArgumentParser(
        description="Standalone LilyPond lossless normalizer.",
    )

    parser.add_argument("--in", dest="inp", required=True, help="Input .ly file path")
    parser.add_argument(
        "--out",
        dest="out",
        default=None,
        help="Output file path (if omitted, prints to stdout)",
    )
    parser.add_argument(
        "--lily",
        dest="lily",
        default=None,
        help="Path to lilypond executable (auto-detected if omitted)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a JSON summary report to stderr",
    )

    def add_onoff(flag: str, dest: str, default: bool) -> None:
        """
        Add --flag / --no-flag style options based on default value.
        """
        if default:
            parser.add_argument(
                f"--no-{flag}",
                dest=dest,
                action="store_false",
                help=f"Disable {flag.replace('-', ' ')}",
            )
        else:
            parser.add_argument(
                f"--{flag}",
                dest=dest,
                action="store_true",
                help=f"Enable {flag.replace('-', ' ')}",
            )

    add_onoff("expand-relative", "expand_relative", True)
    add_onoff("inline-variables", "inline_variables", True)
    add_onoff("expand-music-functions", "expand_music_functions", True)
    add_onoff("resolve-transpose", "resolve_transpose", True)
    add_onoff("expand-repeat-unfold", "expand_repeat_unfold", True)
    add_onoff("normalize-tuplets", "normalize_tuplets", True)
    add_onoff("normalize-drums", "normalize_drums", True)
    add_onoff("normalize-whitespace", "normalize_whitespace", False)
    add_onoff("preserve-linebreaks", "preserve_linebreaks", True)
    add_onoff("canonicalize-chord-brackets", "canonicalize_chord_brackets", True)

    return parser


def main() -> int:
    """
    Command-line entry point.

    Returns an exit code:
      0 on success
      1 on missing input file
      2 if LilyPond is not available
    """

    parser = build_arg_parser()
    args = parser.parse_args()


    in_path = Path(args.inp)
    if not in_path.exists():
        print(f"File not found: {in_path}", file=sys.stderr)
        return 1

    lily_cmd = args.lily or resolve_lily_cmd()
    if not lily_available(lily_cmd):
        print(
            f"Error: LilyPond not found or not runnable: {lily_cmd}",
            file=sys.stderr,
        )
        return 2

    src = in_path.read_text(encoding="utf-8", errors="ignore")

    cli_opts = ParseOptions(
        expand_relative=args.expand_relative,
        inline_variables=args.inline_variables,
        expand_music_functions=args.expand_music_functions,
        resolve_transpose=args.resolve_transpose,
        expand_repeat_unfold=args.expand_repeat_unfold,
        normalize_tuplets=args.normalize_tuplets,
        normalize_drums=args.normalize_drums,
        normalize_whitespace=args.normalize_whitespace,
        preserve_linebreaks=args.preserve_linebreaks,
        canonicalize_chord_brackets=args.canonicalize_chord_brackets,
    )

    expanded, report = process_string(src, lily_cmd=lily_cmd, opts=cli_opts)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(expanded, encoding="utf-8")
        print(f"Wrote: {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(expanded)

    if args.report:
        try:
            print("\n--- REPORT ---", file=sys.stderr)
            print(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"Failed to print report JSON: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
