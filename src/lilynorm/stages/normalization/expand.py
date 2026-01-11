import re
import sys
import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field, fields
import argparse

DEFAULT_LILYPOND_PATH = r"C:\lilypond-2.24.4-mingw-x86_64\lilypond-2.24.4\bin\lilypond.exe"


@dataclass
class ParseOptions:
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
    transpose_blocks: int = 0
    repeats_unfolded: int = 0
    tuplets_normalized: int = 0
    drum_blocks_normalized: int = 0
    lily_failures: int = 0
    notes: list[str] = field(default_factory=list)


def _grab_braces(s: str, start_index: int) -> int:
    depth = 1
    for i in range(start_index + 1, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(s)


def _grab_angles(s: str, start_index: int) -> int:
    depth = 1
    i = start_index + 2
    while i < len(s) and depth > 0:
        if s.startswith("<<", i):
            depth += 1
            i += 2
        elif s.startswith(">>", i):
            depth -= 1
            i += 2
        else:
            i += 1
    return i if depth == 0 else len(s)


def lily_available(lily_cmd: str) -> bool:
    try:
        result = subprocess.run([lily_cmd, "--version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def resolve_lily_cmd() -> str:
    for candidate in [shutil.which("lilypond"), os.environ.get("LILYPOND_BIN"), DEFAULT_LILYPOND_PATH]:
        if candidate and lily_available(candidate):
            return candidate
    return "lilypond"


RE_LANGUAGE_DECL = re.compile(r"\\language\s+\"([^\"]+)\"", re.I)


def _detect_note_language(source: str) -> Optional[str]:
    language_match = RE_LANGUAGE_DECL.search(source)
    if language_match:
        return language_match.group(1)
    return None


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

    note_re = re.compile(r"\b(?:do|re|mi|fa|sol|la|si|[a-g]|r)[',#isbf]*\d", re.I)
    if not note_re.search(text):
        return False

    if "{}" in text or "{ }" in text:
        return False

    return True


RE_DMF_HEADER = re.compile(
    r"(^|[\n;])\s*([A-Za-z][\w-]*)\s*=\s*#\(\s*define-music-function\b",
    re.S,
)


def _collect_music_function_names(source: str) -> list[str]:
    return sorted(set(m.group(2) for m in RE_DMF_HEADER.finditer(source)), key=len, reverse=True)


def _extract_music_function_snippets(source: str) -> str:
    parts = []
    for m in RE_DMF_HEADER.finditer(source):
        start = m.start(2)
        eq = source.find("=", start)
        if eq == -1:
            continue
        open_hash = source.find("#{", eq + 1)
        if open_hash == -1:
            continue
        close_hash = source.find("#}", open_hash + 2)
        if close_hash == -1:
            continue
        close_paren = source.find(")", close_hash + 2)
        if close_paren == -1:
            continue
        parts.append(source[start:close_paren + 1])
    return "\n".join(parts)


def _find_function_calls(source: str, func_names: list[str]) -> list[tuple[int, int, str]]:
    if not func_names:
        return []

    pattern = r"\\(?:" + "|".join(re.escape(n) for n in func_names) + r")\b"
    regex = re.compile(pattern)
    calls = []
    i = 0
    n = len(source)

    while True:
        match = regex.search(source, i)
        if not match:
            break

        start = match.start()
        end = match.end()

        while end < n and source[end].isspace():
            end += 1

        if end >= n:
            i = end
            continue

        if source.startswith("{", end):
            end = _grab_braces(source, end)
        elif source.startswith("<<", end):
            end = _grab_angles(source, end)
        elif source.startswith("<", end):
            j = end + 1
            while j < n and source[j] != ">":
                j += 1
            end = j + 1 if j < n else n
        else:
            while end < n and not source[end].isspace() and source[end] not in "{}<>":
                end += 1

        calls.append((start, end, source[start:end]))
        i = end

    return calls


def _normalize_line_keep_newlines(segment: str) -> str:
    segment = segment.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in segment.split("\n"):
        line = re.sub(r"[ \t]+", " ", line.strip())
        line = re.sub(r"[ ]*\{[ ]*", " { ", line)
        line = re.sub(r"[ ]*\}[ ]*", " } ", line)
        line = re.sub(r"[ ]*<<[ ]*", " << ", line)
        line = re.sub(r"[ ]*>>[ ]*", " >> ", line)
        line = re.sub(r"[ ]{2,}", " ", line)
        lines.append(line)
    return "\n".join(lines).strip()


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


def _run_lily_batch(
    blocks: list[str],
    lily_cmd: str,
    *,
    preserve_linebreaks: bool,
    preamble: str = "",
) -> list[Optional[str]]:
    parts: list[str] = []
    if preamble:
        parts.append(preamble)

    def var_name(idx: int) -> str:

        def _letters(n: int) -> str:
            if n < 0:
                return "a"
            chars: list[str] = []
            while True:
                chars.append(chr(ord("a") + (n % 26)))
                n = n // 26 - 1
                if n < 0:
                    break
            return "".join(reversed(chars))

        return f"music{_letters(idx)}"

    var_names = [var_name(i) for i in range(len(blocks))]

    for idx, block in enumerate(blocks):
        parts.append(f"{var_names[idx]} = \\absolute {{ {block} }}")

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
    results: list[Optional[str]] = [None] * len(blocks)

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

        if block_text in ("", "## { # }"):
            results[idx] = None
        else:
            results[idx] = block_text

    return results


RE_TRANSPOSE = re.compile(
    r"\\transpose\s+([^\s{}]+)\s+([^\s{}]+)\s*\{",
    re.I,
)


def _find_transpose_blocks(source: str) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
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
) -> tuple[str, int, int]:
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

    output_parts: list[str] = []
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



RE_REPEAT_UNFOLD = re.compile(r"\\repeat\s+unfold\s+(\d+)\s*\{", re.I)


def _expand_repeat_unfold_once(source: str) -> tuple[str, int]:
    search_start = 0
    expanded_blocks = 0
    output_parts: list[str] = []
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
        repeated_body: list[str] = []
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
) -> tuple[str, int]:
    total_expanded = 0
    current = source

    for _ in range(max_passes):
        new_source, count = _expand_repeat_unfold_once(current)
        total_expanded += count
        if count == 0:
            return new_source, total_expanded
        current = new_source

    return current, total_expanded


RE_TIMES = re.compile(r"\\times\s+(\d+)\s*/\s*(\d+)\s*\{", re.I)
def _normalize_tuplet_spacing_block(text: str) -> str:
    text = re.sub(r"\\tuplet\s+(\d+)\s*/\s*(\d+)\s*", r"\\tuplet \1/\2 ", text)
    text = re.sub(r"\\tuplet\s+(\d+/\d+)\s*\{", r"\\tuplet \1 {", text)
    return text


def _dedupe_nested_tuplets_once(text: str) -> tuple[str, int]:
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


def normalize_tuplets(source: str) -> tuple[str, int]:
    total_changes = 0
    text = source

    search_start = 0
    output_parts: list[str] = []
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

    def _kill_opt_dur(match: re.Match) -> str:
        a, b = match.group(1), match.group(2)
        return f"\\tuplet {a}/{b} {{"

    text2 = re.sub(r"\\tuplet\s+(\d+)\s*/\s*(\d+)\s+\d+\s*\{", _kill_opt_dur, text)
    if text2 != text:
        diff_count = len(list(re.finditer(r"\\tuplet\s+\d+\s*/\s*\d+\s+\d+\s*\{", text)))
        total_changes += diff_count
        text = text2

    for _ in range(4):
        text, dedup_count = _dedupe_nested_tuplets_once(text)
        total_changes += dedup_count
        if dedup_count == 0:
            break

    text = _normalize_tuplet_spacing_block(text)
    return text, total_changes


RE_DRUMMODE = re.compile(r"\\drummode\s*\{", re.I)

DRUM_MAP: dict[str, str] = {
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


def _normalize_drums_in_block(block: str) -> tuple[str, int]:
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


def normalize_drummode(source: str) -> tuple[str, int]:
    search_start = 0
    output_parts: list[str] = []
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


def expand_music_functions_with_lily(
    source: str,
    lily_cmd: str,
    *,
    preserve_linebreaks: bool,
) -> tuple[str, int, int]:
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

    output_parts: list[str] = []
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


def normalize_whitespace(source: str) -> str:
    text = source.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    return "\n".join(lines).strip()


def process_string(
    src: str,
    lily_cmd: str,
    opts: ParseOptions,
) -> tuple[str, ParseReport]:
    report = ParseReport(notes=[])
    text = src

    if opts.expand_music_functions:
        text_after_functions, ok_count, fail_count = expand_music_functions_with_lily(
            text,
            lily_cmd=lily_cmd,
            preserve_linebreaks=opts.preserve_linebreaks,
        )
        if ok_count or fail_count:
            text = text_after_functions
            report.lily_failures += fail_count

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

    if opts.expand_repeat_unfold:
        text_after_repeat, count = expand_repeat_unfold(text)
        if count:
            text = text_after_repeat
            report.repeats_unfolded = count

    if opts.normalize_tuplets:
        text_after_tuplets, count = normalize_tuplets(text)
        if count:
            text = text_after_tuplets
            report.tuplets_normalized = count

    if opts.normalize_drums:
        text_after_drums, changed_blocks = normalize_drummode(text)
        if changed_blocks:
            text = text_after_drums
            report.drum_blocks_normalized = changed_blocks

    if opts.normalize_whitespace:
        text = normalize_whitespace(text)

    if opts.canonicalize_chord_brackets:
        text = _canonicalize_chord_brackets(text)

    return text, report


try:
    from lilynorm.utils.options import NormOptions
except Exception:
    class NormOptions:  # type: ignore[override]
        keep_engraving: bool = False
        strip_scheme_blocks: bool = True
        strip_comments: bool = True
        normalize_whitespace: bool = False
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
    parse_opts = ParseOptions(**values)

    return parse_opts


def run(text: str, opts: "NormOptions") -> str:
    lily_cmd = resolve_lily_cmd()
    parse_opts = _map_options(opts)

    if not lily_available(lily_cmd):
        if (
            parse_opts.expand_music_functions
            or parse_opts.resolve_transpose
        ):
            print(
                "[normalize] LilyPond not found – skipping music-functions/transpose.",
                file=sys.stderr,
            )
        parse_opts.expand_music_functions = False
        parse_opts.resolve_transpose = False

    output, report = process_string(text, lily_cmd=lily_cmd, opts=parse_opts)

    print(
        f"[normalize] transpose_ok:{report.transpose_blocks} "
        f"repeat:{report.repeats_unfolded} "
        f"tuplets:{report.tuplets_normalized} "
        f"drums:{report.drum_blocks_normalized} "
        f"lily_fail:{report.lily_failures}"
    )
    return output


def build_arg_parser() -> argparse.ArgumentParser:
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
