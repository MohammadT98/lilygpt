#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import music21 as _m21


# ---------- Constants ----------

LILYPOND_TIMEOUT_SECONDS = 5
MAX_ERROR_TEXT_LENGTH = 300
DEFAULT_WINDOWS_LILYPOND = Path(
    r"C:\lilypond-2.24.4-mingw-x86_64\lilypond-2.24.4\bin\lilypond.exe"
)

MIN_NOTES_FOR_TONAL_ANALYSIS = 6
MIN_SEGMENT_NOTES = 3
DEFAULT_SEGMENT_COUNT = 4
DEFAULT_IN_KEY_THRESHOLD = 0.98
DEFAULT_BEATS_EPSILON = 1e-6


# ---------- Regexes & key/time helpers ----------

KEY_RE = re.compile(
    r"\\key\s+((?:[a-g]|do|re|mi|fa|sol|la|si))(isis|eses|is|es|dd|d|bb|b)?\s+\\([A-Za-z]+)"
)
TIME_RE = re.compile(r"\\time\s+(\d+)\s*/\s*(\d+)")

_REL_ANY_RE = re.compile(r"\\relative\b")
_REL_CPRIME_RE = re.compile(r"\\relative\s+c'\s*\{")
_REL_ANCHOR_RE = re.compile(
    r"\\relative\s+(?:[a-g]|do|re|mi|fa|sol|la|si)(?:isis|eses|is|es|dd|d|bb|b)?[',]*\s*\{"
)
_REL_PLAIN_C_RE = re.compile(r"\\relative\s+c\s*\{")

_VERSION_RE = re.compile(r"^\s*\\version\b.*$", re.MULTILINE)
_LANGUAGE_RE = re.compile(r"^\s*\\language\b.*$", re.MULTILINE)
_ASSIGN_START = re.compile(r"(?m)^[A-Za-z_][A-Za-z0-9_-]*\s*=\s*")

_NAME_TO_PC = {
    "c": 0,
    "d": 2,
    "e": 4,
    "f": 5,
    "g": 7,
    "a": 9,
    "b": 11,
    "do": 0,
    "re": 2,
    "mi": 4,
    "fa": 5,
    "sol": 7,
    "la": 9,
    "si": 11,
}
_ACC_OFFSETS = {
    "is": +1,
    "isis": +2,
    "es": -1,
    "eses": -2,
    "d": +1,
    "dd": +2,
    "b": -1,
    "bb": -2,
}

DEFAULT_NOTE_DURATION = 4
NOTE_RE = re.compile(
    r"(?<!\\)(?<![A-Za-z])((?:[a-g]|do|re|mi|fa|sol|la|si))(isis|eses|is|es|dd|d|bb|b)?([',]*)(\d+)?(\.*)"
)
UPPERCASE_NOTE_IN_BODY_RE = re.compile(
    r"\b(?:[A-G]|DO|RE|MI|FA|SOL|LA|SI)(?:isis|eses|is|es|dd|d|bb|b)?[',]*\d?(?:\.*)?"
)

FORBIDDEN_PATTERNS = {
    "rests": re.compile(r"(?<!\\)\br(?:\d+(?:\.*)?)?\b"),
    "chords": re.compile(r"<\s*(?:[a-g]|do|re|mi|fa|sol|la|si)"),
    "voices": re.compile(r"<<|\\\\|>>"),
    "repeat": re.compile(r"\\repeat\b"),
    "tuplet": re.compile(r"\\tuplet\b"),
    "ties": re.compile(r"~"),
    "grace": re.compile(r"\\grace\b|\\acciaccatura\b|\\appoggiatura\b"),
    "skips": re.compile(r"\bs(?:\d+(?:\.*)?)?\b"),
    "score": re.compile(r"\\score\b"),
    "layout": re.compile(r"\\layout\b"),
}


# ---------- Utils ----------


def _truncate_err(message: str | None, max_len: int = MAX_ERROR_TEXT_LENGTH) -> str | None:
    if not message:
        return message
    message = message.strip()
    return (message[:max_len] + "…") if len(message) > max_len else message


def _pc_from_name_acc(letter: str, acc: str | None) -> int:
    return (_NAME_TO_PC[letter] + _ACC_OFFSETS.get(acc or "", 0)) % 12


def strip_comments(text: str) -> str:
    text = re.sub(r"%\{[\s\S]*?%\}", "", text)
    text = re.sub(r"(?m)%.*$", "", text)
    return text


def _extract_declared(text: str) -> tuple[tuple[int, str] | None, tuple[int, int] | None, str]:
    m_key = KEY_RE.search(text)
    m_time = TIME_RE.search(text)

    key_pcm: tuple[int, str] | None = None
    if m_key:
        letter, acc, mode = m_key.groups()
        key_pcm = (_pc_from_name_acc(letter, acc), mode.lower())

    time_sig = (int(m_time.group(1)), int(m_time.group(2))) if m_time else None
    notation = "relative" if (_REL_CPRIME_RE.search(text) or _REL_ANY_RE.search(text)) else "absolute"
    return key_pcm, time_sig, notation


def _extract_brace_block(text: str, start_idx: int) -> str | None:
    depth = 0
    block_start = None
    for i in range(start_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth == 1:
                block_start = i + 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and block_start is not None:
                return text[block_start:i]
    return None


def _extract_relative_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    for match in re.finditer(r"\\relative\b", text):
        brace_idx = text.find("{", match.end())
        if brace_idx < 0:
            continue
        block = _extract_brace_block(text, brace_idx)
        if block is not None:
            blocks.append(block)
    return blocks


def _extract_top_level_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    depth = 0
    block_start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                block_start = i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and block_start is not None:
                blocks.append(text[block_start:i])
                block_start = None
    return blocks


def extract_music_body(text: str) -> str:
    code = strip_comments(text)
    rel_blocks = _extract_relative_blocks(code)
    if rel_blocks:
        return "\n".join(block.strip() for block in rel_blocks if block.strip())

    blocks = _extract_top_level_blocks(code)
    if blocks:
        return "\n".join(block.strip() for block in blocks if block.strip())

    return code.strip()


def tokenize_notes(body: str) -> List[Tuple]:
    tokens: List[Tuple] = []
    i, length = 0, len(body)
    last_dur = None
    last_dots = 0

    while i < length:
        note_match = NOTE_RE.match(body, i)
        if note_match:
            n, acc, marks, dur, dots = note_match.groups()
            if dur:
                dur_val = int(dur)
                dots_val = len(dots)
                last_dur, last_dots = dur_val, dots_val
            else:
                dur_val = last_dur or DEFAULT_NOTE_DURATION
                dots_val = last_dots if last_dur is not None else 0
                last_dur, last_dots = dur_val, dots_val

            tokens.append(("note", n, acc or "", marks, dur_val, dots_val))
            i = note_match.end()
            continue

        i += 1

    return tokens


# ---------- LilyPond compile + text eval ----------


def _is_working_lilypond(executable: Path | str) -> bool:
    try:
        subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            timeout=LILYPOND_TIMEOUT_SECONDS,
            check=True,
        )
        return True
    except Exception:
        return False


def find_lilypond() -> Path:
    lily_name = "lilypond.exe" if os.name == "nt" else "lilypond"
    path_candidate = shutil.which(lily_name)

    if path_candidate and _is_working_lilypond(path_candidate):
        return Path(path_candidate)

    env_candidate = os.environ.get("LILYPOND_BIN")
    if env_candidate and Path(env_candidate).exists():
        if _is_working_lilypond(env_candidate):
            return Path(env_candidate)

    if DEFAULT_WINDOWS_LILYPOND.exists():
        if _is_working_lilypond(DEFAULT_WINDOWS_LILYPOND):
            return DEFAULT_WINDOWS_LILYPOND

    raise FileNotFoundError("LilyPond executable not found.")


def _extract_version_line(text: str) -> Tuple[Optional[str], str]:
    match = _VERSION_RE.search(text)
    if not match:
        return None, text

    version_line = match.group(0).strip()
    start, end = match.span()
    remaining = text[:start] + text[end:]
    remaining = re.sub(r"^\s*\n", "", remaining, count=1, flags=re.MULTILINE)
    return version_line, remaining


def _extract_language_line(text: str) -> Tuple[Optional[str], str]:
    match = _LANGUAGE_RE.search(text)
    if not match:
        return None, text

    lang_line = match.group(0).strip()
    start, end = match.span()
    remaining = text[:start] + text[end:]
    remaining = re.sub(r"^\s*\n", "", remaining, count=1, flags=re.MULTILINE)
    return lang_line, remaining


def _sha256_text(text: str) -> str:
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _derive_midi_name_from_out_ly(out_ly: Path) -> str:
    stem = out_ly.stem
    if stem.startswith("out_"):
        return "midi_" + stem[len("out_"):]
    return stem + "_midi"


def _pull_block_by_name(name: str, text: str) -> Tuple[str, List[str]]:
    pattern = re.compile(rf"\\{name}\s*\{{", re.IGNORECASE)
    remaining = text
    blocks: List[str] = []
    search_start = 0

    while True:
        match = pattern.search(remaining, search_start)
        if not match:
            break

        depth = 0
        brace_index = match.end() - 1
        end_index: Optional[int] = None

        while brace_index < len(remaining):
            char = remaining[brace_index]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end_index = brace_index + 1
                    break
            brace_index += 1

        if end_index is None:
            break

        blocks.append(remaining[match.start():end_index])
        remaining = remaining[:match.start()] + remaining[end_index:]
        search_start = match.start()

    return remaining, blocks


def _ensure_token_block(block: str, name: str) -> str:
    if re.search(rf"\\{name}\s*\{{", block, flags=re.IGNORECASE):
        return block

    insert_position = block.rfind('}')
    if insert_position == -1:
        return block + f"\n  \\{name} {{}}\n"

    return (
        block[:insert_position]
        + f"\n  \\{name} {{}}\n"
        + block[insert_position:]
    )


def _strip_comments(text: str) -> str:
    return re.sub(r"%.*$", "", text, flags=re.MULTILINE)


def _has_top_level_staff(text: str) -> bool:
    without_comments = _strip_comments(text)
    pattern = re.compile(r"\\(?:new\s+Staff|context\s+Staff)\s*\{", re.IGNORECASE)
    return bool(pattern.search(without_comments))


def _hoist_top_level_assignments(text: str) -> Tuple[str, str]:
    index = 0
    length = len(text)

    body_parts: List[str] = []
    definitions: List[str] = []

    while index < length:
        match = _ASSIGN_START.search(text, index)
        if not match:
            body_parts.append(text[index:])
            break

        body_parts.append(text[index:match.start()])
        scan_index = match.end()

        while scan_index < length and text[scan_index].isspace():
            scan_index += 1

        brace_start = text.find('{', scan_index)
        if brace_start == -1:
            body_parts.append(text[match.start():])
            break

        depth = 0
        brace_index = brace_start
        end_index: Optional[int] = None

        while brace_index < length:
            char = text[brace_index]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end_index = brace_index + 1
                    break
            brace_index += 1

        if end_index is None:
            body_parts.append(text[match.start():])
            break

        trailing_index = end_index
        while trailing_index < length and text[trailing_index] in (" ", "\t", "\r", "\n"):
            trailing_index += 1

        definitions.append(text[match.start():trailing_index])
        index = trailing_index

    body = "".join(body_parts)
    defs_text = "".join(definitions).strip()
    if defs_text:
        defs_text += "\n\n"

    return body, defs_text


def _build_score_from_body(full_text: str) -> str:
    version_line, remaining = _extract_version_line(full_text)
    language_line, remaining = _extract_language_line(remaining)
    remaining, header_blocks = _pull_block_by_name("header", remaining)
    remaining, paper_blocks = _pull_block_by_name("paper", remaining)
    header_and_paper = "".join(header_blocks + paper_blocks)

    remainder_without_scores, score_blocks = _pull_block_by_name("score", remaining)
    if score_blocks:
        first_score = score_blocks[0]
        first_score = _ensure_token_block(first_score, "layout")
        first_score = _ensure_token_block(first_score, "midi")
        prefix_parts = []
        if version_line:
            prefix_parts.append(version_line)
        if language_line:
            prefix_parts.append(language_line)
        header_prefix = "\n\n".join(prefix_parts)
        prefix = (header_prefix + "\n\n") if header_prefix else ""
        prefix += header_and_paper
        remainder_clean = remainder_without_scores.strip()
        if remainder_clean:
            prefix += remainder_clean + "\n\n"
        return f"{prefix}{first_score}\n"

    body = remaining
    body, hoisted_defs = _hoist_top_level_assignments(body)
    body = body.strip()

    if not body:
        staff_payload = "s1"
    else:
        if _has_top_level_staff(body):
            staff_payload = body
        else:
            staff_payload = f"\\new Staff {{\n{body}\n}}"

    score = f"""\\score {{
  {staff_payload}
  \\layout {{}}
  \\midi {{}}
}}"""

    prefix_parts = []
    if version_line:
        prefix_parts.append(version_line)
    if language_line:
        prefix_parts.append(language_line)
    header_prefix = "\n\n".join(prefix_parts)
    prefix = (header_prefix + "\n\n") if header_prefix else ""
    return f"{prefix}{header_and_paper}{hoisted_defs}{score}\n"


def lily_to_midi(
    out_ly: str | Path,
    *,
    midi_dir: str | Path,
    force: bool = False,
    lilypond_version_tag: str = "2.24.4",
) -> Dict[str, Any]:
    out_ly_path = Path(out_ly)
    midi_dir_path = Path(midi_dir)
    midi_dir_path.mkdir(parents=True, exist_ok=True)

    midi_stem = _derive_midi_name_from_out_ly(out_ly_path)
    target_midi_path = (midi_dir_path / f"{midi_stem}.mid").resolve()
    ly_with_midi_path = (
        midi_dir_path / f"ly_with_midi_{midi_stem.split('_')[-1]}.ly"
    ).resolve()
    log_path = (midi_dir_path / f"{midi_stem}.log").resolve()

    if target_midi_path.exists() and not force:
        try:
            source_hash = _sha256_text(out_ly_path.read_text(encoding="utf-8"))
        except Exception:
            source_hash = None

        return {
            "ok": True,
            "seconds": 0.0,
            "reason": "exists",
            "paths": {"ly_with_midi": None, "midi": str(target_midi_path)},
            "sha256": {"source_ly": source_hash},
            "tooling": {"lilypond_version": lilypond_version_tag},
            "error": None,
        }

    try:
        original_text = out_ly_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return {
            "ok": False,
            "seconds": None,
            "reason": "encoding_error",
            "paths": {"ly_with_midi": None, "midi": str(target_midi_path)},
            "sha256": {"source_ly": None},
            "tooling": {"lilypond_version": lilypond_version_tag},
            "error": _truncate_err(str(exc)),
        }

    render_text = _build_score_from_body(original_text)
    source_hash = _sha256_text(original_text)

    try:
        ly_with_midi_path.write_text(render_text, encoding="utf-8")
    except Exception:
        pass

    try:
        lilypond_executable = find_lilypond()
    except Exception as exc:
        return {
            "ok": False,
            "seconds": None,
            "reason": "lilypond_not_found",
            "paths": {"ly_with_midi": str(ly_with_midi_path), "midi": str(target_midi_path)},
            "sha256": {"source_ly": source_hash},
            "tooling": {"lilypond_version": lilypond_version_tag},
            "error": _truncate_err(str(exc)),
        }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_ly_path = temp_path / "render.ly"
        temp_ly_path.write_text(render_text, encoding="utf-8")

        out_base = temp_path / "render"
        cmd = [
            str(lilypond_executable),
            "-dno-point-and-click",
            "-o",
            str(out_base),
            str(temp_ly_path),
        ]

        start_time = time.perf_counter()
        proc = None
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=LILYPOND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - start_time
            try:
                log = (
                    f"[lilypond_timeout] cmd: {' '.join(cmd)}\n"
                    f"timeout_seconds: {LILYPOND_TIMEOUT_SECONDS}\n\n"
                    f"STDOUT:\n{exc.stdout or ''}\n\n"
                    f"STDERR:\n{exc.stderr or ''}\n"
                )
                log_path.write_text(log, encoding="utf-8")
            except Exception:
                pass
            return {
                "ok": False,
                "seconds": round(elapsed, 4),
                "reason": "lilypond_timeout",
                "paths": {"ly_with_midi": str(ly_with_midi_path), "midi": str(target_midi_path)},
                "sha256": {"source_ly": source_hash},
                "tooling": {"lilypond_version": lilypond_version_tag},
                "error": _truncate_err(
                    f"LilyPond timed out after {LILYPOND_TIMEOUT_SECONDS} seconds."
                ),
            }
        elapsed = time.perf_counter() - start_time

        def _dump_log(tag: str, extra: str = "") -> None:
            try:
                log = (
                    f"[{tag}] cmd: {' '.join(cmd)}\n"
                    f"returncode: {proc.returncode}\n\n"
                    f"STDOUT:\n{proc.stdout or ''}\n\n"
                    f"STDERR:\n{proc.stderr or ''}\n"
                )
                if extra:
                    log += f"\n{extra}\n"
                log_path.write_text(log, encoding="utf-8")
            except Exception:
                pass

        if proc.returncode != 0:
            _dump_log("lilypond_failed")
            return {
                "ok": False,
                "seconds": elapsed,
                "reason": "lilypond_failed",
                "paths": {"ly_with_midi": str(ly_with_midi_path), "midi": str(target_midi_path)},
                "sha256": {"source_ly": source_hash},
                "tooling": {"lilypond_version": lilypond_version_tag},
                "error": _truncate_err(proc.stderr or proc.stdout or "LilyPond failed"),
            }

        candidates: List[Path] = []
        for candidate in [
            out_base.with_suffix(".midi"),
            out_base.with_suffix(".mid"),
        ]:
            if candidate.exists():
                candidates.append(candidate)

        if not candidates:
            candidates += sorted(temp_path.glob("render-*.midi"))
            candidates += sorted(temp_path.glob("render-*.mid"))

        if not candidates:
            candidates += sorted(temp_path.glob("*.midi"))
            candidates += sorted(temp_path.glob("*.mid"))

        if not candidates:
            _dump_log(
                "no_midi_output",
                extra=(
                    "No MIDI produced. Check generated score or musical content.\n"
                    f"Generated file: {ly_with_midi_path}"
                ),
            )
            return {
                "ok": False,
                "seconds": elapsed,
                "reason": "no_midi_output",
                "paths": {"ly_with_midi": str(ly_with_midi_path), "midi": str(target_midi_path)},
                "sha256": {"source_ly": source_hash},
                "tooling": {"lilypond_version": lilypond_version_tag},
                "error": "LilyPond did not produce MIDI.",
            }

        shutil.move(str(candidates[0]), str(target_midi_path))

    return {
        "ok": True,
        "seconds": round(elapsed, 4),
        "reason": "rendered",
        "paths": {"ly_with_midi": str(ly_with_midi_path), "midi": str(target_midi_path)},
        "sha256": {"source_ly": source_hash},
        "tooling": {"lilypond_version": lilypond_version_tag},
        "error": None,
    }


# ---------- Text evaluation (no bar checks) ----------


def evaluate_lilypond_text(
    lily_text: str,
    *,
    expected_notation: str = "relative",
    require_lowercase: bool = True,
    allow_accidentals: bool = True,
    allowed_forbidden: Optional[set[str]] = None,
) -> Dict[str, Any]:
    stripped = strip_comments(lily_text)
    body = extract_music_body(lily_text)

    has_key = bool(KEY_RE.search(stripped))
    has_time = bool(TIME_RE.search(stripped))
    has_key_time = has_key and has_time

    rel_count = len(_REL_ANY_RE.findall(stripped))
    has_rel_cprime = bool(_REL_CPRIME_RE.search(stripped))
    has_rel_c_plain = bool(_REL_PLAIN_C_RE.search(stripped))
    has_rel_anchor = bool(_REL_ANCHOR_RE.search(stripped))

    notation_mode = (expected_notation or "relative").lower()
    if notation_mode == "relative":
        notation_ok = (rel_count >= 1 and has_rel_anchor)
        relative_ok = notation_ok
    else:
        notation_ok = (rel_count == 0)
        relative_ok = False

    lowercase_ok = True
    if require_lowercase:
        lowercase_ok = not bool(UPPERCASE_NOTE_IN_BODY_RE.search(body))

    allowed_forbidden = set(allowed_forbidden or [])
    forb_hits = {}
    for name, pattern in FORBIDDEN_PATTERNS.items():
        if name in allowed_forbidden:
            continue
        target = body if name not in {"score", "layout"} else stripped
        forb_hits[name] = bool(pattern.search(target))
    no_forbidden = not any(forb_hits.values())

    tokens = tokenize_notes(body)

    accidentals_present = any(
        (token[0] == "note" and token[2])
        for token in tokens
    )
    accidentals_ok = allow_accidentals or not accidentals_present

    flags: Dict[str, bool] = {
        "has_key_time": has_key_time,
        "notation_ok": notation_ok,
        "relative_ok": relative_ok if notation_mode == "relative" else True,
        "lowercase_ok": lowercase_ok,
        "no_forbidden": no_forbidden,
        "accidentals_ok": accidentals_ok,
    }
    adherence = (
        sum(bool(v) for v in flags.values()) / len(flags)
        if flags
        else 0.0
    )
    failed_flags = [name for name, ok in flags.items() if not ok]

    return {
        "has_key_time": bool(has_key_time),
        "relative_anchor_cprime": bool(has_rel_cprime),
        "relative_anchor_plain_c": bool(has_rel_c_plain),
        "relative_anchor_any": bool(has_rel_anchor),
        "notation_ok": bool(notation_ok),
        "lowercase_ok": bool(lowercase_ok),
        "no_forbidden": bool(no_forbidden),
        "forbidden_hits": forb_hits,
        "accidentals_present": bool(accidentals_present),
        "accidentals_ok": bool(accidentals_ok),
        "adherence_text": float(adherence),
        "failed_flags": failed_flags,
        "config": {
            "expected_notation": expected_notation,
            "require_lowercase": require_lowercase,
            "allow_accidentals": allow_accidentals,
            "allowed_forbidden": sorted(allowed_forbidden) if allowed_forbidden else None,
        },
    }


def eval_lily_text(
    ly_path: Path,
    *,
    expected_notation: str,
    require_lowercase: bool,
    allow_accidentals: bool,
    allowed_forbidden: Optional[set[str]],
    lilypond_bin: Optional[str | Path],
) -> Dict[str, Any]:
    start = time.perf_counter()
    text = ly_path.read_text(encoding="utf-8", errors="ignore")

    metrics = evaluate_lilypond_text(
        text,
        expected_notation=expected_notation,
        require_lowercase=require_lowercase,
        allow_accidentals=allow_accidentals,
        allowed_forbidden=allowed_forbidden,
    )

    compiles: Optional[bool] = None
    compile_error: Optional[str] = None
    compile_via: Optional[str] = None
    parser_error: Optional[str] = None

    if lilypond_bin:
        lilypond_bin_str = str(lilypond_bin)
        if not Path(lilypond_bin_str).exists():
            compiles = False
            compile_error = f"lilypond not found: {lilypond_bin_str}"
            compile_via = "lilypond"
        else:
            try:
                with tempfile.TemporaryDirectory() as td:
                    tmp_dir = Path(td)
                    test_file = tmp_dir / "test.ly"

                    src = (
                        text
                        if "\\version" in text
                        else '\\version "2.24.4"\n' + text
                    )
                    test_file.write_text(src, encoding="utf-8")

                    result = subprocess.run(
                        [
                            lilypond_bin_str,
                            "-dno-print-pages",
                            "-dbackend=null",
                            "-o",
                            str(tmp_dir / "out"),
                            str(test_file),
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        encoding="utf-8",
                        timeout=15,
                    )
                    compiles = (result.returncode == 0)
                    compile_via = "lilypond"
                    if not compiles:
                        lines = (result.stderr or result.stdout or "Compilation failed").strip().splitlines()
                        first_line = lines[0] if lines else "Compilation failed"
                        compile_error = first_line[:300]
            except subprocess.TimeoutExpired:
                compiles = False
                compile_error = "Compile timeout"
                compile_via = "lilypond"
            except Exception as exc:
                compiles = False
                compile_error = str(exc)[:300]
                compile_via = "lilypond"

    if compiles is None:
        try:
            src = (
                text
                if "\\version" in text
                else '\\version "2.24.4"\n' + text
            )
            _m21.converter.parseData(src, format="lilypond")
            compiles = True
            compile_via = "music21"
        except Exception as exc:
            compiles = False
            compile_via = "music21"
            parser_error = str(exc)[:300]

    if compiles is not None:
        metrics["compiles"] = bool(compiles)

    elapsed = time.perf_counter() - start
    return {
        "ok": True,
        "seconds": round(elapsed, 4),
        "metrics": metrics,
        "error": None,
        "compile_error": compile_error,
        "compile_via": compile_via,
        "parser_error": parser_error,
    }


# ---------- MIDI evaluation (no bar checks) ----------


def _normalize_key_string(key: str | None) -> str | None:
    if not isinstance(key, str):
        return key

    parts = key.strip().split()
    if len(parts) == 2:
        return f"{parts[0].lower()} {parts[1].lower()}"
    return key.lower()


def _scale_pcset(tonic_pc: int, mode: str) -> set[int]:
    mode_intervals = {
        "ionian": [0, 2, 4, 5, 7, 9, 11],
        "major": [0, 2, 4, 5, 7, 9, 11],
        "aeolian": [0, 2, 3, 5, 7, 8, 10],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
        "lydian": [0, 2, 4, 6, 7, 9, 11],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "locrian": [0, 1, 3, 5, 6, 8, 10],
    }
    intervals = mode_intervals.get(mode.lower(), mode_intervals["major"])
    return {(tonic_pc + interval) % 12 for interval in intervals}


def _compute_tonal_stability(stream, declared_key_pc_mode: Optional[Tuple[int, str]] = None) -> Dict[str, Any]:
    try:
        flat_stream = stream.flatten()
        notes_and_chords = [
            n for n in flat_stream.notes
            if getattr(n, "isNote", False) or getattr(n, "isChord", False)
        ]

        if len(notes_and_chords) < MIN_NOTES_FOR_TONAL_ANALYSIS:
            return {"tonal_stability_score": None, "key_changes": None, "drift_detected": None}

        segment_size = len(notes_and_chords) // DEFAULT_SEGMENT_COUNT
        if segment_size < MIN_SEGMENT_NOTES:
            segment_size = max(MIN_SEGMENT_NOTES, len(notes_and_chords) // 2)

        from music21 import stream as m21_stream
        segments: List[Dict[str, Any]] = []

        for i in range(0, len(notes_and_chords), segment_size):
            segment_notes = notes_and_chords[i:i + segment_size]
            if len(segment_notes) < MIN_SEGMENT_NOTES:
                continue

            try:
                temp_stream = m21_stream.Stream()
                for note_or_chord in segment_notes:
                    temp_stream.append(note_or_chord)

                key_analysis = temp_stream.analyze("key")
                segments.append(
                    {
                        "index": len(segments),
                        "note_count": len(segment_notes),
                        "key": str(key_analysis),
                        "tonic_pc": key_analysis.tonic.midi % 12,
                        "mode": key_analysis.mode,
                    }
                )
            except Exception:
                continue

        if len(segments) < 2:
            return {"tonal_stability_score": None, "key_changes": None, "drift_detected": None}

        if declared_key_pc_mode is None:
            key_changes = 0
            previous_tonic = segments[0]["tonic_pc"]

            for segment in segments[1:]:
                if segment["tonic_pc"] != previous_tonic:
                    key_changes += 1
                previous_tonic = segment["tonic_pc"]

            stable_segments = sum(
                1
                for segment in segments
                if segment["tonic_pc"] == segments[0]["tonic_pc"]
            )
            stability_score = stable_segments / len(segments)
            drift_detected = (stability_score < 0.7) or (key_changes > 1)

            return {
                "tonal_stability_score": float(stability_score),
                "key_changes": int(key_changes),
                "drift_detected": bool(drift_detected),
            }

        tonic_pc, mode = declared_key_pc_mode
        same_key_count = 0
        key_changes = 0
        previous_tonic = segments[0]["tonic_pc"]

        for segment in segments:
            if (
                segment["tonic_pc"] == tonic_pc
                and segment["mode"].lower() == mode.lower()
            ):
                same_key_count += 1

            if segment["tonic_pc"] != previous_tonic:
                key_changes += 1
            previous_tonic = segment["tonic_pc"]

        stability_score = same_key_count / len(segments)
        drift_detected = (stability_score < 0.7) or (key_changes > 1)

        return {
            "tonal_stability_score": float(stability_score),
            "key_changes": int(key_changes),
            "drift_detected": bool(drift_detected),
        }

    except Exception:
        return {"tonal_stability_score": None, "key_changes": None, "drift_detected": None}


def _compute_contour_analysis(sequential_pitches: List[int]) -> Dict[str, Any]:
    if len(sequential_pitches) < 4:
        return {
            "contour_types": [],
            "contour_diversity": 0,
            "ascending_tendency": 0.0,
            "descending_tendency": 0.0,
            "arch_shapes": 0,
            "downward_arches": 0,
        }

    contour_symbols: List[str] = []
    for i in range(1, len(sequential_pitches)):
        diff = sequential_pitches[i] - sequential_pitches[i - 1]
        if diff > 0:
            contour_symbols.append("U")
        elif diff < 0:
            contour_symbols.append("D")
        else:
            contour_symbols.append("S")

    contour_patterns: List[str] = []
    pattern_counts: Dict[str, int] = defaultdict(int)

    for i in range(len(contour_symbols) - 1):
        pattern = "".join(contour_symbols[i:i + 2])
        contour_patterns.append(pattern)
        pattern_counts[pattern] += 1

    directional_moves = [s for s in contour_symbols if s != "S"]
    total_moves = len(directional_moves)
    ascending_tendency = contour_symbols.count("U") / max(1, total_moves)
    descending_tendency = contour_symbols.count("D") / max(1, total_moves)

    arch_shapes = 0
    downward_arches = 0
    for pattern in contour_patterns:
        if pattern == "UD":
            arch_shapes += 1
        elif pattern == "DU":
            downward_arches += 1

    contour_diversity = len(pattern_counts) / max(1, len(contour_patterns))

    common_contours = sorted(
        pattern_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    contour_types = [pattern for pattern, _count in common_contours]

    return {
        "contour_types": contour_types,
        "contour_diversity": float(contour_diversity),
        "ascending_tendency": float(ascending_tendency),
        "descending_tendency": float(descending_tendency),
        "arch_shapes": int(arch_shapes),
        "downward_arches": int(downward_arches),
    }


def _compute_interval_entropy(intervals: List[int]) -> float:
    if not intervals:
        return 0.0

    counts = Counter(intervals)
    total = sum(counts.values())
    entropy = 0.0

    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return float(entropy)


def _build_pitch_sequences(sf, notes, chords) -> Tuple[List[int], List[Tuple[str, Any]]]:
    all_pitches: List[int] = []
    for n in notes:
        all_pitches.append(int(n.pitch.midi))

    for chord in chords:
        for pitch in chord.pitches:
            all_pitches.append(int(pitch.midi))

    all_elements: List[Tuple[str, Any]] = []
    for n in notes:
        all_elements.append(("note", n))
    for chord in chords:
        all_elements.append(("chord", chord))

    all_elements.sort(key=lambda pair: pair[1].offset)

    return all_pitches, all_elements


def _compute_intervals_from_elements(all_elements: List[Tuple[str, Any]]) -> Tuple[List[int], List[int]]:
    sequential_pitches: List[int] = []

    for elem_type, elem in all_elements:
        if elem_type == "note":
            sequential_pitches.append(int(elem.pitch.midi))
        else:
            chord_pitches = [int(p.midi) for p in elem.pitches]
            sequential_pitches.append(min(chord_pitches))

    intervals = [
        sequential_pitches[i + 1] - sequential_pitches[i]
        for i in range(len(sequential_pitches) - 1)
    ]
    return sequential_pitches, intervals


def _compute_detected_key(s) -> Optional[str]:
    try:
        detected_key_raw = str(s.analyze("key"))
    except Exception:
        detected_key_raw = None
    return _normalize_key_string(detected_key_raw)


def _compute_in_key_share_and_cadence(
    notes,
    chords,
    sequential_pitches: List[int],
    declared_key_pc_mode: Optional[Tuple[int, str]],
    in_key_threshold: float,
) -> Tuple[Optional[float], Optional[bool], Optional[bool]]:
    if declared_key_pc_mode is None:
        return None, None, None

    tonic_pc, mode = declared_key_pc_mode
    pcset = _scale_pcset(tonic_pc, mode)

    total_q = 0.0
    in_q = 0.0

    for n in notes:
        q = float(getattr(n.duration, "quarterLength", 0.0))
        total_q += q
        if int(n.pitch.midi) % 12 in pcset:
            in_q += q

    for chord in chords:
        q = float(getattr(chord.duration, "quarterLength", 0.0))
        total_q += q
        chord_pcs = {int(p.midi) % 12 for p in chord.pitches}
        if chord_pcs.issubset(pcset):
            in_q += q

    if total_q <= 0:
        in_key_time_share = None
    else:
        in_key_time_share = in_q / total_q

    in_key_ok = (
        (in_key_time_share is not None)
        and (in_key_time_share >= in_key_threshold)
    )

    ends_on_tonic = None
    if sequential_pitches:
        ends_on_tonic = (sequential_pitches[-1] % 12 == (tonic_pc % 12))

    return in_key_time_share, ends_on_tonic, in_key_ok


def _compute_rhythmic_diversity(notes, chords) -> int:
    rhythmic_values = set()
    for n in notes:
        rhythmic_values.add(n.duration.quarterLength)
    for chord in chords:
        rhythmic_values.add(chord.duration.quarterLength)
    return len(rhythmic_values)


def _compute_range_ok(
    all_pitches: List[int],
    range_bounds: Optional[Tuple[int, int]],
) -> Optional[bool]:
    if range_bounds is None or not all_pitches:
        return None

    low, high = range_bounds
    return (min(all_pitches) >= low) and (max(all_pitches) <= high)


def _analyze_midi_with_music21(
    midi_path: Path,
    *,
    declared_key_pc_mode: Optional[Tuple[int, str]] = None,
    declared_time: Optional[Tuple[int, int]] = None,
    range_bounds: Optional[Tuple[int, int]] = None,
    in_key_threshold: float = DEFAULT_IN_KEY_THRESHOLD,
) -> Dict[str, Any]:
    from music21 import converter

    score = converter.parse(str(midi_path))
    sf = score.flatten()

    notes = [n for n in sf.notes if getattr(n, "isNote", False)]
    chords = [c for c in sf.notes if getattr(c, "isChord", False)]

    all_pitches, all_elements = _build_pitch_sequences(sf, notes, chords)

    if not all_pitches:
        return {
            "note_count": 0,
            "min_midi": None,
            "max_midi": None,
            "interval_entropy": 0.0,
            "step_vs_leap": 0.0,
            "direction_changes": 0,
            "repeat_rate": 0.0,
            "avg_interval_size": 0.0,
            "rhythmic_diversity": 0,
            "detected_key": None,
            "in_key_pct_time_weighted": None,
            "ends_on_tonic": None,
            "range_ok": None,
            "in_key_ok": None,
            "tonal_stability_score": None,
            "key_changes": None,
            "drift_detected": None,
            "contour_types": [],
            "contour_diversity": 0.0,
            "ascending_tendency": 0.0,
            "descending_tendency": 0.0,
            "arch_shapes": 0,
            "downward_arches": 0,
            "config": {
                "declared_key_pc_mode": declared_key_pc_mode,
                "declared_time": declared_time,
                "range_bounds": range_bounds,
                "in_key_threshold": in_key_threshold,
            },
        }

    sequential_pitches, intervals = _compute_intervals_from_elements(all_elements)

    interval_entropy = _compute_interval_entropy(intervals)

    steps = sum(1 for d in intervals if abs(d) <= 2)
    leaps = sum(1 for d in intervals if abs(d) > 2)
    total_interval_count = steps + leaps
    step_vs_leap = (steps / total_interval_count) if total_interval_count else 0.0

    direction_changes = sum(
        1
        for i in range(1, len(intervals))
        if (intervals[i - 1] < 0 < intervals[i])
        or (intervals[i - 1] > 0 > intervals[i])
    )

    repeats = sum(
        1
        for i in range(1, len(sequential_pitches))
        if sequential_pitches[i] == sequential_pitches[i - 1]
    )
    repeat_rate = repeats / max(1, len(sequential_pitches) - 1)

    detected_key = _compute_detected_key(score)

    in_key_time_share, ends_on_tonic, in_key_ok = _compute_in_key_share_and_cadence(
        notes,
        chords,
        sequential_pitches,
        declared_key_pc_mode,
        in_key_threshold,
    )

    rhythmic_diversity = _compute_rhythmic_diversity(notes, chords)

    avg_interval_size = (
        sum(abs(i) for i in intervals) / len(intervals) if intervals else 0.0
    )

    range_ok = _compute_range_ok(all_pitches, range_bounds)

    tonal_stability = _compute_tonal_stability(score, declared_key_pc_mode)

    contour_analysis = _compute_contour_analysis(sequential_pitches)

    return {
        "note_count": len(all_pitches),
        "chord_count": len(chords),
        "element_count": len(sequential_pitches),
        "min_midi": min(all_pitches),
        "max_midi": max(all_pitches),
        "detected_key": detected_key,
        "interval_entropy": float(interval_entropy),
        "step_vs_leap": float(step_vs_leap),
        "direction_changes": int(direction_changes),
        "in_key_pct_time_weighted": (
            float(in_key_time_share)
            if in_key_time_share is not None else None
        ),
        "ends_on_tonic": bool(ends_on_tonic) if ends_on_tonic is not None else None,
        "range_ok": bool(range_ok) if range_ok is not None else None,
        "in_key_ok": bool(in_key_ok) if in_key_ok is not None else None,
        "repeat_rate": float(repeat_rate),
        "avg_interval_size": float(avg_interval_size),
        "rhythmic_diversity": int(rhythmic_diversity),
        "tonal_stability_score": tonal_stability["tonal_stability_score"],
        "key_changes": tonal_stability["key_changes"],
        "drift_detected": tonal_stability["drift_detected"],
        "contour_types": contour_analysis["contour_types"],
        "contour_diversity": contour_analysis["contour_diversity"],
        "ascending_tendency": contour_analysis["ascending_tendency"],
        "descending_tendency": contour_analysis["descending_tendency"],
        "arch_shapes": contour_analysis["arch_shapes"],
        "downward_arches": contour_analysis["downward_arches"],
        "config": {
            "declared_key_pc_mode": declared_key_pc_mode,
            "declared_time": declared_time,
            "range_bounds": range_bounds,
            "in_key_threshold": in_key_threshold,
        },
    }


def eval_midi(
    midi_path: str | Path,
    *,
    declared_key_pc_mode: Optional[Tuple[int, str]] = None,
    declared_time: Optional[Tuple[int, int]] = None,
    range_bounds: Optional[Tuple[int, int]] = None,
    in_key_threshold: float = DEFAULT_IN_KEY_THRESHOLD,
) -> Dict[str, Any]:
    midi_path = Path(midi_path)
    start_time = time.perf_counter()

    try:
        import music21  # lazy import for clearer error reporting
        tooling = {"music21_version": getattr(music21, "__version__", None)}
    except Exception as exc:
        return {
            "ok": False,
            "seconds": 0.0,
            "metrics": None,
            "error": _truncate_err(f"music21 import failed: {exc}"),
            "tooling": {"music21_version": None},
        }

    if not midi_path.exists():
        return {
            "ok": False,
            "seconds": 0.0,
            "metrics": None,
            "error": f"MIDI not found: {midi_path}",
            "tooling": tooling,
        }

    try:
        metrics = _analyze_midi_with_music21(
            midi_path,
            declared_key_pc_mode=declared_key_pc_mode,
            declared_time=declared_time,
            range_bounds=range_bounds,
            in_key_threshold=in_key_threshold,
        )
        elapsed = time.perf_counter() - start_time
        return {
            "ok": True,
            "seconds": round(elapsed, 4),
            "metrics": metrics,
            "error": None,
            "tooling": tooling,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        return {
            "ok": False,
            "seconds": round(elapsed, 4),
            "metrics": None,
            "error": _truncate_err(str(exc)),
            "tooling": tooling,
        }


# ---------- Aggregation ----------


def _iter_ly_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.ly") if p.is_file())


def _parse_allowed_forbidden(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    return {s.strip() for s in raw.split(",") if s.strip()}


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _rate(values: List[bool]) -> Optional[float]:
    return (sum(1 for v in values if v) / len(values)) if values else None


def _collect_numeric(records: List[Dict[str, Any]], path: List[str]) -> List[float]:
    out: List[float] = []
    for r in records:
        cur: Any = r
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if cur is None:
            continue
        try:
            out.append(float(cur))
        except Exception:
            continue
    return out


def _collect_bool(records: List[Dict[str, Any]], path: List[str]) -> List[bool]:
    out: List[bool] = []
    for r in records:
        cur: Any = r
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if isinstance(cur, bool):
            out.append(cur)
    return out


def _build_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"count": 0}

    summary = {
        "count": len(records),
        "lily_ok_rate": _rate(_collect_bool(records, ["lily_eval", "ok"])),
        "compiles_rate": _rate(_collect_bool(records, ["lily_eval", "metrics", "compiles"])),
        "midi_render_ok_rate": _rate(_collect_bool(records, ["midi_render", "ok"])),
        "midi_eval_ok_rate": _rate(_collect_bool(records, ["midi_eval", "ok"])),
        "has_key_time_rate": _rate(_collect_bool(records, ["lily_eval", "metrics", "has_key_time"])),
        "notation_ok_rate": _rate(_collect_bool(records, ["lily_eval", "metrics", "notation_ok"])),
        "adherence_text_avg": _mean(_collect_numeric(records, ["lily_eval", "metrics", "adherence_text"])),
        "interval_entropy_avg": _mean(_collect_numeric(records, ["midi_eval", "metrics", "interval_entropy"])),
        "step_vs_leap_avg": _mean(_collect_numeric(records, ["midi_eval", "metrics", "step_vs_leap"])),
        "avg_interval_size_avg": _mean(_collect_numeric(records, ["midi_eval", "metrics", "avg_interval_size"])),
        "direction_changes_avg": _mean(_collect_numeric(records, ["midi_eval", "metrics", "direction_changes"])),
        "repeat_rate_avg": _mean(_collect_numeric(records, ["midi_eval", "metrics", "repeat_rate"])),
        "rhythmic_diversity_avg": _mean(_collect_numeric(records, ["midi_eval", "metrics", "rhythmic_diversity"])),
        "note_count_avg": _mean(_collect_numeric(records, ["midi_eval", "metrics", "note_count"])),
        "in_key_pct_time_weighted_avg": _mean(_collect_numeric(records, ["midi_eval", "metrics", "in_key_pct_time_weighted"])),
        "in_key_ok_rate": _rate(_collect_bool(records, ["midi_eval", "metrics", "in_key_ok"])),
        "tonal_stability_score_avg": _mean(_collect_numeric(records, ["midi_eval", "metrics", "tonal_stability_score"])),
        "drift_detected_rate": _rate(_collect_bool(records, ["midi_eval", "metrics", "drift_detected"])),
        "contour_diversity_avg": _mean(_collect_numeric(records, ["midi_eval", "metrics", "contour_diversity"])),
        "ascending_tendency_avg": _mean(_collect_numeric(records, ["midi_eval", "metrics", "ascending_tendency"])),
        "descending_tendency_avg": _mean(_collect_numeric(records, ["midi_eval", "metrics", "descending_tendency"])),
        "ends_on_tonic_rate": _rate(_collect_bool(records, ["midi_eval", "metrics", "ends_on_tonic"])),
    }
    return {k: v for k, v in summary.items() if v is not None}


def _group_key(input_dir: Path, ly_path: Path) -> str:
    try:
        rel = ly_path.relative_to(input_dir)
    except ValueError:
        return "all"
    parts = rel.parts
    return parts[0] if parts else "all"


# ---------- Main ----------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate LilyPond files under a directory (Lily text + MIDI music21), Italian-aware, no bar checks.",
    )
    parser.add_argument("input_dir", nargs="?", default="data/inference/samples")
    parser.add_argument("--out", default="data/inference/sample_eval/eval.jsonl")
    parser.add_argument("--summary", default="data/inference/sample_eval/summary.json")
    parser.add_argument("--midi-dir", default="data/inference/sample_eval/midi")
    parser.add_argument("--expected-notation", default="relative")
    parser.add_argument("--require-lowercase", action="store_true", default=True)
    parser.add_argument("--no-require-lowercase", action="store_false", dest="require_lowercase")
    parser.add_argument("--disallow-accidentals", action="store_true")
    parser.add_argument(
        "--allowed-forbidden",
        help="Comma-separated list of forbidden constructs to allow (e.g. rests,score,tuplets).",
        default=None,
    )
    parser.add_argument("--force-midi", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.out)
    summary_path = Path(args.summary)
    midi_dir = Path(args.midi_dir)

    allow_accidentals = not args.disallow_accidentals
    allowed_forbidden = _parse_allowed_forbidden(args.allowed_forbidden)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)

    try:
        lilypond_bin = find_lilypond()
    except Exception:
        lilypond_bin = None

    records: List[Dict[str, Any]] = []
    for ly_path in _iter_ly_files(input_dir):
        text = ly_path.read_text(encoding="utf-8", errors="ignore")
        declared_key, declared_time, declared_notation = _extract_declared(strip_comments(text))

        lily_eval = eval_lily_text(
            ly_path,
            expected_notation=args.expected_notation,
            require_lowercase=args.require_lowercase,
            allow_accidentals=allow_accidentals,
            allowed_forbidden=allowed_forbidden,
            lilypond_bin=lilypond_bin,
        )

        group = _group_key(input_dir, ly_path)
        midi_out_dir = midi_dir / group if group and group != "all" else midi_dir
        midi_render = lily_to_midi(
            ly_path,
            midi_dir=midi_out_dir,
            force=args.force_midi,
        )

        if midi_render.get("ok"):
            midi_eval = eval_midi(
                midi_render["paths"]["midi"],
                declared_key_pc_mode=declared_key,
                declared_time=declared_time,
            )
        else:
            midi_eval = {
                "ok": False,
                "seconds": None,
                "metrics": None,
                "error": midi_render.get("error"),
                "tooling": None,
            }

        records.append(
            {
                "path": str(ly_path),
                "declared_key_pc_mode": declared_key,
                "declared_time": declared_time,
                "declared_notation": declared_notation,
                "lily_eval": lily_eval,
                "midi_render": {
                    "ok": bool(midi_render.get("ok")),
                    "seconds": midi_render.get("seconds"),
                    "reason": midi_render.get("reason"),
                    "error": midi_render.get("error"),
                    "paths": midi_render.get("paths"),
                },
                "midi_eval": midi_eval,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        key = _group_key(input_dir, Path(rec["path"]))
        grouped.setdefault(key, []).append(rec)

    summary = {
        "all": _build_summary(records),
        "by_group": {k: _build_summary(v) for k, v in sorted(grouped.items())},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Evaluated {len(records)} files. Wrote {out_path} and {summary_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
