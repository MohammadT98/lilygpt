#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Batch-normalize a folder of LilyPond sources that follow the baroquemusic.it layout.

Dataset layout (per work folder):
    variabili.ly                            - shared macros / tweaks
    vivaldi_..._header.ly                   - top-level include (paper, language, includes movements)
    vivaldi_..._format.ly                   - empty @macro placeholders
    vivaldi_..._allegro.ly                  - movement files defining I*/II*/III* macros with music
    vivaldi_..._violino1.ly, ...            - part wrappers (only \score <<\I...>>)

The music lives inside the movement files where identifiers start with roman numerals
(`Iobn`, `IIvla`, ...).  This script walks the dataset, finds those files automatically,
runs the existing normalization pipeline (preparse -> normalize -> engrave strip),
and writes the normalized results to a mirroring directory structure.

Usage:
    python scripts/process_dataset.py \\
        --input data/raw/Dataset \\
        --normalized-out data/normalized_dataset
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import TextIO

try:
    from lilynorm.utils.options import NormOptions
    from lilynorm.stages import preparse, normalize, engrave_strip, tokenize_gpt
except ModuleNotFoundError:
    # Allow running the script directly from the repo without installing the package.
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
    from lilynorm.utils.options import NormOptions
    from lilynorm.stages import preparse, normalize, engrave_strip, tokenize_gpt

# Movement files define identifiers such as `Iglobal`, `IIvla`, `IIIobn`, ...
ROMAN_DEF_RE = re.compile(r"\b[IVX]{1,4}[A-Za-z_-]*\s*=")
# Notes in Italian or English spelling followed by a duration (e.g., sol'8, bes4, c16).
NOTE_RE = re.compile(r"\b(?:do|re|mi|fa|sol|la|si|[a-g])[',#isbf]*\d", re.I)
FIGURE_ASSIGN_RE = re.compile(r"(?sm)^[ \t]*[\w@]+\s*=\s*\\figuremode\s*\{.*?\}\s*")
FIGURE_INLINE_RE = re.compile(r"\\figuremode\b")
VOICE_ASSIGN_RE = re.compile(r"(?m)^([IVX]{1,4}[A-Za-z0-9_-]*)\s*=")
VERSION_DECL_RE = re.compile(r"\\version\s+\"([^\"]+)\"", re.I)
LANGUAGE_DECL_RE = re.compile(r"\\language\s+\"([^\"]+)\"", re.I)
VARIABILI_INCLUDE_RE = re.compile(r"\\include\s+\"([^\"]*variabili[^\"]*)\"", re.I)
RELATIVE_LANG_RE = re.compile(r"\\relative\s+([^\s{]+)", re.I)
ITALIAN_SOLFEGE = ("do", "re", "mi", "fa", "sol", "la", "si")
DEFAULT_VERSION = "2.24.0"

# Filenames to ignore even if the heuristics would otherwise pass.
NAME_BLACKLIST = (
    "format",       # empty macro placeholders
    "header",       # includes + paper setup
    "header_part",
    "score",        # full score layout
    "violino1",
    "violino2",
    "violino3",
    "violino4",
    "viola",
    "basso",
    "violoncello",
)


def should_process(path: Path, text: str) -> bool:
    """Return True if this .ly file contains actual music definitions."""
    stem = path.stem.lower()
    for tag in NAME_BLACKLIST:
        if stem == tag or stem.endswith(f"_{tag}"):
            return False
    if not ROMAN_DEF_RE.search(text):
        return False
    if not NOTE_RE.search(text):
        return False
    return True


def normalize_file(path: Path, opts: NormOptions) -> str:
    """Run the pipeline stages and return the normalized text."""
    text = path.read_text(encoding="utf-8", errors="ignore")

    stage1 = preparse.run(text, opts)
    stage2 = normalize.run(stage1, opts)
    stage3 = engrave_strip.run(stage2, opts)
    return stage3


class _Tee:
    """Mirror writes to both the original stream and a log file."""

    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        written = self._stream.write(data)
        self._log_file.write(data)
        return written

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None)


def find_voice_blocks(text: str) -> list[tuple[str, int, int]]:
    """Return (name, start_index, end_index) slices for musical assignment blocks."""
    matches = list(VOICE_ASSIGN_RE.finditer(text))
    blocks: list[tuple[str, int, int]] = []
    for idx, match in enumerate(matches):
        name = match.group(1)
        if not name.endswith(("n", "N")):
            continue
        start = match.start()
        body_start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:end]
        if NOTE_RE.search(body):
            blocks.append((name, start, end))
    return blocks


def extract_music_assignments(text: str) -> list[str]:
    """Return assignment names that contain note material (heuristic voice detection)."""
    return [name for name, _, _ in find_voice_blocks(text)]


def _infer_language_from_music(text: str) -> str | None:
    """Heuristic: guess italiano if \\relative uses solfege tokens."""
    for match in RELATIVE_LANG_RE.finditer(text):
        token = match.group(1).strip().lower().strip(",;'\"")
        if any(token.startswith(solfege) for solfege in ITALIAN_SOLFEGE):
            return "italiano"
    return None


def _read_header_metadata(work_dir: Path) -> tuple[str | None, str | None, str | None]:
    """Extract version, language, and variabili include from sibling header files."""
    version = language = variabili_include = None
    for header in sorted(work_dir.glob("*header*.ly")):
        try:
            text = header.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if version is None:
            match = VERSION_DECL_RE.search(text)
            if match:
                version = match.group(1)
        if language is None:
            match = LANGUAGE_DECL_RE.search(text)
            if match:
                language = match.group(1)
        if variabili_include is None:
            match = VARIABILI_INCLUDE_RE.search(text)
            if match:
                variabili_include = match.group(1)
        if version and language and variabili_include:
            break
    return version, language, variabili_include


def _ensure_preamble(
    text: str,
    *,
    src_path: Path,
    header_version: str | None,
    header_language: str | None,
    variabili_include: str | None,
) -> str:
    """Guarantee that \\version, \\language and macro includes exist."""
    additions: list[str] = []

    if not VERSION_DECL_RE.search(text):
        version = header_version
        if version is None:
            var_path = src_path.parent / "variabili.ly"
            if var_path.exists():
                match = VERSION_DECL_RE.search(var_path.read_text(encoding="utf-8", errors="ignore"))
                if match:
                    version = match.group(1)
        if version is None:
            version = DEFAULT_VERSION
        additions.append(f'\\version "{version}"')

    if not LANGUAGE_DECL_RE.search(text):
        language = header_language or _infer_language_from_music(text)
        if language:
            additions.append(f'\\language "{language}"')

    if variabili_include and not VARIABILI_INCLUDE_RE.search(text):
        additions.append(f'\\include "{variabili_include}"')
    elif variabili_include is None:
        var_candidate = src_path.parent / "variabili.ly"
        if var_candidate.exists() and not VARIABILI_INCLUDE_RE.search(text):
            additions.append('\\include "variabili.ly"')

    if not additions:
        return text

    stripped = text.lstrip("\ufeff\n")
    prefix = "\n".join(additions) + "\n\n"
    return prefix + stripped


def _copy_variabili_files(input_root: Path, output_root: Path) -> None:
    """Mirror every variabili.ly file so \\include statements keep working."""
    for src in input_root.rglob("variabili.ly"):
        rel = src.relative_to(input_root)
        dest = output_root / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize a LilyPond dataset.")
    ap.add_argument("--input", required=True, help="Root directory containing the raw dataset.")
    DEFAULT_NORMALIZED_OUT = "data/normalized_dataset"
    DEFAULT_TOKENIZED_OUT = "data/tokenized_dataset"
    ap.add_argument(
        "--normalized-out",
        default=DEFAULT_NORMALIZED_OUT,
        help="Destination root for normalized .ly files (default: data/normalized_dataset).",
    )
    ap.add_argument(
        "--tokenized-out",
        default=DEFAULT_TOKENIZED_OUT,
        help="Destination root for GPT-token files (default: data/tokenized_dataset).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report which files would be processed without writing output.",
    )
    ap.add_argument(
        "--keep-figures",
        action="store_true",
        help="Preserve basso-figure (`\\figuremode`) blocks in normalized output.",
    )
    ap.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Do not produce GPT token files.",
    )
    ap.add_argument(
        "--single-voice-only",
        action="store_true",
        help="Filter outputs to files containing exactly one voice assignment.",
    )
    ap.add_argument(
        "--tokenizer-model",
        default=tokenize_gpt.DEFAULT_MODEL_NAME,
        help="HuggingFace tokenizer to use for GPT tokenization (default: EleutherAI/gpt-neox-20b).",
    )
    args = ap.parse_args()

    log_dir = Path("data/single_voice/logs" if args.single_voice_only else "data/logs").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "process_dataset.log"

    stdout_backup = sys.stdout
    stderr_backup = sys.stderr
    log_file: TextIO | None = None

    try:
        log_file = log_path.open("w", encoding="utf-8")
        sys.stdout = _Tee(stdout_backup, log_file)
        sys.stderr = _Tee(stderr_backup, log_file)

        input_root = Path(args.input).expanduser().resolve()
        if not input_root.exists():
            print(f"[dataset] input folder not found: {input_root}", file=sys.stderr)
            return 2

        normalized_out = args.normalized_out
        tokenized_out = args.tokenized_out
        if args.single_voice_only:
            single_base = Path("data/single_voice")
            if normalized_out == DEFAULT_NORMALIZED_OUT:
                normalized_out = str(single_base / "normalized_dataset")
            if tokenized_out == DEFAULT_TOKENIZED_OUT:
                tokenized_out = str(single_base / "tokenized_dataset")

        norm_root = Path(normalized_out).expanduser().resolve()
        tok_root = Path(tokenized_out).expanduser().resolve()
        opts = NormOptions()
        processed = 0
        skipped = 0
        trimmed_multi_voice = 0
        single_voice_missing = 0

        ly_files = sorted(input_root.rglob("*.ly"))
        if not ly_files:
            print(f"[dataset] no .ly files found under {input_root}")
            return 0

        for src in ly_files:
            rel = src.relative_to(input_root)
            text = src.read_text(encoding="utf-8", errors="ignore")
            if not should_process(src, text):
                skipped += 1
                continue

            print(f"[dataset] processing {rel}")

            try:
                normalized_text = normalize_file(src, opts)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[dataset] ! failed to normalize {rel}: {exc}", file=sys.stderr)
                continue

            header_version, header_language, variabili_include = _read_header_metadata(src.parent)
            normalized_text = _ensure_preamble(
                normalized_text,
                src_path=src,
                header_version=header_version,
                header_language=header_language,
                variabili_include=variabili_include,
            )

            if not args.keep_figures:
                normalized_text = FIGURE_ASSIGN_RE.sub("", normalized_text)
                normalized_text = FIGURE_INLINE_RE.sub("", normalized_text)
                normalized_text = normalized_text.replace("<figure>", "").replace("</figure>", "")

            selected_voice = None
            if args.single_voice_only:
                voice_blocks = find_voice_blocks(normalized_text)
                if not voice_blocks:
                    single_voice_missing += 1
                    print(
                        f"[dataset] ! skipped {rel} (voices=0) "
                        "due to --single-voice-only"
                    )
                    continue
                selected_voice, block_start, block_end = voice_blocks[0]
                prefix = normalized_text[:block_start]
                normalized_text = (
                    prefix + normalized_text[block_start:block_end]
                ).rstrip() + "\n"
                if len(voice_blocks) > 1:
                    trimmed_multi_voice += 1
                    print(
                        f"[dataset] ! trimmed {rel} to single voice {selected_voice} "
                        f"(removed {len(voice_blocks) - 1} voices)"
                    )

            if args.dry_run:
                processed += 1
                continue

            # Normalized output mirrors source path (.ly extension retained).
            norm_path = norm_root / rel
            norm_path.parent.mkdir(parents=True, exist_ok=True)
            norm_path.write_text(normalized_text.rstrip() + "\n", encoding="utf-8")

            norm_path = norm_root / rel
            norm_path.parent.mkdir(parents=True, exist_ok=True)
            norm_path.write_text(normalized_text.rstrip() + "\n", encoding="utf-8")

            if not args.skip_tokenize:
                tok_ids = tokenize_gpt.run(normalized_text, model_name=args.tokenizer_model)
                tok_path = tok_root / rel.with_suffix(".tokens.json")
                tok_path.parent.mkdir(parents=True, exist_ok=True)
                tok_path.write_text(json.dumps({"input_ids": tok_ids}) + "\n", encoding="utf-8")

            processed += 1

        if not args.dry_run:
            _copy_variabili_files(input_root, norm_root)

        print(
            f"[dataset] done. processed={processed} skipped={skipped} "
            f"normalized_out={norm_root}"
            + ("" if args.skip_tokenize else f" tokenized_out={tok_root}")
            + (
                ""
                if not args.single_voice_only
                else f" single_voice_trimmed={trimmed_multi_voice} single_voice_skipped={single_voice_missing}"
            )
        )
        return 0
    finally:
        if log_file is not None:
            log_file.flush()
            log_file.close()
        sys.stdout = stdout_backup
        sys.stderr = stderr_backup


if __name__ == "__main__":
    raise SystemExit(main())
