"""
CLI for processing LilyPond datasets.

Usage:
    python -m lilynorm.cli --input data/raw --normalized-out data/normalized_dataset --skip-tokenize
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import TextIO

# Prefer local source tree when running from the repo (avoid stale installed package).
# This file is at src/lilynorm/cli.py, so go up 3 levels to get repo root
_repo_root = Path(__file__).resolve().parents[2]
_src_dir = _repo_root / "src"
if _src_dir.exists():
    sys.path.insert(0, str(_src_dir))

from lilynorm.utils.options import NormOptions
from lilynorm.normalize import normalize_file
from lilynorm.stages import tokenization as tokenize_gpt
from lilynorm.stages.dataset.voice_extraction import find_voice_blocks
from lilynorm.utils.formatting import format_full_text, format_example


# Filenames to skip during processing
NAME_BLACKLIST = (
    "format",       # empty macro placeholders
    "header",       # includes + paper setup
    "header_part",
    "variabili",    # variable definitions and engraving macros (kept as includes in resolved files)
    "violino1",
    "violino2",
    "violino3",
    "violino4",
    "viola",
    "basso",
    "violoncello",
)

ITALIAN_SOLFEGE = ("do", "re", "mi", "fa", "sol", "la", "si")
DEFAULT_VERSION = "2.24.0"
DEFAULT_NORMALIZED_OUT = "data/normalized_dataset"
DEFAULT_TOKENIZED_OUT = "data/tokenized_dataset"

VERSION_DECL_RE = re.compile(r"\\version\s+\"([^\"]+)\"", re.I)
LANGUAGE_DECL_RE = re.compile(r"\\language\s+\"([^\"]+)\"", re.I)
VARIABILI_INCLUDE_RE = re.compile(r"\\include\s+\"([^\"]*variabili[^\"]*)\"", re.I)
RELATIVE_LANG_RE = re.compile(r"\\relative\s+([^\s{]+)", re.I)

class _Tee:
    """Mirror writes to both stdout and a log file."""
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


def should_process(path: Path, text: str) -> bool:
    """Check if this .ly file should be processed."""
    stem = path.stem.lower()

    for tag in NAME_BLACKLIST:
        if stem == tag or stem.endswith(f"_{tag}"):
            return False

    # Score files get processed (file_resolver inlines all includes)
    if stem.endswith("_score") or stem == "score":
        return True

    return False


def _infer_language_from_music(text: str) -> str | None:
    """Infer LilyPond language from musical content (italiano vs english)."""
    for match in RELATIVE_LANG_RE.finditer(text):
        token = match.group(1).strip().lower().strip(",;'\"")
        if any(token.startswith(solfege) for solfege in ITALIAN_SOLFEGE):
            return "italiano"
    return None


def _read_header_metadata(work_dir: Path) -> tuple[str | None, str | None, str | None]:
    """Extract version, language, and variabili include from header files."""
    version: str | None = None
    language: str | None = None
    variabili_include: str | None = None

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


def _copy_variabili_files(input_root: Path, output_root: Path) -> None:
    """Mirror variabili.ly files to output to keep \\include statements working."""
    for src in input_root.rglob("variabili.ly"):
        rel = src.relative_to(input_root)
        dest = output_root / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize a LilyPond dataset.")

    parser.add_argument(
        "--input",
        required=True,
        help="Root directory containing the raw dataset.",
    )
    parser.add_argument(
        "--normalized-out",
        default=DEFAULT_NORMALIZED_OUT,
        help="Destination root for normalized .ly files "
             "(default: data/normalized_dataset).",
    )
    parser.add_argument(
        "--tokenized-out",
        default=DEFAULT_TOKENIZED_OUT,
        help="Destination root for GPT-token files "
             "(default: data/tokenized_dataset).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report which files would be processed without writing output.",
    )
    parser.add_argument(
        "--keep-figures",
        action="store_true",
        help="Preserve basso-figure (`\\figuremode`) blocks in normalized output.",
    )
    parser.add_argument(
        "--keep-compilable",
        action="store_true",
        help="Keep files compilable with layout/midi/paper blocks. By default, files are stripped for ML training.",
    )
    parser.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Do not produce GPT token files.",
    )
    parser.add_argument(
        "--mirror-variabili",
        action="store_true",
        help=(
            "Mirror variabili.ly files into normalized output so \\include continues to work. "
            "By default variabili.ly is NOT mirrored to keep training data clean."
        ),
    )
    parser.add_argument(
        "--single-voice-only",
        action="store_true",
        help="Filter outputs to files containing exactly one voice assignment.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=tokenize_gpt.DEFAULT_MODEL_NAME,
        help=(
            "HuggingFace tokenizer to use for GPT tokenization "
            "(default: openai/gpt-oss-20b)."
        ),
    )
    parser.add_argument(
        "--instruction-format",
        choices=["none", "plain", "chatml"],
        default="chatml",
        help=(
            "Wrap normalized music with instruction prompts for fine-tuning. "
            "'none' = no wrapping (baseline); "
            "'plain' = simple text prefix; "
            "'chatml' = ChatML format with <|user|>/<|assistant|> tokens. "
            "(default: chatml)"
        ),
    )
    parser.add_argument(
        "--instruction",
        default="Generate LilyPond music notation.",
        help="Instruction text to prepend when using --instruction-format. (default: 'Generate LilyPond music notation.')",
    )

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    log_dir = Path(
        "data/single_voice/logs" if args.single_voice_only else "data/logs"
    ).expanduser()
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

        opts = NormOptions(keep_engraving=args.keep_compilable)

        processed = 0
        skipped = 0
        trimmed_multi_voice = 0
        single_voice_missing = 0

        # Statistics counters
        stats: dict[str, int] = {
            "line_removed": 0,
            "block_removed": 0,
            "rel_removed": 0,
            "vars_removed": 0,
            "transpose_removed": 0,
            "repeat_removed": 0,
            "tuplets_removed": 0,
            "lily_fail": 0,
            "overrides_removed": 0,
            "markups_removed": 0,
            "marks_removed": 0,
            "dynamics_removed": 0,
            "hairpins_removed": 0,
            "quotes_removed": 0,
        }

        ly_files = sorted(input_root.rglob("*.ly"))
        if not ly_files:
            print(f"[dataset] no .ly files found under {input_root}")
            return 0

        for src in ly_files:
            # Skip macOS metadata artifacts.
            if "__MACOSX" in src.parts or src.name.startswith("._"):
                skipped += 1
                continue
            rel = src.relative_to(input_root)
            text = src.read_text(encoding="utf-8", errors="ignore")

            if not should_process(src, text):
                skipped += 1
                continue

            print(f"[dataset] processing {rel}")

            try:
                # Call the pipeline module to normalize the file
                forma_pieces = normalize_file(src, opts, stats)
            except Exception as exc:  # pragma: no cover - defensive
                print(
                    f"[dataset] ! failed to normalize {rel}: {exc}",
                    file=sys.stderr,
                )
                stats["lily_fail"] += 1
                continue

            header_version, header_language, variabili_include = _read_header_metadata(
                src.parent
            )

            # Process each piece (single-voice filtering if needed)
            processed_pieces = []
            for piece in forma_pieces:
                # Optionally keep only a single voice
                if args.single_voice_only:
                    voice_blocks = find_voice_blocks(piece)
                    if not voice_blocks:
                        single_voice_missing += 1
                        print(
                            f"[dataset] ! skipped {rel} (voices=0) "
                            "due to --single-voice-only"
                        )
                        continue

                    selected_voice, block_start, block_end = voice_blocks[0]
                    prefix = piece[:block_start]
                    piece = (
                        prefix + piece[block_start:block_end]
                    ).rstrip() + "\n"

                    if len(voice_blocks) > 1:
                        trimmed_multi_voice += 1
                        print(
                            f"[dataset] ! trimmed {rel} to single voice {selected_voice} "
                            f"(removed {len(voice_blocks) - 1} voices)"
                        )

                processed_pieces.append(piece)

            # Use the processed pieces (already split by forma in pipeline Stage 0)
            forma_pieces = processed_pieces

            if args.dry_run:
                processed += 1
                continue

            # Write normalized pieces and tokenize (already fully processed by the pipeline)
            for idx, piece in enumerate(forma_pieces, start=1):
                norm_path = norm_root / rel
                if len(forma_pieces) > 1:
                    norm_path = norm_path.with_stem(norm_path.stem + f"_part{idx}")

                norm_path.parent.mkdir(parents=True, exist_ok=True)

                output_text = piece.lstrip() + "\n"
                norm_path.write_text(output_text, encoding="utf-8")

                # Tokenization output
                if not args.skip_tokenize:
                    # Apply instruction formatting if requested
                    text_to_tokenize = piece
                    prompt_preview = ""
                    prefix_token_count = 0
                    if args.instruction_format != "none":
                        # Build full text
                        text_to_tokenize = format_full_text(
                            piece,
                            instruction_format=args.instruction_format,
                            instruction=args.instruction,
                        )
                        # Also compute a prompt preview and token length of the prompt
                        formatted = format_example(
                            piece,
                            instruction_format=args.instruction_format,
                            instruction=args.instruction,
                        )
                        if isinstance(formatted, tuple):
                            prompt_preview = formatted[0]
                            # Tokenize prompt to measure its token length
                            prompt_tok = tokenize_gpt.tokenize_gpt.run(
                                prompt_preview,
                                model_name=args.tokenizer_model,
                            )
                            prefix_token_count = len(prompt_tok.get("input_ids", []))
                        else:
                            # plain: prompt is just the instruction + newline
                            prompt_preview = args.instruction + "\n"
                            prompt_tok = tokenize_gpt.tokenize_gpt.run(
                                prompt_preview,
                                model_name=args.tokenizer_model,
                            )
                            prefix_token_count = len(prompt_tok.get("input_ids", []))

                    tok_info = tokenize_gpt.tokenize_gpt.run(
                        text_to_tokenize,
                        model_name=args.tokenizer_model,
                        # max_length=1024  # override here if you want
                    )
                    # Attach instruction metadata for verification
                    tok_info["instruction_format"] = args.instruction_format
                    tok_info["instruction"] = (
                        args.instruction if args.instruction_format != "none" else None
                    )
                    tok_info["prompt_preview"] = prompt_preview
                    tok_info["prefix_token_count"] = prefix_token_count

                    tok_path = tok_root / norm_path.name
                    tok_path = tok_path.with_suffix(".tokens.json")
                    tok_path.parent.mkdir(parents=True, exist_ok=True)
                    tok_path.write_text(
                        json.dumps(tok_info) + "\n",
                        encoding="utf-8",
                    )

            processed += 1

        # Default: do NOT mirror variabili.ly into normalized output.
        # Only mirror when explicitly requested via --mirror-variabili.
        if not args.dry_run and args.mirror_variabili:
            _copy_variabili_files(input_root, norm_root)

        print(
            f"[dataset] done. processed={processed} skipped={skipped} "
            f"normalized_out={norm_root}"
            + ("" if args.skip_tokenize else f" tokenized_out={tok_root}")
            + (
                ""
                if not args.single_voice_only
                else f" single_voice_trimmed={trimmed_multi_voice} "
                     f"single_voice_skipped={single_voice_missing}"
            )
        )

        # Summary of preprocessing/normalize stats
        print("--- Stage summaries ---")
        print(
            f"[preparse] line_removed={stats['line_removed']} "
            f"block_removed={stats['block_removed']}"
        )
        print(
            f"[normalize] rel_removed:{stats['rel_removed']} vars_removed:{stats['vars_removed']} "
            f"transpose_removed:{stats['transpose_removed']} repeat_removed:{stats['repeat_removed']} "
            f"tuplets_removed:{stats['tuplets_removed']} "
            f"lily_fail:{stats['lily_fail']}"
        )
        print(
            f"[engrave_strip] overrides_removed:{stats['overrides_removed']} "
            f"markups_removed:{stats['markups_removed']} marks_removed:{stats['marks_removed']} "
            f"dynamics_removed:{stats['dynamics_removed']} "
            f"hairpins_removed:{stats['hairpins_removed']} quotes_removed:{stats['quotes_removed']}"
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
