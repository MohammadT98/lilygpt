from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import TextIO

# Prefer local source tree when running from the repo (avoid stale installed package).
_repo_root = Path(__file__).resolve().parents[1]
_src_dir = _repo_root / "src"
if _src_dir.exists():
    sys.path.insert(0, str(_src_dir))

try:
    from lilynorm.utils.options import NormOptions
    from lilynorm.stages import preprocessing
    from lilynorm.stages import tokenization as tokenize_gpt
    from lilynorm.stages.preprocessing.file_resolver import split_on_multiple_forma
    from lilynorm.utils.formatting import format_full_text, format_example
except ModuleNotFoundError:
    # Allow running the script directly from the repo without installing the package.
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
    from lilynorm.utils.options import NormOptions
    from lilynorm.stages import preprocessing
    from lilynorm.stages import tokenization as tokenize_gpt
    from lilynorm.stages.splitting import build_splits
    from lilynorm.utils.formatting import format_full_text, format_example


# ---------------------------------------------------------------------------
# Regexes and constants
# ---------------------------------------------------------------------------

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
# Note: "score" is intentionally NOT blacklisted; score files are processed and file_resolver
# will inline all includes (headers, parts, etc.) to create complete standalone documents.
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


DEFAULT_NORMALIZED_OUT = "data/normalized_dataset"
DEFAULT_TOKENIZED_OUT = "data/tokenized_dataset"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def should_process(path: Path, text: str) -> bool:
    """Return True if this .ly file contains actual music definitions."""
    stem = path.stem.lower()

    for tag in NAME_BLACKLIST:
        if stem == tag or stem.endswith(f"_{tag}"):
            return False

    # Score files are explicitly allowed (file resolution will inline all includes)
    if stem.endswith("_score") or stem == "score":
        return True

    # All other files (movement files, parts, etc.) are skipped.
    # They will be inlined into the score via file_resolver.
    return False


def normalize_file(path: Path, opts: NormOptions, stats: dict[str, int] | None = None) -> str:
    """Run the pipeline stages and return the normalized text.

    If a stats dict is provided, it will be updated with simple counters for
    preparse and normalize phases.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Stage 0: Resolve all \include statements including variabili.ly
    # (inline everything for training; we need complete, standalone files)
    stage0 = preprocessing.file_resolver.run(text, path, exclude_variabili=False)
    
    # Stage 1: Preparse
    stage1 = preprocessing.preparse.run(stage0, opts)
    if stats is not None:
        if len(stage1.splitlines()) < len(stage0.splitlines()):
            stats["line_removed"] += 1
        if stage1.count("{") < stage0.count("{"):
            stats["block_removed"] += 1

    # Stage 2: Normalize
    stage2 = preprocessing.normalize.run(stage1, opts)
    if stats is not None:
        if "\\relative" in stage2:
            stats["rel"] += 1
        if "=" in stage2 and "{" in stage2:
            stats["vars"] += 1
        if "\\transpose" in stage2:
            stats["transpose_ok"] += 1
        if "\\repeat" in stage2:
            stats["repeat"] += 1
        if "\\tuplet" in stage2 or "\\times" in stage2:
            stats["tuplets"] += 1
        if "\\drums" in stage2 or "DrumStaff" in stage2:
            stats["drums"] += 1

    # Stage 3: Strip engraving directives
    stage3 = preprocessing.engrave_strip.run(stage2, opts)

    if stats is not None:
        # Simple heuristic counts after stripping
        stats["overrides"] += stage3.count("\\override")
        stats["markups"] += stage3.count("\\markup")
        stats["marks"] += stage3.count("\\mark")
        stats["dynamics"] += sum(stage3.count(tok) for tok in ["\\pp", "\\p", "\\mp", "\\mf", "\\f", "\\ff", "\\fp", "\\sfz"])
        stats["hairpins"] += stage3.count("\\<") + stage3.count("\\>")
        stats["quotes"] += stage3.count("\\quote")

    # Final cleanup now handled inside engraving stage
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
        # Voices are heuristically detected by names ending in 'n' / 'N'
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


def _ensure_preamble(
    text: str,
    *,
    src_path: Path,
    header_version: str | None,
    header_language: str | None,
    variabili_include: str | None,
) -> str:
    """Guarantee that \\version, \\language and macro includes exist.

    For ML/fine-tuning, remove all LilyPond directives and do not add any preamble.
    Currently this function is intentionally a no-op to preserve the existing behavior.
    """
    return text


def _copy_variabili_files(input_root: Path, output_root: Path) -> None:
    """Mirror every variabili.ly file so \\include statements keep working."""
    for src in input_root.rglob("variabili.ly"):
        rel = src.relative_to(input_root)
        dest = output_root / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


# ---------------------------------------------------------------------------
# CLI construction
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for this script."""
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


# ---------------------------------------------------------------------------
# Main dataset processing
# ---------------------------------------------------------------------------

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

        # Statistics counters (aligned with test_normalize.py)
        stats: dict[str, int] = {
            "line_removed": 0,
            "block_removed": 0,
            "rel": 0,
            "vars": 0,
            "transpose_ok": 0,
            "repeat": 0,
            "tuplets": 0,
            "drums": 0,
            "lily_fail": 0,
            "overrides": 0,
            "markups": 0,
            "marks": 0,
            "dynamics": 0,
            "hairpins": 0,
            "quotes": 0,
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
                normalized_text = normalize_file(src, opts, stats)
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
            normalized_text = _ensure_preamble(
                normalized_text,
                src_path=src,
                header_version=header_version,
                header_language=header_language,
                variabili_include=variabili_include,
            )

            # Strip figuremode if requested
            # DISABLED: Stripping figured bass causes broken references when definitions
            # are removed but usages remain (e.g., \Ibfn undefined after Ibfn = \figuremode {...} is stripped)
            # For ML training, figured bass should be preserved as it contains harmonic information.
            # if not args.keep_figures and not args.keep_engraving:
            #     normalized_text = FIGURE_ASSIGN_RE.sub("", normalized_text)
            #     normalized_text = FIGURE_INLINE_RE.sub("", normalized_text)
            #     normalized_text = normalized_text.replace("<figure>", "").replace(
            #         "</figure>", ""
            #     )

            selected_voice = None

            # Optionally keep only a single voice
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

            # Split files that embed multiple forma blocks into separate pieces
            forma_pieces = split_on_multiple_forma(normalized_text)

            if args.dry_run:
                processed += 1
                continue

            removed_empty_total = 0
            removed_empty_parts = 0
            for idx, piece in enumerate(forma_pieces, start=1):
                norm_path = norm_root / rel
                if len(forma_pieces) > 1:
                    norm_path = norm_path.with_stem(norm_path.stem + f"_part{idx}")

                norm_path.parent.mkdir(parents=True, exist_ok=True)

                # Remove existing \version declarations, then add \version "2.24.4"
                cleaned = re.sub(r'(^|\n)\\version\s+"[^"]+"\s*', "", piece)

                # Remove stray caret articulations that break parsing (e.g., mi^ la, dod^ \n mi)
                note_token = r"(?:do|re|mi|fa|sol|la|si|[a-gr])[a-z]*"
                cleaned = re.sub(
                    rf"(\b{note_token}[',]*\d?)\^\s+(?={note_token})",
                    r"\1 ",
                    cleaned,
                    flags=re.MULTILINE,
                )
                cleaned = re.sub(
                    rf"(\b{note_token}[',]*\d?)\^(?=\s*{note_token})",
                    r"\1",
                    cleaned,
                    flags=re.MULTILINE,
                )
                cleaned = re.sub(
                    rf"(\b{note_token}[',]*\d?)\^\s*$",
                    r"\1",
                    cleaned,
                    flags=re.MULTILINE,
                )
                # Fix malformed durations like la168 -> la16 8 (two glued durations)
                cleaned = re.sub(
                    rf"(\b{note_token}[',]*)(128|64|32|16)([1248])\b",
                    r"\1\2 \3",
                    cleaned,
                )
                # Remove malformed \tempo directives (break LilyPond parsing)
                cleaned = re.sub(r"(?m)^\s*\\tempo\s+.*$", "", cleaned)
                # Drop bare \mark lines and \mark"..." labels (engraving-only)
                cleaned = re.sub(r'(?m)^\s*\\mark\s*"?[^"\n]*"?\s*$', "", cleaned)
                # Remove inline empty text blocks like {" "}
                cleaned = re.sub(r'\{\s*"\s*"\s*\}', "", cleaned)
                # Remove broken polyphonic openings (<< without >> on the same line)
                cleaned = re.sub(r"(?m)^\s*<<[^>]*$", "", cleaned)
                # Remove stray ornament tokens like [tr]
                cleaned = re.sub(r"\[\s*tr\s*\]", "", cleaned)

                # Split glued solfege note names (e.g., faddod, remi2).
                solfege = r"(?:dod|red|mid|fad|sold|lad|sid|do|re|mi|fa|sol|la|si)"
                cleaned = re.sub(
                    rf"\b({solfege})([',]*?)({solfege})([',]*\d*)\b",
                    r"\1\2 \3\4",
                    cleaned,
                )

                # Close any open blocks before top-level assignments.
                assign_re = re.compile(r"^[A-Za-z_][\w-]*\s*=")
                lines = cleaned.splitlines()
                out: list[str] = []
                depth = 0
                for line in lines:
                    if assign_re.match(line) and depth > 0:
                        out.extend(["}"] * depth)
                        depth = 0
                    out.append(line)
                    line_no_comment = line.split("%", 1)[0]
                    depth += line_no_comment.count("{") - line_no_comment.count("}")
                    if depth < 0:
                        depth = 0
                cleaned = "\n".join(out) + ("\n" if cleaned.endswith("\n") else "")
                
                # Final cleanup: Remove any empty variable assignments that might have been created
                # during file splitting or other post-processing steps
                from lilynorm.stages.preprocessing.engrave_strip import _remove_empty_variable_assignments
                cleaned, empty_count = _remove_empty_variable_assignments(cleaned)
                if empty_count > 0:
                    removed_empty_total += empty_count
                    removed_empty_parts += 1
                
                output_text = '\\version "2.24.4"\n' + cleaned.lstrip() + "\n"
                norm_path.write_text(output_text, encoding="utf-8")

            if removed_empty_total > 0:
                part_note = f" parts={removed_empty_parts}" if removed_empty_parts > 1 else ""
                print(
                    f"[dataset] removed {removed_empty_total} empty variable assignment(s) from {rel}{part_note}",
                    file=sys.stderr,
                )

            # Tokenization output
            if not args.skip_tokenize:
                # Apply instruction formatting if requested
                text_to_tokenize = normalized_text
                prompt_preview = ""
                prefix_token_count = 0
                if args.instruction_format != "none":
                    # Build full text
                    text_to_tokenize = format_full_text(
                        normalized_text,
                        instruction_format=args.instruction_format,
                        instruction=args.instruction,
                    )
                    # Also compute a prompt preview and token length of the prompt
                    formatted = format_example(
                        normalized_text,
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

                tok_path = tok_root / rel.with_suffix(".tokens.json")
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
            f"[normalize] rel:{stats['rel']} vars:{stats['vars']} "
            f"transpose_ok:{stats['transpose_ok']} repeat:{stats['repeat']} "
            f"tuplets:{stats['tuplets']} drums:{stats['drums']} "
            f"lily_fail:{stats['lily_fail']}"
        )
        print(
            f"[engrave_strip] overrides:{stats['overrides']} markups:{stats['markups']} "
            f"marks:{stats['marks']} dynamics:{stats['dynamics']} "
            f"hairpins:{stats['hairpins']} quotes:{stats['quotes']}"
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
