from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import TextIO

_repo_root = Path(__file__).resolve().parents[2]
_src_dir = _repo_root / "src"
if _src_dir.exists():
    sys.path.insert(0, str(_src_dir))

from lilynorm.utils.options import NormOptions
from lilynorm.normalize import normalize_file
from lilynorm.stages import tokenization as tokenize_gpt
from lilynorm.utils.formatting import format_full_text, format_example


NAME_BLACKLIST = (
    "format",
    "header",
    "header_part",
    "variabili",
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

class _Tee:
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
    stem = path.stem.lower()

    for tag in NAME_BLACKLIST:
        if stem == tag or stem.endswith(f"_{tag}"):
            return False

    if stem.endswith("_score") or stem == "score":
        return True

    return False


def _copy_variabili_files(input_root: Path, output_root: Path) -> None:
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

    log_dir = Path("data/logs").expanduser()
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

        norm_root = Path(args.normalized_out).expanduser().resolve()
        tok_root = Path(args.tokenized_out).expanduser().resolve()

        opts = NormOptions(keep_engraving=args.keep_compilable)

        processed = 0
        skipped = 0

        stats: dict[str, int] = {
            "line_removed": 0,
            "block_removed": 0,
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
                forma_pieces = normalize_file(src, opts, stats)
            except Exception as exc:
                print(
                    f"[dataset] ! failed to normalize {rel}: {exc}",
                    file=sys.stderr,
                )
                stats["lily_fail"] += 1
                continue

            if args.dry_run:
                processed += 1
                continue

            for idx, piece in enumerate(forma_pieces, start=1):
                norm_path = norm_root / rel
                if len(forma_pieces) > 1:
                    norm_path = norm_path.with_stem(norm_path.stem + f"_part{idx}")

                norm_path.parent.mkdir(parents=True, exist_ok=True)

                output_text = piece.lstrip() + "\n"
                norm_path.write_text(output_text, encoding="utf-8")

                if not args.skip_tokenize:
                    text_to_tokenize = piece
                    prompt_preview = ""
                    prefix_token_count = 0
                    if args.instruction_format != "none":
                        text_to_tokenize = format_full_text(
                            piece,
                            instruction_format=args.instruction_format,
                            instruction=args.instruction,
                        )
                        formatted = format_example(
                            piece,
                            instruction_format=args.instruction_format,
                            instruction=args.instruction,
                        )
                        if isinstance(formatted, tuple):
                            prompt_preview = formatted[0]
                            prompt_tok = tokenize_gpt.tokenize_gpt.run(
                                prompt_preview,
                                model_name=args.tokenizer_model,
                            )
                            prefix_token_count = len(prompt_tok.get("input_ids", []))
                        else:
                            prompt_preview = args.instruction + "\n"
                            prompt_tok = tokenize_gpt.tokenize_gpt.run(
                                prompt_preview,
                                model_name=args.tokenizer_model,
                            )
                            prefix_token_count = len(prompt_tok.get("input_ids", []))

                    tok_info = tokenize_gpt.tokenize_gpt.run(
                        text_to_tokenize,
                        model_name=args.tokenizer_model,
                    )
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

        if not args.dry_run and args.mirror_variabili:
            _copy_variabili_files(input_root, norm_root)

        print(
            f"[dataset] done. processed={processed} skipped={skipped} "
            f"normalized_out={norm_root}"
            + ("" if args.skip_tokenize else f" tokenized_out={tok_root}")
        )

        print("--- Stage summaries ---")
        print(
            f"[preparse] line_removed={stats['line_removed']} "
            f"block_removed={stats['block_removed']}"
        )
        print(
            f"[normalize] vars_removed:{stats['vars_removed']} "
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
