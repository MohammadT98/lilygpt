from __future__ import annotations

"""CLI entry point for normalizing LilyPond datasets."""

import argparse
import sys
from pathlib import Path
from typing import TextIO

_repo_root = Path(__file__).resolve().parents[2]
_src_dir = _repo_root / "src"
if _src_dir.exists():
    sys.path.insert(0, str(_src_dir))

from lilynorm.normalize import normalize_file
from lilynorm.utils.options import NormOptions

DEFAULT_NORMALIZED_OUT = "data/normalized_dataset"
LOG_DIR = Path("data/logs")
LOG_FILENAME = "process_dataset.log"

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


class _Tee:
    """Write output to a stream and a log file simultaneously."""

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
    def encoding(self) -> str | None:
        return getattr(self._stream, "encoding", None)


def should_process(path: Path, text: str) -> bool:
    """Return True for score files that should be normalized."""
    stem = path.stem.lower()
    for tag in NAME_BLACKLIST:
        if stem == tag or stem.endswith(f"_{tag}"):
            return False

    if stem.endswith("_score") or stem == "score":
        return True

    return False


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _init_stats() -> dict[str, int]:
    return {
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


def _print_stage_summaries(stats: dict[str, int]) -> None:
    print("--- Stage summaries ---")
    print(
        f"[preprocess] line_removed={stats['line_removed']} "
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for dataset normalization."""
    parser = argparse.ArgumentParser(description="Normalize a LilyPond dataset.")

    parser.add_argument(
        "--input",
        required=True,
        help="Root directory containing the raw dataset.",
    )
    parser.add_argument(
        "--normalized-out",
        default=DEFAULT_NORMALIZED_OUT,
        help=(
            "Destination root for normalized .ly files "
            "(default: data/normalized_dataset)."
        ),
    )

    return parser


def main() -> int:
    """Normalize a dataset and emit a processing log."""
    args = build_arg_parser().parse_args()

    log_dir = LOG_DIR.expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILENAME

    stdout_backup = sys.stdout
    stderr_backup = sys.stderr
    log_file: TextIO | None = None

    try:
        log_file = log_path.open("w", encoding="utf-8")
        sys.stdout = _Tee(stdout_backup, log_file)
        sys.stderr = _Tee(stderr_backup, log_file)

        input_root = _resolve_path(args.input)
        if not input_root.exists():
            print(f"[dataset] input folder not found: {input_root}", file=sys.stderr)
            return 2

        norm_root = _resolve_path(args.normalized_out)
        opts = NormOptions()

        processed = 0
        skipped = 0
        stats = _init_stats()

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

            for idx, piece in enumerate(forma_pieces, start=1):
                norm_path = norm_root / rel
                if len(forma_pieces) > 1:
                    norm_path = norm_path.with_stem(norm_path.stem + f"_part{idx}")

                norm_path.parent.mkdir(parents=True, exist_ok=True)

                output_text = piece.lstrip() + "\n"
                norm_path.write_text(output_text, encoding="utf-8")

            processed += 1

        print(
            f"[dataset] done. processed={processed} skipped={skipped} "
            f"normalized_out={norm_root}"
        )

        _print_stage_summaries(stats)
        return 0

    finally:
        if log_file is not None:
            log_file.flush()
            log_file.close()
        sys.stdout = stdout_backup
        sys.stderr = stderr_backup


if __name__ == "__main__":
    raise SystemExit(main())
