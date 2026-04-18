"""Run file_resolver and preprocess on raw files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Add src to path so this script can run from the repo root.
repo_root = Path(__file__).resolve().parents[2]
src_dir = repo_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

from lilybench.stages.normalization import file_resolver, preprocess
from lilybench.utils.options import NormOptions

# Same blacklist as process_dataset.py.
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


def should_process(path: Path, text: str) -> bool:
    """Return True for score-like LilyPond files worth preprocessing."""
    stem = path.stem.lower()

    for tag in NAME_BLACKLIST:
        if stem == tag or stem.endswith(f"_{tag}"):
            return False

    if "score" not in stem:
        return False

    if not re.search(r"\\version|\\language", text):
        return False

    return True


def main():
    """Process raw data through preprocess and write outputs."""
    input_root = Path("data/raw").resolve()
    output_root = Path("data/test_preprocess").resolve()

    if not input_root.exists():
        print(f"Error: Input folder not found: {input_root}", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    opts = NormOptions()
    processed = 0

    ly_files = sorted(input_root.rglob("*.ly"))
    print(f"Found {len(ly_files)} .ly files")
    print()

    for src in ly_files:
        rel = src.relative_to(input_root)
        text = src.read_text(encoding="utf-8", errors="ignore")

        if not should_process(src, text):
            continue

        print(f"[{processed + 1}] Processing: {rel}")

        try:
            stage0_pieces = file_resolver.run(src, exclude_variabili=False)

            pieces = []
            for piece in stage0_pieces:
                stage1 = preprocess.run(piece, opts)
                pieces.append(stage1)

            for idx, piece in enumerate(pieces, start=1):
                out_path = output_root / rel
                if len(pieces) > 1:
                    out_path = out_path.with_stem(out_path.stem + f"_part{idx}")

                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(piece, encoding="utf-8")

            processed += 1

        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue

    print()
    print(f"=== Processed {processed}/{len(ly_files)} files ===")
    print(f"Output saved to: {output_root}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
