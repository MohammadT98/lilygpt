"""Run file_resolver, preprocess, and normalize on raw files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Add src to path so this script can run from the repo root.
repo_root = Path(__file__).resolve().parents[2]
src_dir = repo_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

from lilynorm.stages.normalization import file_resolver, preprocess, normalize_syntax as expand_module
from lilynorm.utils.options import NormOptions

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
    """Return True for score-like LilyPond files worth processing."""
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
    """Process raw data through normalization and write outputs."""
    input_root = Path("data/raw").resolve()
    output_root = Path("data/test_normalize").resolve()

    if not input_root.exists():
        print(f"Error: Input folder not found: {input_root}", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    opts = NormOptions()
    processed = 0

    stats = {
        "line_removed": 0,
        "block_removed": 0,
        "rel": 0,
        "vars": 0,
        "transpose_ok": 0,
        "repeat": 0,
        "tuplets": 0,
        "drums": 0,
        "lily_fail": 0,
    }

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
            for stage0 in stage0_pieces:
                stage1 = preprocess.run(stage0, opts)

                if len(stage1.splitlines()) < len(stage0.splitlines()):
                    stats["line_removed"] += 1
                if stage1.count("{") < stage0.count("{"):
                    stats["block_removed"] += 1

                stage2 = expand_module.run(stage1, opts)

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

                pieces.append(stage2)

            for idx, piece in enumerate(pieces, start=1):
                out_path = output_root / rel
                if len(pieces) > 1:
                    out_path = out_path.with_stem(out_path.stem + f"_part{idx}")

                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(piece, encoding="utf-8")

            processed += 1

        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            stats["lily_fail"] += 1
            continue

    print()
    print(f"=== Processed {processed}/{len(ly_files)} files ===")
    print(f"Output saved to: {output_root}")
    print()
    print(f"[preprocess] line_removed={stats['line_removed']} block_removed={stats['block_removed']}")
    print(f"[normalize] rel:{stats['rel']} vars:{stats['vars']} transpose_ok:{stats['transpose_ok']} repeat:{stats['repeat']} tuplets:{stats['tuplets']} drums:{stats['drums']} lily_fail:{stats['lily_fail']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
