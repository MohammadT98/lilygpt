"""
Minimal processing: file_resolver + preprocess only.
Outputs to data/test_preprocess/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
src_dir = repo_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

from lilynorm.stages.normalization import file_resolver, preprocess
from lilynorm.utils.options import NormOptions

# Same blacklist as process_dataset.py
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
    """Return True if this .ly file contains actual music definitions."""
    stem = path.stem.lower()
    
    for tag in NAME_BLACKLIST:
        if stem == tag or stem.endswith(f"_{tag}"):
            return False
    
    # Only process files that have "score" in the name
    if "score" not in stem:
        return False
    
    # Must have version declaration or at least one note
    import re
    if not re.search(r"\\version|\\language", text):
        return False
    
    return True


def main():
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
        
        # Skip blacklisted files
        if not should_process(src, text):
            continue
        
        print(f"[{processed + 1}] Processing: {rel}")
        
        try:
            # Stage 0: File resolver (returns list of strings)
            stage0_pieces = file_resolver.run(src, exclude_variabili=False)

            # Stage 1: Preprocess each piece
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
