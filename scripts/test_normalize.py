"""
Test processing: file_resolver + preparse + normalize.
Outputs to data/test_normalize/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

from lilynorm.stages.preprocessing import file_resolver, preparse, normalize as norm_module
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
    output_root = Path("data/test_normalize").resolve()
    
    if not input_root.exists():
        print(f"Error: Input folder not found: {input_root}", file=sys.stderr)
        return 1
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    opts = NormOptions()
    processed = 0
    
    # Statistics counters
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
        
        # Skip blacklisted files
        if not should_process(src, text):
            continue
        
        print(f"[{processed + 1}] Processing: {rel}")
        
        try:
            # Stage 0: File resolver
            stage0 = file_resolver.run(text, src, exclude_variabili=False)
            
            # Stage 1: Preparse
            stage1 = preparse.run(stage0, opts)
            
            # Track preparse changes
            if len(stage1.splitlines()) < len(stage0.splitlines()):
                stats["line_removed"] += 1
            if stage1.count("{") < stage0.count("{"):
                stats["block_removed"] += 1
            
            # Stage 2: Normalize
            stage2 = norm_module.run(stage1, opts)
            
            # Track normalize features (heuristic detection)
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
            
            # Save output
            out_path = output_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(stage2, encoding="utf-8")
            
            processed += 1
            
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            stats["lily_fail"] += 1
            continue
    
    print()
    print(f"=== Processed {processed}/{len(ly_files)} files ===")
    print(f"Output saved to: {output_root}")
    print()
    print(f"[preparse] line_removed={stats['line_removed']} block_removed={stats['block_removed']}")
    print(f"[normalize] rel:{stats['rel']} vars:{stats['vars']} transpose_ok:{stats['transpose_ok']} repeat:{stats['repeat']} tuplets:{stats['tuplets']} drums:{stats['drums']} lily_fail:{stats['lily_fail']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
