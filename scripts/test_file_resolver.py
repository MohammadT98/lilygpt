"""
Debug script to save output from each preprocessing stage.

Usage:
    python scripts/debug_stages.py <input_file> [--output-dir <dir>]

Example:
    python scripts/debug_stages.py "data/raw/NO PUB/NO PUB/charpentier_lauda_sion_H_268_egredimini_H_280_score.ly"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

from lilynorm.stages.preprocessing import file_resolver, preparse, normalize as norm_module, engrave_strip
from lilynorm.utils.options import NormOptions


def main():
    parser = argparse.ArgumentParser(description="Save output from each preprocessing stage for debugging")
    parser.add_argument("input_file", help="Path to input LilyPond file")
    parser.add_argument("--output-dir", default="data/test_file_resolver", help="Directory to save stage outputs (default: data/debug_stages)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input
    print(f"Reading: {input_path}")
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    print(f"  Input size: {len(text)} chars")
    
    # Stage 0: File resolver (resolve includes, inline variabili.ly, fix typos, remove -+ markers)
    print("\n=== Stage 0: File Resolver ===")
    stage0 = file_resolver.run(text, input_path, exclude_variabili=False)
    stage0_path = output_dir / f"{input_path.stem}_stage0_file_resolver.ly"
    stage0_path.write_text(stage0, encoding="utf-8")
    print(f"  Output size: {len(stage0)} chars")
    print(f"  Saved to: {stage0_path}")
    print(f"  Has 'set-default-paper-size': {'set-default-paper-size' in stage0}")
    print(f"  Has '-+' markers: {'-+' in stage0}")
    
    # Stage 1: Preparse (remove comments, clean whitespace)
    print("\n=== Stage 1: Preparse ===")
    opts = NormOptions()
    stage1 = preparse.run(stage0, opts)
    stage1_path = output_dir / f"{input_path.stem}_stage1_preparse.ly"
    stage1_path.write_text(stage1, encoding="utf-8")
    print(f"  Output size: {len(stage1)} chars")
    print(f"  Saved to: {stage1_path}")
    
    # Stage 2: Normalize (expand relative, inline variables, resolve transpose, etc.)
    print("\n=== Stage 2: Normalize ===")
    stage2 = norm_module.run(stage1, opts)
    stage2_path = output_dir / f"{input_path.stem}_stage2_normalize.ly"
    stage2_path.write_text(stage2, encoding="utf-8")
    print(f"  Output size: {len(stage2)} chars")
    print(f"  Saved to: {stage2_path}")
    
    # Stage 3: Engrave strip (remove overrides, markups, dynamics, etc.)
    print("\n=== Stage 3: Engrave Strip ===")
    stage3 = engrave_strip.run(stage2, opts)
    stage3_path = output_dir / f"{input_path.stem}_stage3_engrave_strip.ly"
    stage3_path.write_text(stage3, encoding="utf-8")
    print(f"  Output size: {len(stage3)} chars")
    print(f"  Saved to: {stage3_path}")
    print(f"  Has 'set-default-paper-size': {'set-default-paper-size' in stage3}")
    
    print(f"\n✓ All stages saved to: {output_dir}")
    print(f"\nStage size progression:")
    print(f"  Input:         {len(text):>8} chars")
    print(f"  File resolver: {len(stage0):>8} chars ({len(stage0) - len(text):+d})")
    print(f"  Preparse:      {len(stage1):>8} chars ({len(stage1) - len(stage0):+d})")
    print(f"  Normalize:     {len(stage2):>8} chars ({len(stage2) - len(stage1):+d})")
    print(f"  Engrave strip: {len(stage3):>8} chars ({len(stage3) - len(stage2):+d})")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
