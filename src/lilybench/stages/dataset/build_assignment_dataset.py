#!/usr/bin/env python3

"""Build the full assignment dataset from normalized LilyPond files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

ASSIGNMENT_RE = re.compile(r'(\w+)\s*=\s*(\\relative\s+\w+[\'",]*\s*)?(\{)')


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[!] Could not read {path}: {exc}")
        return None


def _find_matching_brace(text: str, start: int) -> int:
    depth = 1
    pos = start + 1
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return pos if depth == 0 else -1


def extract_assignments(ly_path: Path) -> List[Dict[str, Any]]:
    """Extract variable assignments from a normalized LilyPond file."""
    content = _read_text(ly_path)
    if content is None:
        return []

    assignments: List[Dict[str, Any]] = []

    for match in ASSIGNMENT_RE.finditer(content):
        var_name = match.group(1)
        start_pos = match.start()
        brace_start = match.end() - 1
        end_pos = _find_matching_brace(content, brace_start)
        if end_pos == -1:
            continue

        full_text = content[start_pos:end_pos]
        header_end = match.end()
        input_text = content[start_pos:header_end]
        output_text = content[header_end:end_pos]
        token_estimate = len(full_text) // 4

        assignments.append({
            "id": f"{ly_path.stem}_{var_name}",
            "source_file": ly_path.stem,
            "var_name": var_name,
            "input": input_text,
            "output": output_text,
            "full_text": full_text,
            "token_count_estimate": token_estimate,
        })

    return assignments


def main() -> None:
    """Create a JSONL dataset with one assignment per example."""
    normalized_root = Path("data/normalized_dataset")
    output_dir = Path("data/assignment_dataset")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "all_examples.jsonl"

    print("=" * 80)
    print("FULL ASSIGNMENT DATASET PREPARATION")
    print("=" * 80)
    print()
    print(f"Input: {normalized_root}")
    print(f"Output: {output_file}")
    print()

    all_ly_files = sorted(normalized_root.rglob("*.ly"))
    print(f"Found {len(all_ly_files)} LilyPond files")
    print()

    all_examples: List[Dict[str, Any]] = []
    files_processed = 0
    files_with_examples = 0

    for ly_file in all_ly_files:
        assignments = extract_assignments(ly_file)
        if assignments:
            all_examples.extend(assignments)
            files_with_examples += 1

        files_processed += 1
        if files_processed % 100 == 0:
            print(
                f"Processed {files_processed}/{len(all_ly_files)} files, "
                f"extracted {len(all_examples)} assignments"
            )

    print()
    print(f"Total files processed: {files_processed}")
    print(f"Files with assignments: {files_with_examples}")
    print(f"Total assignments extracted: {len(all_examples)}")
    print()

    tokens = [ex["token_count_estimate"] for ex in all_examples]
    if tokens:
        print("Token statistics:")
        print(f"  Average: {sum(tokens) / len(tokens):.1f}")
        print(f"  Min: {min(tokens)}")
        print(f"  Max: {max(tokens)}")
        over_limit = sum(1 for t in tokens if t > 2048)
        print(f"  Over 2048: {over_limit} ({100 * over_limit / len(tokens):.1f}%)")
    print()

    with output_file.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_examples)} examples to {output_file}")
    print()
    print("Next step: Run build_splits.py to create train/val/test splits")
    print("  python src/lilybench/stages/splitting/build_splits.py \\")
    print(f"    --input-jsonl {output_file} \\")
    print("    --output-dir data/splits_full")
    print()


if __name__ == "__main__":
    main()
