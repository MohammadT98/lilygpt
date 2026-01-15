#!/usr/bin/env python3

import json
import re
from pathlib import Path
from typing import List, Dict, Any


def extract_assignments(ly_path: Path) -> List[Dict[str, Any]]:
    try:
        with open(ly_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"[!] Could not read {ly_path}: {e}")
        return []

    pattern = r'(\w+)\s*=\s*(\\relative\s+\w+[\'",]*\s*)?(\{)'

    assignments = []

    for match in re.finditer(pattern, content):
        var_name = match.group(1)
        start_pos = match.start()

        brace_count = 1
        pos = match.end()

        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            full_text = content[start_pos:pos]

            header_end = match.end()  # Position after opening {
            input_text = content[start_pos:header_end]
            output_text = content[header_end:pos]

            token_estimate = len(full_text) // 4

            assignments.append({
                'id': f"{ly_path.stem}_{var_name}",
                'source_file': ly_path.stem,
                'var_name': var_name,
                'input': input_text,
                'output': output_text,
                'full_text': full_text,
                'token_count_estimate': token_estimate,
            })

    return assignments


def main():
    normalized_root = Path("data/normalized_dataset")
    output_dir = Path("data/full_assignment_dataset")
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

    all_examples = []
    files_processed = 0
    files_with_examples = 0

    for ly_file in all_ly_files:
        assignments = extract_assignments(ly_file)

        if assignments:
            all_examples.extend(assignments)
            files_with_examples += 1

        files_processed += 1

        if files_processed % 100 == 0:
            print(f"Processed {files_processed}/{len(all_ly_files)} files, "
                  f"extracted {len(all_examples)} assignments")

    print()
    print(f"Total files processed: {files_processed}")
    print(f"Files with assignments: {files_with_examples}")
    print(f"Total assignments extracted: {len(all_examples)}")
    print()

    tokens = [ex['token_count_estimate'] for ex in all_examples]
    if tokens:
        print(f"Token statistics:")
        print(f"  Average: {sum(tokens) / len(tokens):.1f}")
        print(f"  Min: {min(tokens)}")
        print(f"  Max: {max(tokens)}")
        print(f"  Over 2048: {sum(1 for t in tokens if t > 2048)} ({100*sum(1 for t in tokens if t > 2048)/len(tokens):.1f}%)")
    print()

    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"Wrote {len(all_examples)} examples to {output_file}")
    print()
    print("Next step: Run build_splits.py to create train/val/test splits")
    print(f"  python src/lilynorm/stages/splitting/build_splits.py \\")
    print(f"    --input-jsonl {output_file} \\")
    print(f"    --output-dir data/splits_full")
    print()


if __name__ == '__main__':
    main()
