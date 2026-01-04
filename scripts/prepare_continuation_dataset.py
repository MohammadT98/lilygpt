#!/usr/bin/env python3
"""
Prepare continuation-style dataset for fine-tuning.

Generates multiple training examples per piece by splitting at different points.
Each example is a continuation task: given a prefix, predict the rest.

Strategy:
- 3 examples per variable assignment (start, middle, near-end)
- Smart splitting at phrase boundaries (after complete measures)
- Validation to ensure syntactic correctness
- No metadata instructions (simplest/safest approach)

Usage:
    python scripts/prepare_continuation_dataset.py \
        --input data/normalized_dataset \
        --output data/continuation_dataset \
        --splits-per-piece 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MusicAssignment:
    """A variable assignment containing music."""
    var_name: str
    full_content: str  # The complete RHS: { ... }
    start_pos: int
    end_pos: int


@dataclass
class ContinuationExample:
    """A single training example (prefix + continuation)."""
    id: str
    source_file: str
    var_name: str
    input_text: str
    output_text: str
    split_point: str  # "start", "middle", or "near_end"
    token_count_estimate: int


def extract_music_assignments(text: str) -> List[MusicAssignment]:
    """
    Extract all music variable assignments from LilyPond file.

    Pattern: varName = \relative do'' { ... } or varName = { ... }
    """
    assignments = []

    # Pattern: variable_name = optional_modifiers { ... }
    # Handles: violinoI = \relative do'' { ... }
    #          main = { ... }
    pattern = re.compile(
        r'^([A-Za-z_][\w]*)\s*=\s*'  # Variable name and =
        r'((?:\\relative\s+[^\s{]+\s*)?)'  # Optional \relative do''
        r'\{',  # Opening brace
        re.MULTILINE
    )

    for match in pattern.finditer(text):
        var_name = match.group(1)
        start_pos = match.start()

        # Find matching closing brace
        brace_pos = text.find('{', match.end() - 1)
        if brace_pos == -1:
            continue

        end_pos = find_matching_brace(text, brace_pos)
        if end_pos == -1:
            continue

        # Extract full assignment: varName = ... { ... }
        full_content = text[start_pos:end_pos + 1]

        # Check if it contains musical content
        if contains_music(full_content):
            assignments.append(MusicAssignment(
                var_name=var_name,
                full_content=full_content,
                start_pos=start_pos,
                end_pos=end_pos
            ))

    return assignments


def find_matching_brace(text: str, open_pos: int) -> int:
    """Find the position of the closing brace that matches open_pos."""
    depth = 1
    i = open_pos + 1

    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1

    return -1


def contains_music(content: str) -> bool:
    """Check if content contains actual musical notes."""
    # Look for note names with durations: do4, re8, mi16, etc.
    note_pattern = r'\b(?:do|re|mi|fa|sol|la|si|[a-gr])[is|es|isbf|esbf]*[,\']*\d'
    return bool(re.search(note_pattern, content, re.I))


def find_phrase_boundaries(content: str) -> List[int]:
    """
    Find good split points in the music (phrase boundaries).

    Good split points are typically:
    - After complete measures (newlines often indicate measure boundaries)
    - After cadences or resting points
    - NOT in the middle of chords, groups, or commands

    Returns list of character positions that are safe split points.
    """
    boundaries = []

    # Split on newlines (usually measure boundaries in formatted LilyPond)
    lines = content.split('\n')
    current_pos = 0

    for i, line in enumerate(lines):
        current_pos += len(line)

        # Skip first and last lines (assignment and closing brace)
        if i > 0 and i < len(lines) - 1:
            # Check if this line ends a musical phrase
            # Good indicators: line has notes and doesn't end mid-chord
            if contains_music(line) and not line.rstrip().endswith(','):
                boundaries.append(current_pos)

        current_pos += 1  # Account for newline

    return boundaries


def find_split_point(content: str, target_percent: float) -> int:
    """
    Find a good split point near target_percent of the content.

    Args:
        content: Full assignment text
        target_percent: Desired split point (0.0 to 1.0)

    Returns:
        Character position to split at (or -1 if not found)
    """
    boundaries = find_phrase_boundaries(content)

    if not boundaries:
        # No good boundaries found, split at target percent (less safe)
        return int(len(content) * target_percent)

    # Find boundary closest to target
    target_pos = int(len(content) * target_percent)
    closest = min(boundaries, key=lambda x: abs(x - target_pos))

    return closest


def create_continuation_examples(
    assignment: MusicAssignment,
    source_file: str,
    splits_per_piece: int = 3
) -> List[ContinuationExample]:
    """
    Create multiple continuation examples from a single assignment.

    Args:
        assignment: Music variable assignment
        source_file: Source file path for tracking
        splits_per_piece: Number of examples to create (default: 3)

    Returns:
        List of continuation examples
    """
    examples = []
    content = assignment.full_content

    # Find the opening brace position
    brace_pos = content.find('{')
    if brace_pos == -1:
        return examples

    # Extract the variable declaration part: "varName = \relative do'' {"
    var_declaration = content[:brace_pos + 1]

    # Extract the music content (between braces)
    closing_brace = content.rfind('}')
    if closing_brace == -1:
        return examples

    music_content = content[brace_pos + 1:closing_brace]

    # Example 1: START - Give variable name, predict everything
    example_1 = ContinuationExample(
        id=f"{source_file}_{assignment.var_name}_start",
        source_file=source_file,
        var_name=assignment.var_name,
        input_text=var_declaration,
        output_text=music_content + "\n}",
        split_point="start",
        token_count_estimate=len(content.split())
    )
    examples.append(example_1)

    if splits_per_piece >= 2:
        # Example 2: MIDDLE - Split at ~50%
        split_pos_2 = find_split_point(music_content, 0.5)
        if split_pos_2 > 0:
            example_2 = ContinuationExample(
                id=f"{source_file}_{assignment.var_name}_middle",
                source_file=source_file,
                var_name=assignment.var_name,
                input_text=var_declaration + music_content[:split_pos_2],
                output_text=music_content[split_pos_2:] + "\n}",
                split_point="middle",
                token_count_estimate=len(content.split())
            )
            examples.append(example_2)

    if splits_per_piece >= 3:
        # Example 3: NEAR END - Split at ~75%
        split_pos_3 = find_split_point(music_content, 0.75)
        if split_pos_3 > 0 and split_pos_3 != split_pos_2:
            example_3 = ContinuationExample(
                id=f"{source_file}_{assignment.var_name}_near_end",
                source_file=source_file,
                var_name=assignment.var_name,
                input_text=var_declaration + music_content[:split_pos_3],
                output_text=music_content[split_pos_3:] + "\n}",
                split_point="near_end",
                token_count_estimate=len(content.split())
            )
            examples.append(example_3)

    return examples


def validate_example(example: ContinuationExample) -> Tuple[bool, str]:
    """
    Validate that an example is syntactically correct.

    Returns:
        (is_valid, error_message)
    """
    full_text = example.input_text + example.output_text

    # Check 1: Balanced braces
    if full_text.count('{') != full_text.count('}'):
        return False, f"Unbalanced braces: {full_text.count('{')} open, {full_text.count('}')} close"

    # Check 2: Has musical content
    if not contains_music(full_text):
        return False, "No musical content found"

    # Check 3: Variable assignment is complete
    if '=' in full_text and not re.search(r'=\s*(?:\\relative[^{]*?)?\{.*\}', full_text, re.DOTALL):
        return False, "Incomplete variable assignment"

    # Check 4: Output must end with closing brace
    if not example.output_text.rstrip().endswith('}'):
        return False, "Output doesn't end with closing brace"

    # Check 5: Input must not be empty
    if len(example.input_text.strip()) == 0:
        return False, "Empty input"

    # Check 6: Output must not be empty
    if len(example.output_text.strip()) <= 2:  # Just "}" is too short
        return False, "Output too short"

    return True, ""


def process_file(
    file_path: Path,
    splits_per_piece: int = 3
) -> Tuple[List[ContinuationExample], Dict[str, int]]:
    """
    Process a single LilyPond file and generate continuation examples.

    Returns:
        (examples, stats)
    """
    stats = {
        'files_processed': 1,
        'assignments_found': 0,
        'examples_created': 0,
        'examples_valid': 0,
        'examples_invalid': 0,
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return [], stats

    # Extract music assignments
    assignments = extract_music_assignments(content)
    stats['assignments_found'] = len(assignments)

    valid_examples = []

    for assignment in assignments:
        # Create continuation examples
        examples = create_continuation_examples(
            assignment,
            source_file=file_path.stem,
            splits_per_piece=splits_per_piece
        )
        stats['examples_created'] += len(examples)

        # Validate each example
        for example in examples:
            is_valid, error_msg = validate_example(example)

            if is_valid:
                valid_examples.append(example)
                stats['examples_valid'] += 1
            else:
                stats['examples_invalid'] += 1
                print(f"Invalid example {example.id}: {error_msg}", file=sys.stderr)

    return valid_examples, stats


def save_examples_jsonl(examples: List[ContinuationExample], output_path: Path):
    """Save examples in JSONL format (not yet tokenized)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            obj = {
                'id': example.id,
                'source_file': example.source_file,
                'var_name': example.var_name,
                'input': example.input_text,
                'output': example.output_text,
                'split_point': example.split_point,
                'token_count_estimate': example.token_count_estimate,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Prepare continuation-style dataset for fine-tuning"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/normalized_dataset'),
        help='Input directory with normalized .ly files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/continuation_dataset'),
        help='Output directory for continuation examples'
    )
    parser.add_argument(
        '--splits-per-piece',
        type=int,
        default=3,
        help='Number of examples per piece (default: 3)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )

    args = parser.parse_args()

    print("="*80)
    print("CONTINUATION DATASET PREPARATION")
    print("="*80)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Splits per piece: {args.splits_per_piece}")
    if args.limit:
        print(f"Limit: {args.limit} files (testing mode)")
    print()

    # Find all .ly files
    input_files = list(args.input.rglob('*.ly'))

    if args.limit:
        input_files = input_files[:args.limit]

    print(f"Found {len(input_files)} .ly files")
    print()

    # Process all files
    all_examples = []
    total_stats = defaultdict(int)

    for i, file_path in enumerate(input_files, 1):
        if i % 100 == 0:
            print(f"Processing {i}/{len(input_files)}...")

        examples, stats = process_file(file_path, args.splits_per_piece)
        all_examples.extend(examples)

        for key, value in stats.items():
            total_stats[key] += value

    print()
    print("="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Assignments found: {total_stats['assignments_found']}")
    print(f"Examples created: {total_stats['examples_created']}")
    print(f"Examples valid: {total_stats['examples_valid']}")
    print(f"Examples invalid: {total_stats['examples_invalid']}")
    print(f"Success rate: {100 * total_stats['examples_valid'] / max(1, total_stats['examples_created']):.1f}%")
    print()

    # Save all examples
    output_file = args.output / 'all_examples.jsonl'
    print(f"Saving {len(all_examples)} examples to {output_file}...")
    save_examples_jsonl(all_examples, output_file)

    return 0


if __name__ == '__main__':
    sys.exit(main())
