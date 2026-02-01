#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

def extract_detokenized_outputs(content: str) -> list[str]:
    """Extract all Detokenized Output sections from file content."""
    outputs = []

    # Pattern to match "Detokenized Output N:" followed by separator and content
    # Content ends at the next separator line (60 '=' characters)
    pattern = r'Detokenized Output \d+:\n={60,}\n(.*?)(?=\n={60,}\n|$)'

    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        # Clean up the content - strip trailing whitespace
        cleaned = match.strip()
        if cleaned:
            outputs.append(cleaned)

    return outputs


def extract_experiment_name(filename: str) -> str:
    """Extract experiment name (e.g., 'exp10') from filename."""
    match = re.match(r'(exp\d+)', filename)
    if match:
        return match.group(1)
    return "unknown"


def process_file(input_path: Path, output_dir: Path) -> int:
    """Process a single inference output file and extract all detokenized outputs."""
    print(f"Processing: {input_path.name}")

    # Read the file content
    content = input_path.read_text(encoding='utf-8', errors='replace')

    # Extract experiment name
    exp_name = extract_experiment_name(input_path.name)

    # Create output directory for this experiment
    exp_output_dir = output_dir / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract all detokenized outputs
    outputs = extract_detokenized_outputs(content)

    # Save each output as a .ly file
    for i, output in enumerate(outputs, start=1):
        output_file = exp_output_dir / f"sample_{i:03d}.ly"
        output_file.write_text(output, encoding='utf-8')

    print(f"  -> Extracted {len(outputs)} samples to {exp_output_dir}")
    return len(outputs)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Detokenized Output sections from inference output files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/inference/outputs"),
        help="Directory containing the inference output files (default: data/inference/outputs)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/inference/samples"),
        help="Output directory for .ly files (default: data/inference/samples)"
    )

    args = parser.parse_args()

    # Find all inference output files
    input_dir = args.input_dir
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Look for exp*-infer-*.out files
    input_files = list(input_dir.glob("exp*-infer-*.out"))

    if not input_files:
        print(f"No inference output files found in {input_dir}")
        return 1

    print(f"Found {len(input_files)} inference output file(s)")
    print(f"Output directory: {args.output_dir}")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    total_samples = 0
    for input_file in sorted(input_files):
        count = process_file(input_file, args.output_dir)
        total_samples += count

    print()
    print(f"Done! Extracted {total_samples} total samples.")
    return 0


if __name__ == "__main__":
    exit(main())
