#!/usr/bin/env python3
"""
Wrapper to run continuation dataset preparation with comprehensive logging.
Saves both console output and final statistics to a log file.
"""

import subprocess
import sys
import json
import random
from pathlib import Path
from datetime import datetime


def run_continuation_preparation(log_file):
    """Run the continuation dataset preparation script."""

    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write("STEP 1: Generate Continuation Examples\n")
    log_file.write("=" * 80 + "\n\n")

    cmd = [
        sys.executable,
        "scripts/prepare_continuation_dataset.py",
        "--input", "data/normalized_dataset",
        "--output", "data/continuation_dataset",
        "--splits-per-piece", "3"
    ]

    print("=" * 80)
    print("STEP 1: Generate Continuation Examples")
    print("=" * 80)
    print()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    # Stream output to both console and log
    for line in process.stdout:
        print(line, end='', flush=True)
        log_file.write(line)
        log_file.flush()

    process.wait()

    if process.returncode != 0:
        log_file.write(f"\nERROR: Process exited with code {process.returncode}\n")
        return False

    return True


def split_dataset(log_file):
    """Split the dataset into train/val/test."""

    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write("STEP 2: Split into Train/Val/Test\n")
    log_file.write("=" * 80 + "\n\n")

    print()
    print("=" * 80)
    print("STEP 2: Split into Train/Val/Test")
    print("=" * 80)
    print()

    try:
        # Load examples
        print("Loading examples...")
        log_file.write("Loading examples...\n")

        with open('data/continuation_dataset/all_examples.jsonl', encoding='utf-8') as f:
            examples = [json.loads(line) for line in f if line.strip()]

        msg = f"Total examples: {len(examples)}\n"
        print(msg, end='')
        log_file.write(msg)

        # Shuffle
        random.seed(42)
        random.shuffle(examples)

        # Split: 80% train, 10% val, 10% test
        n = len(examples)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)

        train = examples[:train_size]
        val = examples[train_size:train_size + val_size]
        test = examples[train_size + val_size:]

        msg = f"Train: {len(train)} examples\n"
        print(msg, end='')
        log_file.write(msg)

        msg = f"Val:   {len(val)} examples\n"
        print(msg, end='')
        log_file.write(msg)

        msg = f"Test:  {len(test)} examples\n"
        print(msg, end='')
        log_file.write(msg)

        # Save
        Path('data/splits').mkdir(exist_ok=True)

        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            output_path = f'data/splits/{split_name}.jsonl'
            with open(output_path, 'w', encoding='utf-8') as f:
                for ex in split_data:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')

            msg = f"Saved {split_name}.jsonl\n"
            print(msg, end='')
            log_file.write(msg)

        print()
        log_file.write("\n")

        return True

    except Exception as e:
        error_msg = f"ERROR: {e}\n"
        print(error_msg, end='')
        log_file.write(error_msg)
        return False


def main():
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"continuation_pipeline_{timestamp}.log"

    print("=" * 80)
    print("CONTINUATION DATASET PIPELINE")
    print("=" * 80)
    print()
    print(f"Log file: {log_path}")
    print()

    with open(log_path, 'w', encoding='utf-8') as log_file:
        # Write header
        log_file.write("=" * 80 + "\n")
        log_file.write("CONTINUATION DATASET PIPELINE LOG\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 80 + "\n")

        # Step 1: Generate continuation examples
        success = run_continuation_preparation(log_file)
        if not success:
            log_file.write("\nERROR: Continuation dataset preparation failed!\n")
            print("\nERROR: Continuation dataset preparation failed!")
            print(f"Check log: {log_path}")
            return 1

        # Step 2: Split dataset
        success = split_dataset(log_file)
        if not success:
            log_file.write("\nERROR: Dataset splitting failed!\n")
            print("\nERROR: Dataset splitting failed!")
            print(f"Check log: {log_path}")
            return 1

        # Success
        print("=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print()
        print("Dataset ready at:")
        print("  - data/splits/train.jsonl")
        print("  - data/splits/val.jsonl")
        print("  - data/splits/test.jsonl")
        print()
        print(f"Log saved to: {log_path}")
        print()

        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("SUCCESS! Pipeline completed\n")
        log_file.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
