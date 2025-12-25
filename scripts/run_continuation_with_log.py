#!/usr/bin/env python3
"""
Wrapper script to run continuation dataset preparation with logging.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def main():
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"continuation_dataset_{timestamp}.log"

    print(f"Running continuation dataset preparation...")
    print(f"Log file: {log_file}")
    print()

    # Run the script and capture output
    cmd = [
        sys.executable,
        "scripts/prepare_continuation_dataset.py",
        "--input", "data/normalized_dataset",
        "--output", "data/continuation_dataset",
        "--splits-per-piece", "3"
    ]

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Continuation Dataset Generation Log\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("=" * 80 + "\n\n")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end='')
            f.write(line)

        process.wait()

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Finished: {datetime.now()}\n")
        f.write(f"Exit code: {process.returncode}\n")

    print()
    print(f"Log saved to: {log_file}")

    return process.returncode

if __name__ == '__main__':
    sys.exit(main())
