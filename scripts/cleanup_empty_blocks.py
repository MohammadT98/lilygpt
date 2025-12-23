#!/usr/bin/env python3
"""Clean up empty variable assignments from normalized LilyPond files.

This script removes empty variable assignments (including nested empty blocks)
from existing normalized dataset files.
"""

import argparse
import sys
from pathlib import Path

# Add src to path to import lilynorm modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lilynorm.stages.preprocessing.engrave_strip import _remove_empty_variable_assignments


def cleanup_file(file_path: Path) -> tuple[bool, int]:
    """Clean up a single file and return (success, removed_count)."""
    try:
        content = file_path.read_text(encoding="utf-8")
        cleaned, removed_count = _remove_empty_variable_assignments(content)
        
        if removed_count > 0:
            file_path.write_text(cleaned, encoding="utf-8")
            return True, removed_count
        return True, 0
    except Exception as e:
        print(f"ERROR processing {file_path}: {e}", file=sys.stderr)
        return False, 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove empty variable assignments from LilyPond files"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to file or directory to clean up",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without making changes",
    )
    
    args = parser.parse_args()
    target = Path(args.path).expanduser().resolve()
    
    if not target.exists():
        print(f"ERROR: Path not found: {target}", file=sys.stderr)
        return 1
    
    files_to_process = []
    if target.is_file():
        if target.suffix == ".ly":
            files_to_process = [target]
        else:
            print(f"ERROR: Not a .ly file: {target}", file=sys.stderr)
            return 1
    else:
        files_to_process = list(target.rglob("*.ly"))
    
    if not files_to_process:
        print(f"No .ly files found in {target}")
        return 0
    
    print(f"Found {len(files_to_process)} file(s) to process")
    
    total_removed = 0
    success_count = 0
    
    for file_path in files_to_process:
        if args.dry_run:
            content = file_path.read_text(encoding="utf-8")
            _, removed_count = _remove_empty_variable_assignments(content)
            if removed_count > 0:
                print(f"Would remove {removed_count} empty assignment(s) from {file_path}")
                total_removed += removed_count
        else:
            success, removed_count = cleanup_file(file_path)
            if success:
                success_count += 1
                if removed_count > 0:
                    print(f"Removed {removed_count} empty assignment(s) from {file_path}")
                    total_removed += removed_count
    
    if args.dry_run:
        print(f"\nDry run complete: would remove {total_removed} empty assignment(s) from {len(files_to_process)} file(s)")
    else:
        print(f"\nComplete: removed {total_removed} empty assignment(s) from {success_count}/{len(files_to_process)} file(s)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

