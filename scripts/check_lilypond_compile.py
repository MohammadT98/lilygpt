"""Batch-compile normalized LilyPond files to detect syntax breakage.

Usage:
    uv run python scripts/check_lilypond_compile.py [--root data/normalized_dataset] [--pattern *.ly] [--timeout 60]

The script invokes lilypond with no output generation (writes to a temp dir)
so it won't clutter the dataset directories. It reports which files fail and
summaries counts at the end.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# LilyPond resolution (reuse project helper if available)
# ---------------------------------------------------------------------------

def _resolve_lilypond_cmd() -> str:
    # 1. Check for local lilypond installation in project root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    local_lily = repo_root / "lilypond-2.24.4" / "bin" / "lilypond.exe"
    if local_lily.exists():
        return str(local_lily)
    
    # 2. Try the project's resolver if importable
    try:
        from lilynorm.stages.preprocessing.normalize import resolve_lily_cmd  # type: ignore
        return resolve_lily_cmd()
    except Exception:
        pass
    
    # 3. Fallback: try PATH, then default name
    return "lilypond"


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _iter_files(root: Path, pattern: str) -> Iterable[Path]:
    for path in root.rglob(pattern):
        # Skip macOS resource forks and __MACOSX bundles
        if any(part == "__MACOSX" for part in path.parts):
            continue
        if path.name.startswith("._"):
            continue
        if path.is_file():
            yield path


def _run_lilypond(lily_cmd: str, ly_file: Path, timeout: int) -> subprocess.CompletedProcess:
    # Write output to a temp dir to avoid polluting the dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = ly_file.resolve()
        cmd: Sequence[str] = [
            lily_cmd,
            "--formats=ps",  # lightweight output
            # Keep default loglevel to surface real error context
            "--output",
            str(Path(tmpdir) / "out"),
            str(src_path),
        ]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
            cwd=None,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check LilyPond compilation of normalized files")
    parser.add_argument(
        "--root",
        default="data/normalized_dataset",
        help="Root directory containing normalized .ly files",
    )
    parser.add_argument(
        "--pattern",
        default="*.ly",
        help="Glob pattern for files to test (default: *.ly)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-file timeout in seconds",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"[error] root does not exist: {root}", file=sys.stderr)
        return 1

    lily_cmd = _resolve_lilypond_cmd()
    print(f"[info] using lilypond: {lily_cmd}")

    files = sorted(_iter_files(root, args.pattern))
    if not files:
        print(f"[warn] no files matching pattern under {root}")
        return 0

    ok = 0
    fail = 0
    failures: list[tuple[Path, str]] = []

    for ly in files:
        result = _run_lilypond(lily_cmd, ly, args.timeout)
        if result.returncode == 0:
            ok += 1
        else:
            fail += 1
            stderr_tail = (result.stderr or "").strip()
            stdout_tail = (result.stdout or "").strip()
            combined = (stderr_tail + "\n" + stdout_tail).strip()
            if not combined:
                combined = "(no output)"
            # Trim long logs
            if len(combined) > 6000:
                combined = combined[-6000:]
            lines = combined.splitlines()
            preview = " | ".join(lines[:5]) if lines else combined
            failures.append((ly, combined))
            print(f"[fail] {ly} :: {preview}")

    print(f"[done] success: {ok}  failed: {fail}  total: {ok + fail}")

    if failures:
        print("\nFailures:")
        for path, err in failures:
            print(f"- {path}\n  {err.replace('\n', '\n  ')}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
