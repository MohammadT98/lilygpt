"""Run convert-ly on every LilyPond file referenced by the Mutopia manifest.

The Mutopia corpus carries ~15 years of LilyPond syntax drift — `\\include
"deutsch.ly"` (now `\\language "deutsch"`), `\\override Score.MetronomeMark
#'padding = #4.5` (missing `.`), various Scheme top-level scope changes.
LilyPond 2.24.4 ships `convert-ly`, the official upgrader, which handles all
of these mechanical migrations.

This script:
  1. Reads the source manifest at ``--source``.
  2. For each entry's ``localPath``, runs convert-ly to upgrade the file to
     ``--to`` (default 2.24.4) and writes the result to ``--out-dir`` preserving
     the relative tree.
  3. Emits a new manifest at ``--manifest-out`` with the same entries plus a
     ``convert_ly_path`` field holding the absolute path to the upgraded file
     (or null if conversion failed — the original ``localPath`` remains a
     valid fallback for downstream readers).

Usage:
    python scripts/convert_mutopia.py \\
        --source /nfsd/.../mutopia/dataset_mutopia.json \\
        --out-dir /nfsd/.../experiments/lilybench/data/mutopia_convert_ly \\
        --manifest-out /nfsd/.../experiments/lilybench/data/mutopia/dataset_mutopia_converted.json \\
        --convert-ly /home/spanio/lilypond-2.24.4/bin/convert-ly \\
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _convert_one(args: tuple[Path, Path, Path, str]) -> tuple[Path, Path | None, str]:
    """Run convert-ly on one file. Returns (src, out_path | None, error_message)."""
    src, out_path, convert_ly, target_version = args
    if not src.exists():
        return src, None, "missing"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [str(convert_ly), f"--to={target_version}", str(src)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return src, None, "timeout"
    except Exception as exc:
        return src, None, f"exception:{exc}"
    if proc.returncode != 0:
        return src, None, f"convert_ly_rc={proc.returncode}"
    if not proc.stdout:
        return src, None, "empty_output"
    try:
        out_path.write_text(proc.stdout, encoding="utf-8")
    except Exception as exc:
        return src, None, f"write_error:{exc}"
    return src, out_path, ""


def _entry_relpath(entry: dict) -> str | None:
    return entry.get("localPath") or entry.get("path") or entry.get("lyFile")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="dataset_mutopia.json")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="root for upgraded .ly files (parallel to source tree)")
    p.add_argument("--manifest-out", type=Path, required=True,
                   help="path for the new manifest with convert_ly_path field")
    p.add_argument("--convert-ly", default="convert-ly",
                   help="convert-ly executable (default: PATH lookup)")
    p.add_argument("--to", default="2.24.4", help="target LilyPond version")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    convert_ly = shutil.which(args.convert_ly) or args.convert_ly
    if not Path(convert_ly).exists():
        print(f"[convert] convert-ly not found at {convert_ly}", file=sys.stderr)
        return 2

    manifest = json.loads(args.source.read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        # preserve dict keys
        items = list(manifest.items())
        is_dict = True
    elif isinstance(manifest, list):
        items = list(enumerate(manifest))
        is_dict = False
    else:
        print(f"[convert] unexpected manifest shape: {type(manifest).__name__}", file=sys.stderr)
        return 2

    src_root = args.source.parent
    args.out_dir.mkdir(parents=True, exist_ok=True)

    work: list[tuple[Path, Path, Path, str]] = []
    targets: dict[object, Path] = {}  # key -> intended out_path
    for key, entry in items:
        if not isinstance(entry, dict):
            continue
        rel = _entry_relpath(entry)
        if not rel:
            continue
        src = (src_root / rel).resolve()
        out_path = (args.out_dir / rel).resolve()
        work.append((src, out_path, Path(convert_ly), args.to))
        targets[key] = out_path

    if not work:
        print("[convert] no eligible entries in manifest", file=sys.stderr)
        return 3

    print(f"[convert] manifest entries: {len(items)}, eligible files: {len(work)}, workers: {args.workers}")
    print(f"[convert] convert-ly: {convert_ly}, target: {args.to}")
    print(f"[convert] out_dir: {args.out_dir}")

    successes: dict[Path, Path] = {}
    failures: dict[Path, str] = {}
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_convert_one, w) for w in work]
        for fut in as_completed(futures):
            src, out, err = fut.result()
            n_done += 1
            if out is not None:
                successes[src] = out
            else:
                failures[src] = err
            if n_done % 200 == 0:
                rate = n_done / (time.time() - t0 + 1e-9)
                print(f"[convert] {n_done}/{len(work)} ({len(successes)} ok), {rate:.1f}/s",
                      flush=True)

    elapsed = time.time() - t0
    print(f"[convert] done: {len(successes)}/{len(work)} ok in {elapsed:.1f}s")
    if failures:
        # show top 5 distinct error reasons
        reasons: dict[str, int] = {}
        for r in failures.values():
            reasons[r] = reasons.get(r, 0) + 1
        for r, c in sorted(reasons.items(), key=lambda kv: -kv[1])[:5]:
            print(f"[convert]   failure: {c}x {r!r}")

    # write new manifest
    new_entries = []
    for key, entry in items:
        if not isinstance(entry, dict):
            new_entries.append(entry)
            continue
        new_entry = dict(entry)
        out_path = targets.get(key)
        if out_path is not None and out_path.exists():
            new_entry["convert_ly_path"] = str(out_path)
        else:
            new_entry["convert_ly_path"] = None
        new_entries.append((key, new_entry) if is_dict else new_entry)

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    if is_dict:
        out_obj = {k: v for k, v in new_entries}
    else:
        out_obj = new_entries
    args.manifest_out.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"[convert] wrote {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
