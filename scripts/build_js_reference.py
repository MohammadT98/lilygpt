"""Build a JS Divergence Similarity reference aggregate from a corpus of LilyPond files.

Reads either a JSONL of records with a ``full_text`` field (e.g. ``test.jsonl``)
or a Mutopia manifest JSON, renders each ``.ly`` to MIDI via the existing
``lily_to_midi`` pipeline, computes muspy metrics, aggregates them, and writes
the resulting ``{metric: {mean, std, n}}`` dict to a JSON cache that
``lilybench.evaluate.text_midi`` can consume via ``reference_aggregate_path``.

Usage:
    python scripts/build_js_reference.py \\
        --source test.jsonl \\
        --kind test \\
        --out js_refs/test_muspy_agg.json \\
        --workdir /tmp/js_ref_test \\
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from lilybench.evaluate.js_similarity import aggregate
from lilybench.evaluate.muspy_metrics import compute_muspy_metrics
from lilybench.evaluate.text_midi import lily_to_midi


def _iter_test_full_texts(jsonl_path: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("full_text") or ""
            if not text:
                continue
            rid = rec.get("id") or f"test_{i:05d}"
            items.append((str(rid), text))
    return items


def _iter_mutopia_full_texts(manifest_path: Path) -> list[tuple[str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent

    if isinstance(manifest, dict):
        entries = list(manifest.items())
    else:
        entries = [(str(i), e) for i, e in enumerate(manifest)]

    items: list[tuple[str, str]] = []
    for piece_id, entry in entries:
        if not isinstance(entry, dict):
            continue
        rel = entry.get("localPath") or entry.get("path") or entry.get("lyFile")
        if not rel:
            continue
        ly_path = (base / rel).resolve()
        if not ly_path.exists():
            continue
        try:
            text = ly_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if text:
            items.append((str(piece_id), text))
    return items


def _process_one(args: tuple[str, str, Path, Path]) -> tuple[str, dict]:
    rid, text, ly_dir, midi_dir = args
    ly_path = ly_dir / f"{rid}.ly"
    ly_path.write_text(text, encoding="utf-8")
    result = lily_to_midi(ly_path, midi_dir=midi_dir)
    if not result.get("ok"):
        return rid, {}
    midi_path = Path(result["paths"]["midi"])
    return rid, compute_muspy_metrics(midi_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="test.jsonl or mutopia manifest path")
    parser.add_argument("--kind", choices=("test", "mutopia"), required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="output JSON cache path")
    parser.add_argument("--workdir", type=Path, required=True,
                        help="scratch dir for .ly + .mid files")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="optional cap on number of files processed (for smoke runs)")
    args = parser.parse_args()

    if args.kind == "test":
        items = _iter_test_full_texts(args.source)
    else:
        items = _iter_mutopia_full_texts(args.source)

    if args.limit:
        items = items[: args.limit]

    if not items:
        print(f"[js-ref] no items found in {args.source}", file=sys.stderr)
        return 2

    args.workdir.mkdir(parents=True, exist_ok=True)
    ly_dir = args.workdir / "ly"
    midi_dir = args.workdir / "midi"
    ly_dir.mkdir(exist_ok=True)
    midi_dir.mkdir(exist_ok=True)

    print(f"[js-ref] kind={args.kind} items={len(items)} workers={args.workers}")
    print(f"[js-ref] workdir={args.workdir}")
    t0 = time.time()

    per_file: dict[str, dict] = {}
    work = [(rid, text, ly_dir, midi_dir) for rid, text in items]
    n_done = 0
    n_ok = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_process_one, w) for w in work]
        for fut in as_completed(futures):
            rid, metrics = fut.result()
            n_done += 1
            if metrics:
                per_file[rid] = metrics
                n_ok += 1
            if n_done % 200 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                print(f"[js-ref] {n_done}/{len(items)} ({n_ok} ok), {rate:.1f}/s, {elapsed:.0f}s elapsed",
                      flush=True)

    elapsed = time.time() - t0
    print(f"[js-ref] done: {n_done} processed, {n_ok} produced metrics, {elapsed:.0f}s total")

    if not per_file:
        print("[js-ref] no successful renders, nothing to aggregate", file=sys.stderr)
        return 3

    agg = aggregate(per_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"[js-ref] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
