#!/usr/bin/env python3
"""Build the music-understanding benchmark JSONL from the Mutopia corpus.

Reads ``dataset_mutopia.json``, loads every .ly file it points at, parses
title/key/meter from the score body, and emits one record per (task, item)
into ``--out``. The bench is byte-stable given the same ``--seed``.

Default invocation::

    python scripts/build_understanding_bench.py \\
      --mutopia-json /nfsd/voce/machine_learning/datasets/mutopia/dataset_mutopia.json \\
      --mutopia-root /nfsd/voce/machine_learning/datasets/mutopia/stripped \\
      --out data/understanding/bench.jsonl --seed 1234

Pass ``--limit`` to cap each task's sample count for quick local smoke tests.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lilybench.understanding import dataset_builder, tasks


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mutopia-json", type=Path, required=True)
    p.add_argument("--mutopia-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap each task's sample count to this value (for smoke tests)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[bench] loading corpus from {args.mutopia_json}")
    corpus = dataset_builder.build_corpus(args.mutopia_json, args.mutopia_root)
    print(f"[bench] loaded {len(corpus)} pieces")

    sizes = None
    if args.limit is not None:
        sizes = {name: min(spec.n, args.limit) for name, spec in tasks.TASKS.items()}

    records = dataset_builder.build_bench(corpus, seed=args.seed, task_sizes=sizes)
    dataset_builder.write_jsonl(records, args.out)

    counts = Counter(r["task"] for r in records)
    print(f"[bench] wrote {len(records)} records to {args.out}")
    for name in sorted(tasks.TASKS):
        target = (sizes or {}).get(name, tasks.TASKS[name].n)
        print(f"        {name:24s} {counts.get(name, 0):4d} / target {target}")


if __name__ == "__main__":
    main()
