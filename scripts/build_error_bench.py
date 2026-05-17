#!/usr/bin/env python3
"""Build the error-detection bench JSONL by injecting synthetic errors
into Mutopia scores.

Wraps ``lilybench.understanding.dataset_builder.build_corpus`` +
``build_error_bench``. Byte-stable under ``--seed``.

Default invocation::

    python scripts/build_error_bench.py \\
      --mutopia-json /nfsd/voce/machine_learning/datasets/mutopia/dataset_mutopia.json \\
      --mutopia-root /nfsd/voce/machine_learning/datasets/mutopia/ \\
      --out /nfsd/voce/machine_learning/experiments/lilybench/data/understanding/bench_errors.jsonl \\
      --seed 1234 --n 220
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from lilybench.understanding import dataset_builder


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mutopia-json", type=Path, required=True)
    p.add_argument("--mutopia-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n", type=int, default=220)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[error-bench] loading corpus from {args.mutopia_json}")
    corpus = dataset_builder.build_corpus(args.mutopia_json, args.mutopia_root)
    print(f"[error-bench] loaded {len(corpus)} pieces")

    records = dataset_builder.build_error_bench(corpus, seed=args.seed, n=args.n)
    dataset_builder.write_jsonl(records, args.out)

    counts = Counter(r["category"] for r in records)
    print(f"[error-bench] wrote {len(records)} records to {args.out}")
    for cat in sorted(counts):
        print(f"        {cat:24s} {counts[cat]}")


if __name__ == "__main__":
    main()
