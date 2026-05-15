#!/usr/bin/env python3
"""Build the emotion-recognition bench JSONL from EMOPIA's converted LilyPond.

Wraps ``lilybench.understanding.dataset_builder.load_emotion_corpus`` +
``build_emotion_bench``. The bench is byte-stable under ``--seed``.

Default invocation::

    python scripts/build_emotion_bench.py \\
      --manifest /nfsd/voce/machine_learning/datasets/emopia/emopia_manifest.csv \\
      --ly-root /nfsd/voce/machine_learning/datasets/emopia \\
      --out /nfsd/voce/machine_learning/experiments/lilybench/data/understanding/bench_emotion.jsonl \\
      --seed 1234 --n 120 --max-bars 16
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from lilybench.understanding import dataset_builder


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--ly-root", type=Path, required=True,
                   help="Root that ``ly_path`` entries in the manifest are relative to.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--max-bars", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[emotion-bench] loading corpus from {args.manifest}")
    corpus = dataset_builder.load_emotion_corpus(
        args.manifest, args.ly_root, max_bars=args.max_bars
    )
    print(f"[emotion-bench] loaded {len(corpus)} clips")

    records = dataset_builder.build_emotion_bench(corpus, seed=args.seed, n=args.n)
    dataset_builder.write_jsonl(records, args.out)
    counts = Counter(r["gold"] for r in records)
    print(f"[emotion-bench] wrote {len(records)} records to {args.out}")
    for q in ("Q1", "Q2", "Q3", "Q4"):
        print(f"        {q}: {counts.get(q, 0)}")


if __name__ == "__main__":
    main()
