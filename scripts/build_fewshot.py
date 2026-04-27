"""Build a few-shot demonstration file by sampling actual records from train.jsonl.

The current default.txt was hand-written with stylistically-narrow examples
(A minor, 4/4, 4 bars, monophonic, no metadata header) that don't match the
training-data shape. This pushes the model to produce outputs that are
syntactically clean but distributionally far from the corpus — visible in the
JS-sim and FMD numbers.

This script samples N records from train.jsonl, stratified by composer and
filtered to a moderate length range, so the demonstrations cover real corpus
diversity (multiple keys, forms, voices) and the same structural shape
(metadata header + version/language + named assignments) the model expects.

Usage:
    python scripts/build_fewshot.py \\
        --source data/splits_full/train.jsonl \\
        --out configs/fewshot/default.txt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

_COMPOSER_RE = re.compile(r"^%%\s*composer:\s*(.+)$", re.MULTILINE)
_FORM_RE = re.compile(r"^%%\s*musical_form:\s*(.+)$", re.MULTILINE)


def _parse_meta(text: str) -> tuple[str | None, str | None]:
    c = _COMPOSER_RE.search(text)
    f = _FORM_RE.search(text)
    return (
        c.group(1).strip() if c else None,
        f.group(1).strip() if f else None,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="train.jsonl")
    p.add_argument("--out", type=Path, required=True, help="output .txt path")
    p.add_argument("--n", type=int, default=3, help="number of demonstrations")
    p.add_argument("--min-chars", type=int, default=500)
    p.add_argument("--max-chars", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    by_composer: dict[str, list[str]] = defaultdict(list)
    n_total = 0
    n_eligible = 0

    with args.source.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ft = rec.get("full_text") or ""
            if not (args.min_chars <= len(ft) <= args.max_chars):
                continue
            composer, _ = _parse_meta(ft)
            if not composer:
                continue
            n_eligible += 1
            by_composer[composer].append(ft)

    if not by_composer:
        print(f"[fewshot] no eligible records found in {args.source}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    composers = sorted(by_composer.keys())
    rng.shuffle(composers)

    picks: list[str] = []
    picked_composers: list[str] = []
    for c in composers:
        if len(picks) >= args.n:
            break
        choice = rng.choice(by_composer[c])
        picks.append(choice)
        picked_composers.append(c)

    if len(picks) < args.n:
        print(
            f"[fewshot] only {len(picks)} eligible composers; needed {args.n}",
            file=sys.stderr,
        )
        return 3

    preamble = (
        "Here are examples of LilyPond fragments in the requested style. "
        "Each example follows the structure used by the training corpus: a "
        "metadata header (composer, period, musical form, ensemble, part) "
        "in `%% ===` comments, then `\\version` / `\\language`, then one or "
        "more named music assignments. The model must output only LilyPond "
        "code in the same shape — no prose, no markdown fences."
    )

    parts: list[str] = [preamble, ""]
    for i, demo in enumerate(picks, 1):
        parts.append(f"Example {i}:")
        parts.append(demo.rstrip())
        parts.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    total_chars = sum(len(d) for d in picks)
    print(
        f"[fewshot] wrote {args.out}\n"
        f"  records scanned: {n_total}, eligible: {n_eligible}, "
        f"composers picked: {picked_composers}\n"
        f"  total demo chars: {total_chars}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
