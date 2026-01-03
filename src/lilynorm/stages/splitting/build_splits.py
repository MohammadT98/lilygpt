# python -m scripts.build_splits

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

try:
    # Import is not needed but helps with type checking
    pass
except ModuleNotFoundError:
    # Allow running the script directly from the repo without installing the package.
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))


DEFAULT_INPUT_JSONL = "data/continuation_dataset/all_examples.jsonl"
DEFAULT_OUTPUT_DIR = "data/splits"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42


@dataclass
class Sample:
    """Representation of a single LilyPond training example (raw or tokenized)."""
    id: str
    source_file: str | None  # Used to track source file for proper splitting
    raw_data: Dict[str, Any]  # Store the entire JSON object to write back


def _get_base_work(source_file: str) -> str:
    """Extract base work name by removing _partN suffix.

    Example: 'vivaldi_concerto_score_part3' -> 'vivaldi_concerto_score'
    """
    return re.sub(r'_part\d+$', '', source_file)


def _load_from_jsonl(jsonl_path: Path) -> List[Sample]:
    """Load all samples from a JSONL file (works with both raw and tokenized data)."""
    samples: List[Sample] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[build_splits] ! skipping line {line_num} due to JSON error: {exc}")
                continue

            # Extract ID and source_file for splitting
            sample_id = obj.get("id", f"sample_{line_num}")
            source_file = obj.get("source_file")

            if not isinstance(source_file, str):
                source_file = None

            # Store entire object to write back unchanged
            samples.append(Sample(
                id=sample_id,
                source_file=source_file,
                raw_data=obj,
            ))

    if not samples:
        raise RuntimeError(f"No valid samples loaded from {jsonl_path}")

    return samples


def _train_val_test_split(
    samples: List[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[List[Sample], List[Sample], List[Sample]]:
    """Randomly split samples into train/val/test BY BASE WORK to prevent data leakage.

    This ensures that all parts/examples from the same musical work end up in the same split.
    For example, 'vivaldi_concerto_part1', 'vivaldi_concerto_part2', 'vivaldi_concerto_part3'
    all get grouped together and assigned to the same split.
    """
    rng = random.Random(seed)

    # Group samples by base work (removing _partN suffix)
    work_to_samples: Dict[str, List[Sample]] = {}
    no_source_samples: List[Sample] = []

    for sample in samples:
        if sample.source_file:
            base_work = _get_base_work(sample.source_file)
            if base_work not in work_to_samples:
                work_to_samples[base_work] = []
            work_to_samples[base_work].append(sample)
        else:
            no_source_samples.append(sample)

    # Get list of unique base works and shuffle them
    base_works = list(work_to_samples.keys())
    rng.shuffle(base_works)

    # Split base works into train/val/test
    n_works = len(base_works)
    n_train_works = int(n_works * train_ratio)
    n_val_works = int(n_works * val_ratio)

    train_works = base_works[:n_train_works]
    val_works = base_works[n_train_works:n_train_works + n_val_works]
    test_works = base_works[n_train_works + n_val_works:]

    # Collect all samples from each work group
    train_samples = []
    val_samples = []
    test_samples = []

    for work in train_works:
        train_samples.extend(work_to_samples[work])
    for work in val_works:
        val_samples.extend(work_to_samples[work])
    for work in test_works:
        test_samples.extend(work_to_samples[work])

    # Handle samples without source_file (split them randomly as fallback)
    if no_source_samples:
        rng.shuffle(no_source_samples)
        n = len(no_source_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_samples.extend(no_source_samples[:n_train])
        val_samples.extend(no_source_samples[n_train:n_train + n_val])
        test_samples.extend(no_source_samples[n_train + n_val:])

        print(f"[build_splits] WARNING: {len(no_source_samples)} samples without source_file, "
              f"split randomly (may cause leakage)")

    print(f"[build_splits] Split by base work (grouped _partN files):")
    print(f"  train: {len(train_works)} works -> {len(train_samples)} samples")
    print(f"  val:   {len(val_works)} works -> {len(val_samples)} samples")
    print(f"  test:  {len(test_works)} works -> {len(test_samples)} samples")

    return train_samples, val_samples, test_samples


def _write_jsonl(samples: List[Sample], path: Path) -> None:
    """Write samples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            # Write the original data unchanged
            f.write(json.dumps(s.raw_data, ensure_ascii=False) + "\n")


def _print_stats(name: str, samples: List[Sample]) -> None:
    if not samples:
        print(f"[build_splits] {name}: empty set")
        return

    # Count unique base works
    base_works = set(_get_base_work(s.source_file) for s in samples if s.source_file)

    print(
        f"[build_splits] {name}: "
        f"n={len(samples)} samples from {len(base_works)} unique base works"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build train/val/test splits from tokenized LilyPond dataset (BY BASE WORK)."
    )
    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help=f"Input JSONL file with tokenized examples (default: {DEFAULT_INPUT_JSONL}).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for train/val/test JSONL files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Fraction of BASE WORKS to use for training (default: {DEFAULT_TRAIN_RATIO}).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Fraction of BASE WORKS to use for validation (default: {DEFAULT_VAL_RATIO}). "
             "The remaining works are used for test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for shuffling (default: {DEFAULT_SEED}).",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_jsonl.exists():
        print(f"[build_splits] input JSONL not found: {input_jsonl}")
        return 2

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1.0:
        print(
            "[build_splits] invalid split ratios: "
            f"train={args.train_ratio}, val={args.val_ratio}"
        )
        return 2

    print(f"[build_splits] loading samples from {input_jsonl}")
    samples = _load_from_jsonl(input_jsonl)
    print(f"[build_splits] loaded {len(samples)} total samples")

    print(
        f"[build_splits] splitting BY BASE WORK (grouping _partN files) "
        f"(train={args.train_ratio}, val={args.val_ratio}, "
        f"test={1.0 - args.train_ratio - args.val_ratio})"
    )
    train, val, test = _train_val_test_split(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    _write_jsonl(train, train_path)
    _write_jsonl(val, val_path)
    _write_jsonl(test, test_path)

    print(f"[build_splits] wrote:")
    print(f"  train -> {train_path}")
    print(f"  val   -> {val_path}")
    print(f"  test  -> {test_path}")

    _print_stats("train", train)
    _print_stats("val", val)
    _print_stats("test", test)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
