"""Create train/val/test splits without cross-work leakage."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

DEFAULT_INPUT_JSONL = "data/full_assignment_dataset/all_examples.jsonl"
DEFAULT_OUTPUT_DIR = "data/splits_full"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42


@dataclass
class Sample:
    """Lightweight wrapper for JSONL samples."""
    id: str
    source_file: str | None
    raw_data: Dict[str, Any]


def _get_base_work(source_file: str) -> str:
    """Strip part suffixes to group parts from the same work."""
    return re.sub(r'_part\d+$', '', source_file)


def _load_from_jsonl(jsonl_path: Path) -> List[Sample]:
    """Load samples from a JSONL dataset."""
    samples: List[Sample] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[build_splits] skipping line {line_num}: {exc}")
                continue

            sample_id = obj.get("id", f"sample_{line_num}")
            source_file = obj.get("source_file")

            if not isinstance(source_file, str):
                source_file = None

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
    """Split samples by base work to avoid leakage."""
    rng = random.Random(seed)

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

    base_works = list(work_to_samples.keys())
    rng.shuffle(base_works)

    n_works = len(base_works)
    n_train_works = int(n_works * train_ratio)
    n_val_works = int(n_works * val_ratio)

    train_works = base_works[:n_train_works]
    val_works = base_works[n_train_works:n_train_works + n_val_works]
    test_works = base_works[n_train_works + n_val_works:]

    train_samples = []
    val_samples = []
    test_samples = []

    for work in train_works:
        train_samples.extend(work_to_samples[work])
    for work in val_works:
        val_samples.extend(work_to_samples[work])
    for work in test_works:
        test_samples.extend(work_to_samples[work])

    if no_source_samples:
        rng.shuffle(no_source_samples)
        n = len(no_source_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_samples.extend(no_source_samples[:n_train])
        val_samples.extend(no_source_samples[n_train:n_train + n_val])
        test_samples.extend(no_source_samples[n_train + n_val:])

        print(f"[build_splits] WARNING: {len(no_source_samples)} samples without source_file")

    print(f"[build_splits] train: {len(train_works)} works -> {len(train_samples)} samples")
    print(f"[build_splits] val: {len(val_works)} works -> {len(val_samples)} samples")
    print(f"[build_splits] test: {len(test_works)} works -> {len(test_samples)} samples")

    return train_samples, val_samples, test_samples


def _write_jsonl(samples: List[Sample], path: Path) -> None:
    """Write samples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.raw_data, ensure_ascii=False) + "\n")


def _print_stats(name: str, samples: List[Sample]) -> None:
    """Print a brief summary of the split."""
    if not samples:
        print(f"[build_splits] {name}: empty")
        return

    base_works = set(_get_base_work(s.source_file) for s in samples if s.source_file)
    print(f"[build_splits] {name}: {len(samples)} samples from {len(base_works)} works")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Split dataset by base work to avoid leakage")
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> int:
    """CLI entry point for dataset splitting."""
    args = build_arg_parser().parse_args()

    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_jsonl.exists():
        print(f"[build_splits] not found: {input_jsonl}")
        return 2

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1.0:
        print(f"[build_splits] invalid ratios: train={args.train_ratio}, val={args.val_ratio}")
        return 2

    print(f"[build_splits] loading {input_jsonl}")
    samples = _load_from_jsonl(input_jsonl)
    print(f"[build_splits] loaded {len(samples)} samples")

    train, val, test = _train_val_test_split(samples, args.train_ratio, args.val_ratio, args.seed)

    _write_jsonl(train, output_dir / "train.jsonl")
    _write_jsonl(val, output_dir / "val.jsonl")
    _write_jsonl(test, output_dir / "test.jsonl")

    _print_stats("train", train)
    _print_stats("val", val)
    _print_stats("test", test)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
