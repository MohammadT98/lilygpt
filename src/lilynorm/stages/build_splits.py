# python -m lilynorm.stages.build_splits

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


DEFAULT_TOKENIZED_ROOT = "data/tokenized_dataset"
DEFAULT_OUTPUT_DIR = "data/splits"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42


@dataclass
class Sample:
    """Representation of a single tokenized LilyPond piece."""
    id: str
    rel_path: str
    input_ids: List[int]
    attention_mask: List[int] | None
    token_count: int | None
    truncated: bool | None


def _load_token_file(path: Path, tokenized_root: Path) -> Sample:
    """Load a single *.tokens.json file and wrap it into a Sample."""
    with path.open("r", encoding="utf-8") as f:
        obj: Dict[str, Any] = json.load(f)

    # Old files might only have input_ids; handle that gracefully.
    input_ids = obj.get("input_ids")
    if not isinstance(input_ids, list):
        raise ValueError(f"{path} does not contain 'input_ids' list")

    attention_mask = obj.get("attention_mask")
    if attention_mask is not None and not isinstance(attention_mask, list):
        attention_mask = None

    token_count = obj.get("token_count")
    if not isinstance(token_count, int):
        token_count = len(input_ids)

    truncated = obj.get("truncated")
    if not isinstance(truncated, bool):
        truncated = False

    rel_path = str(path.relative_to(tokenized_root))
    sample_id = path.stem.replace(".tokens", "")

    return Sample(
        id=sample_id,
        rel_path=rel_path,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_count=token_count,
        truncated=truncated,
    )


def _scan_tokenized(tokenized_root: Path) -> List[Sample]:
    """Collect all tokenized pieces under tokenized_root."""
    token_files = sorted(tokenized_root.rglob("*.tokens.json"))
    samples: List[Sample] = []

    if not token_files:
        raise RuntimeError(f"No *.tokens.json files found under {tokenized_root}")

    for path in token_files:
        try:
            sample = _load_token_file(path, tokenized_root)
        except Exception as exc:
            print(f"[build_splits] ! skipping {path} due to error: {exc}")
            continue
        samples.append(sample)

    if not samples:
        raise RuntimeError("No valid tokenized samples were loaded")

    return samples


def _train_val_test_split(
    samples: List[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[List[Sample], List[Sample], List[Sample]]:
    """Randomly split samples into train/val/test."""
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def select(idxs: List[int]) -> List[Sample]:
        return [samples[i] for i in idxs]

    return select(train_idx), select(val_idx), select(test_idx)


def _write_jsonl(samples: List[Sample], path: Path) -> None:
    """Write samples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            obj: Dict[str, Any] = {
                "id": s.id,
                "rel_path": s.rel_path,
                "input_ids": s.input_ids,
            }
            if s.attention_mask is not None:
                obj["attention_mask"] = s.attention_mask
            if s.token_count is not None:
                obj["token_count"] = s.token_count
            if s.truncated is not None:
                obj["truncated"] = s.truncated

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _print_stats(name: str, samples: List[Sample]) -> None:
    lengths = [len(s.input_ids) for s in samples]
    if not lengths:
        print(f"[build_splits] {name}: empty set")
        return

    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    min_len = min(lengths)
    truncated_count = sum(1 for s in samples if s.truncated)

    print(
        f"[build_splits] {name}: "
        f"n={len(samples)} "
        f"avg_len={avg_len:.1f} "
        f"min_len={min_len} max_len={max_len} "
        f"truncated={truncated_count}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build train/val/test splits from tokenized LilyPond dataset."
    )
    parser.add_argument(
        "--tokenized-root",
        default=DEFAULT_TOKENIZED_ROOT,
        help=f"Root directory containing *.tokens.json files (default: {DEFAULT_TOKENIZED_ROOT}).",
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
        help=f"Fraction of samples to use for training (default: {DEFAULT_TRAIN_RATIO}).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Fraction of samples to use for validation (default: {DEFAULT_VAL_RATIO}). "
             "The remaining samples are used for test.",
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

    tokenized_root = Path(args.tokenized_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not tokenized_root.exists():
        print(f"[build_splits] tokenized root not found: {tokenized_root}")
        return 2

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1.0:
        print(
            "[build_splits] invalid split ratios: "
            f"train={args.train_ratio}, val={args.val_ratio}"
        )
        return 2

    print(f"[build_splits] loading tokenized samples from {tokenized_root}")
    samples = _scan_tokenized(tokenized_root)

    print(
        f"[build_splits] splitting {len(samples)} samples "
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
    print(f"  train → {train_path}")
    print(f"  val   → {val_path}")
    print(f"  test  → {test_path}")

    _print_stats("train", train)
    _print_stats("val", val)
    _print_stats("test", test)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())