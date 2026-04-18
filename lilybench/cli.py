from __future__ import annotations

"""CLI entry point for LilyPond dataset building.

Subcommands:

* ``build-dataset`` — reads ``data/bmdataset/preprocessed/`` and writes the
  full-file JSONL used for LoRA training.
* ``build-splits`` — deterministic train/val/test split of that JSONL.
"""

import argparse
from pathlib import Path

from lilybench.stages.dataset import build_fullfile_dataset
from lilybench.stages.splitting import build_splits


def _add_build_dataset_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = build_fullfile_dataset.build_arg_parser()
    p.prog = "lilybench build-dataset"
    sub = subparsers.add_parser(
        "build-dataset",
        parents=[p],
        conflict_handler="resolve",
        add_help=False,
        help="Build the full-file training JSONL from bmdataset/preprocessed.",
    )
    sub.set_defaults(_subcommand=_run_build_dataset)


def _add_build_splits_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = build_splits.build_arg_parser()
    p.prog = "lilybench build-splits"
    sub = subparsers.add_parser(
        "build-splits",
        parents=[p],
        conflict_handler="resolve",
        add_help=False,
        help="Split the training JSONL into train/val/test by source work.",
    )
    sub.set_defaults(_subcommand=_run_build_splits)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lilybench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_build_dataset_subparser(subparsers)
    _add_build_splits_subparser(subparsers)
    return parser


def _run_build_dataset(args: argparse.Namespace) -> int:
    build_fullfile_dataset.build_dataset(
        input_dir=Path(args.input_dir),
        metadata_path=Path(args.metadata),
        output_path=Path(args.output),
        max_chars=args.max_chars,
        global_seed=args.seed,
    )
    return 0


def _run_build_splits(args: argparse.Namespace) -> int:
    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_jsonl.exists():
        print(f"[build_splits] not found: {input_jsonl}")
        return 2
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1.0:
        print(
            f"[build_splits] invalid ratios: train={args.train_ratio}, val={args.val_ratio}"
        )
        return 2
    samples = build_splits._load_from_jsonl(input_jsonl)
    train, val, test = build_splits._train_val_test_split(
        samples, args.train_ratio, args.val_ratio, args.seed
    )
    build_splits._write_jsonl(train, output_dir / "train.jsonl")
    build_splits._write_jsonl(val, output_dir / "val.jsonl")
    build_splits._write_jsonl(test, output_dir / "test.jsonl")
    build_splits._print_stats("train", train)
    build_splits._print_stats("val", val)
    build_splits._print_stats("test", test)
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    return args._subcommand(args)


if __name__ == "__main__":
    raise SystemExit(main())
