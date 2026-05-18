"""Smoke tests for the unified CLI argparse wiring."""

from __future__ import annotations

import pytest

from lilybench.cli import build_arg_parser


def test_help_runs():
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_prompt_bank_build_minimal():
    parser = build_arg_parser()
    ns = parser.parse_args([
        "prompt-bank", "build",
        "--bmdataset-dir", "data/bmdataset/preprocessed",
        "--bmdataset-metadata", "data/bmdataset/metadata.json",
        "--out", "/tmp/bank.jsonl",
    ])
    assert ns.command == "prompt-bank"
    assert ns.action == "build"
    assert ns.n == 200
    assert ns.seed == 1234


def test_generation_run_required_fields():
    parser = build_arg_parser()
    ns = parser.parse_args([
        "generation", "run",
        "--model", "phi4",
        "--regime", "zero",
        "--prompts", "/tmp/bank.jsonl",
        "--out", "/tmp/runs/phi4_zero",
    ])
    assert ns.model == "phi4"
    assert ns.regime == "zero"
    assert ns.max_new_tokens == 3000


def test_understanding_build_mutopia():
    parser = build_arg_parser()
    ns = parser.parse_args([
        "understanding", "build",
        "--corpus", "mutopia",
        "--mutopia-manifest", "/tmp/mutopia.json",
        "--out", "/tmp/bench.jsonl",
    ])
    assert ns.corpus == "mutopia"
    assert ns.tasks == "all"


def test_metrics_understanding_requires_dirs():
    parser = build_arg_parser()
    ns = parser.parse_args([
        "metrics", "understanding",
        "--bench", "/tmp/bench.jsonl",
        "--predictions", "/tmp/preds/phi4",
        "--out", "/tmp/summary.json",
    ])
    assert ns.command == "metrics"
    assert ns.action == "understanding"
