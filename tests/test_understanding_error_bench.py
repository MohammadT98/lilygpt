"""Tests for the error_detection bench builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilybench.understanding import dataset_builder, tasks
from lilybench.understanding.corruptor import ERROR_CATEGORIES


def _write_mutopia_corpus(tmp_path: Path, n: int = 20) -> Path:
    """Write n synthetic Mutopia-shaped .ly files + a manifest, return manifest path."""
    import json
    root = tmp_path / "stripped"
    root.mkdir()
    composers = ["Bach", "Mozart", "Chopin", "Handel"]
    styles = ["Baroque", "Classical", "Romantic"]
    pieces = {}
    for i in range(n):
        rel = f"piece_{i:02d}.ly"
        (root / rel).write_text(
            '\\version "2.24.0"\n'
            f'\\header {{\n  title = "Piece {i}"\n}}\n'
            'music = {\n'
            '  \\key c \\major\n'
            '  \\time 4/4\n'
            "  c'4 d' e' f' |\n"
            "  g'4 a' b' c'' |\n"
            "  d''4 c'' b' a' |\n"
            "  g'4 f' e' d' |\n"
            "  c'1 |\n"
            "  d'1 |\n"
            "}\n",
            encoding="utf-8",
        )
        pieces[f"piece_{i:02d}"] = {
            "composer": composers[i % len(composers)],
            "style": styles[i % len(styles)],
            "localPath": rel,
        }
    manifest = tmp_path / "dataset_mutopia.json"
    manifest.write_text(json.dumps(pieces), encoding="utf-8")
    return manifest


def test_error_detection_task_registered():
    assert "error_detection" in tasks.TASKS
    spec = tasks.TASKS["error_detection"]
    assert spec.template_kind == "structured_output"


def test_build_error_bench_emits_records(tmp_path: Path):
    manifest = _write_mutopia_corpus(tmp_path, n=20)
    corpus = dataset_builder.build_corpus(manifest, tmp_path / "stripped")
    bench = dataset_builder.build_error_bench(corpus, seed=1234, n=10)
    assert len(bench) == 10
    for r in bench:
        assert r["task"] == "error_detection"
        assert r["template_kind"] == "structured_output"
        assert isinstance(r["gold_bars"], list)
        assert all(isinstance(b, int) for b in r["gold_bars"])
        assert r["category"] in ERROR_CATEGORIES


def test_build_error_bench_balanced_across_categories(tmp_path: Path):
    manifest = _write_mutopia_corpus(tmp_path, n=40)
    corpus = dataset_builder.build_corpus(manifest, tmp_path / "stripped")
    bench = dataset_builder.build_error_bench(corpus, seed=1234, n=20)
    from collections import Counter
    counts = Counter(r["category"] for r in bench)
    # 20 records / 5 categories = 4 per category (balanced floor).
    assert sum(counts.values()) <= 20
    # Each category that appears should be balanced.
    for v in counts.values():
        assert v in {3, 4, 5}, counts  # allow off-by-one for tight rounding


def test_build_error_bench_byte_stable_under_seed(tmp_path: Path):
    manifest = _write_mutopia_corpus(tmp_path, n=20)
    corpus = dataset_builder.build_corpus(manifest, tmp_path / "stripped")
    a = dataset_builder.build_error_bench(corpus, seed=42, n=10)
    b = dataset_builder.build_error_bench(corpus, seed=42, n=10)
    assert a == b


def test_build_error_bench_record_shape(tmp_path: Path):
    manifest = _write_mutopia_corpus(tmp_path, n=10)
    corpus = dataset_builder.build_corpus(manifest, tmp_path / "stripped")
    bench = dataset_builder.build_error_bench(corpus, seed=7, n=5)
    required = {"task", "id", "source_file", "input_content", "task_instruction",
                "structured_output_template", "gold_bars", "category",
                "template_kind", "prompt"}
    for r in bench:
        missing = required - r.keys()
        assert not missing, f"record missing keys: {missing}"


def test_build_error_bench_gold_bars_within_input(tmp_path: Path):
    """Each gold_bar must refer to a bar that exists in the input."""
    from lilybench.understanding.bar_utils import count_bars

    manifest = _write_mutopia_corpus(tmp_path, n=10)
    corpus = dataset_builder.build_corpus(manifest, tmp_path / "stripped")
    bench = dataset_builder.build_error_bench(corpus, seed=7, n=10)
    for r in bench:
        n_bars = count_bars(r["input_content"])
        for b in r["gold_bars"]:
            assert 1 <= b <= n_bars + 1  # +1 because injection can append
