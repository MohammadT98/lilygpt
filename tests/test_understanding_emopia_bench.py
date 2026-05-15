"""Tests for the EMOPIA corpus loader + emotion bench builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from lilybench.understanding import dataset_builder, tasks
from lilybench.understanding.bar_utils import count_bars


def _write_ly(path: Path, n_bars: int) -> None:
    """Write a synthetic .ly with ``n_bars`` bar-delimited segments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = " | ".join("c4 d4 e4 f4" for _ in range(n_bars)) + " |\n"
    path.write_text(
        '\\version "2.24.0"\n'
        'foo = {\n'
        '  \\key c \\major\n'
        '  \\time 4/4\n'
        + body
        + "}\n",
        encoding="utf-8",
    )


@pytest.fixture
def fake_emopia(tmp_path: Path) -> tuple[Path, Path]:
    """12 synthetic EMOPIA-shaped .ly files (3 per quadrant) + manifest CSV.

    Returns (manifest_path, ly_root).
    """
    ly_root = tmp_path / "lilypond"
    ly_root.mkdir()
    rows = []
    for q in ("Q1", "Q2", "Q3", "Q4"):
        for i in range(3):
            clip_id = f"{q}_song{i}_clip0"
            song_id = f"{q}_song{i}"
            # 20 bars so we can verify the 16-bar truncation kicks in.
            _write_ly(ly_root / f"{clip_id}.ly", n_bars=20)
            rows.append({
                "clip_id": clip_id,
                "song_id": song_id,
                "label": q,
                "ly_path": f"{clip_id}.ly",
                "n_bars_full": 20,
                "n_bars_truncated": 16,
            })
    manifest = tmp_path / "emopia_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return manifest, ly_root


def test_load_emotion_corpus_returns_entries(fake_emopia):
    manifest, ly_root = fake_emopia
    corpus = dataset_builder.load_emotion_corpus(manifest, ly_root, max_bars=16)
    assert len(corpus) == 12
    labels = {e.label for e in corpus}
    assert labels == {"Q1", "Q2", "Q3", "Q4"}
    # Every entry has text and the text is truncated to ≤16 bars.
    for e in corpus:
        assert e.text
        assert count_bars(e.text) <= 16


def test_load_emotion_corpus_truncates_to_16_bars(fake_emopia):
    manifest, ly_root = fake_emopia
    corpus = dataset_builder.load_emotion_corpus(manifest, ly_root, max_bars=16)
    for e in corpus:
        # Each fixture file has 20 bars; truncation must bring it to 16.
        assert count_bars(e.text) == 16


def test_load_emotion_corpus_skips_missing_files(tmp_path: Path):
    ly_root = tmp_path / "lilypond"
    ly_root.mkdir()
    _write_ly(ly_root / "Q1_only_one.ly", n_bars=8)
    manifest = tmp_path / "m.csv"
    rows = [
        {"clip_id": "Q1_only_one", "song_id": "s", "label": "Q1",
         "ly_path": "Q1_only_one.ly", "n_bars_full": 8, "n_bars_truncated": 8},
        {"clip_id": "Q2_missing", "song_id": "s2", "label": "Q2",
         "ly_path": "Q2_missing.ly", "n_bars_full": 0, "n_bars_truncated": 0},
    ]
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    corpus = dataset_builder.load_emotion_corpus(manifest, ly_root)
    assert len(corpus) == 1
    assert corpus[0].label == "Q1"


def test_build_emotion_bench_balanced_quadrants(fake_emopia):
    manifest, ly_root = fake_emopia
    corpus = dataset_builder.load_emotion_corpus(manifest, ly_root)
    bench = dataset_builder.build_emotion_bench(corpus, seed=1234, n=8)
    assert len(bench) == 8
    # Balanced: 2 per quadrant (floor of 8/4).
    from collections import Counter
    counts = Counter(r["gold"] for r in bench)
    assert counts == {"Q1": 2, "Q2": 2, "Q3": 2, "Q4": 2}


def test_build_emotion_bench_options_are_the_four_quadrants(fake_emopia):
    manifest, ly_root = fake_emopia
    corpus = dataset_builder.load_emotion_corpus(manifest, ly_root)
    bench = dataset_builder.build_emotion_bench(corpus, seed=1234, n=8)
    quads = {"Q1", "Q2", "Q3", "Q4"}
    for r in bench:
        assert set(r["options"]) == quads
        assert r["gold"] in r["options"]
        assert r["gold_index"] == r["options"].index(r["gold"])
        assert r["template_kind"] == "multiple_choice"
        assert r["task"] == "emotion_recognition"


def test_build_emotion_bench_byte_stable_under_seed(fake_emopia):
    manifest, ly_root = fake_emopia
    corpus = dataset_builder.load_emotion_corpus(manifest, ly_root)
    a = dataset_builder.build_emotion_bench(corpus, seed=1234, n=8)
    b = dataset_builder.build_emotion_bench(corpus, seed=1234, n=8)
    assert a == b


def test_build_emotion_bench_record_shape_compatible_with_infer(fake_emopia):
    """Every key required by lilybench.infer_understanding._load_bench
    + _build_chat_prompt must be present on each record."""
    manifest, ly_root = fake_emopia
    corpus = dataset_builder.load_emotion_corpus(manifest, ly_root)
    bench = dataset_builder.build_emotion_bench(corpus, seed=42, n=4)
    required = {"task", "id", "source_file", "input_content",
                "task_instruction", "options", "gold", "gold_index",
                "template_kind", "prompt"}
    for r in bench:
        missing = required - r.keys()
        assert not missing, f"record missing keys: {missing}"


def test_build_emotion_bench_prompt_contains_options(fake_emopia):
    manifest, ly_root = fake_emopia
    corpus = dataset_builder.load_emotion_corpus(manifest, ly_root)
    bench = dataset_builder.build_emotion_bench(corpus, seed=42, n=4)
    for r in bench:
        for q in ("Q1", "Q2", "Q3", "Q4"):
            assert q in r["prompt"]
        assert "Russell" in r["prompt"]


def test_emotion_recognition_task_is_registered():
    assert "emotion_recognition" in tasks.TASKS
    spec = tasks.TASKS["emotion_recognition"]
    assert spec.template_kind == "multiple_choice"
    assert spec.n == 120


def test_build_emotion_bench_uniform_records_when_n_not_divisible(fake_emopia):
    """n=10 → floor(10/4)=2 per quadrant → 8 records total (drops the remainder)."""
    manifest, ly_root = fake_emopia
    corpus = dataset_builder.load_emotion_corpus(manifest, ly_root)
    bench = dataset_builder.build_emotion_bench(corpus, seed=1234, n=10)
    from collections import Counter
    counts = Counter(r["gold"] for r in bench)
    assert all(c == 2 for c in counts.values())
    assert sum(counts.values()) == 8
