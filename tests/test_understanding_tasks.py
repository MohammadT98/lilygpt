"""Tests for the music-understanding task definitions and helpers.

Covers the task registry (exists, expected sizes, expected template kinds)
and the distractor sampler + prompt formatters that the dataset builder
relies on.
"""

from __future__ import annotations

import random

import pytest

from lilybench.understanding import tasks


def test_task_registry_has_all_paper_tasks():
    expected = {
        "bar_count",
        "metadata_qa",
        "bar_sequencing",
        "next_bar_prediction",
        "metadata_prediction",
        "music_captioning",
        "composer_recognition",
        "genre_recognition",
        "emotion_recognition",
        "error_detection",
    }
    assert set(tasks.TASKS.keys()) == expected


def test_task_sample_sizes_match_paper():
    sizes = {name: spec.n for name, spec in tasks.TASKS.items()}
    assert sizes == {
        "bar_count": 100,
        "metadata_qa": 60,
        "bar_sequencing": 119,
        "next_bar_prediction": 119,
        "metadata_prediction": 60,
        "music_captioning": 60,
        "composer_recognition": 96,
        "genre_recognition": 132,
        "emotion_recognition": 120,
        "error_detection": 220,
    }


def test_template_kinds():
    mc = {
        "metadata_qa",
        "next_bar_prediction",
        "metadata_prediction",
        "music_captioning",
        "composer_recognition",
        "genre_recognition",
        "emotion_recognition",
    }
    structured = {"bar_count", "bar_sequencing", "error_detection"}
    for name, spec in tasks.TASKS.items():
        if name in mc:
            assert spec.template_kind == "multiple_choice", name
        elif name in structured:
            assert spec.template_kind == "structured_output", name


def test_task_instructions_are_nonempty():
    for name, spec in tasks.TASKS.items():
        assert spec.task_instruction.strip(), name


def test_sample_distractors_returns_three_unique_excluding_gold():
    pool = ["Bach", "Mozart", "Chopin", "Beethoven", "Handel", "Schubert"]
    rng = random.Random(0)
    distractors = tasks.sample_distractors(pool, gold="Bach", rng=rng, k=3)
    assert len(distractors) == 3
    assert len(set(distractors)) == 3
    assert "Bach" not in distractors
    assert all(d in pool for d in distractors)


def test_sample_distractors_deterministic_under_seed():
    pool = ["a", "b", "c", "d", "e", "f", "g"]
    r1 = random.Random(42)
    r2 = random.Random(42)
    assert tasks.sample_distractors(pool, "a", r1) == tasks.sample_distractors(pool, "a", r2)


def test_sample_distractors_raises_when_pool_too_small():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        tasks.sample_distractors(["a", "b"], gold="a", rng=rng, k=3)


def test_format_mc_prompt_matches_paper_template():
    out = tasks.format_mc_prompt(
        input_content="<score>",
        task_instruction="Identify the composer.",
        options=["Bach", "Mozart", "Chopin", "Beethoven"],
    )
    # Required structural elements per the paper.
    assert "Input: <score>" in out
    assert "Task: Identify the composer." in out
    assert "Options:" in out
    assert "0. Bach" in out
    assert "1. Mozart" in out
    assert "2. Chopin" in out
    assert "3. Beethoven" in out
    assert "0, 1, 2, or 3" in out
    assert "additional content" in out


def test_format_structured_prompt_matches_paper_template():
    out = tasks.format_structured_prompt(
        input_content="<score>",
        task_instruction="How many bars are in the score?",
        structured_output_template="<integer>",
    )
    assert "Input: <score>" in out
    assert "Task: How many bars are in the score?" in out
    assert "Template: <integer>" in out
    assert "without any explanation" in out


def test_format_mc_prompt_requires_four_options():
    with pytest.raises(ValueError):
        tasks.format_mc_prompt(
            input_content="x",
            task_instruction="x",
            options=["a", "b", "c"],
        )
