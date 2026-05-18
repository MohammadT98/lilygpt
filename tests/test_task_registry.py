"""Tests for the understanding task registry."""

from __future__ import annotations

import pytest

from lilybench.understanding import TASK_REGISTRY, UnderstandingTask
from lilybench.understanding.registry import get_task, iter_tasks, register_task


PAPER_TASKS = {
    "bar_count",
    "bar_sequencing",
    "composer_recognition",
    "emotion_recognition",
    "error_detection",
    "genre_recognition",
    "metadata_prediction",
    "metadata_qa",
    "music_captioning",
    "next_bar_prediction",
}


def test_ten_paper_tasks_are_registered():
    assert set(TASK_REGISTRY) == PAPER_TASKS


def test_get_task_returns_instance():
    task = get_task("bar_count")
    assert isinstance(task, UnderstandingTask)
    assert task.name == "bar_count"
    assert task.template_kind == "structured_output"


def test_iter_tasks_alphabetical():
    names = [t.name for t in iter_tasks()]
    assert names == sorted(PAPER_TASKS)


def test_register_task_decorator_adds_to_registry():
    @register_task
    class NoveltyTask(UnderstandingTask):
        name = "test_novelty"
        template_kind = "multiple_choice"
        task_instruction = "x"

        def build(self, corpus, *, n, seed):
            return []

        def score(self, bench, predictions):
            return {"task": self.name}

    try:
        assert "test_novelty" in TASK_REGISTRY
        assert isinstance(get_task("test_novelty"), NoveltyTask)
    finally:
        TASK_REGISTRY.pop("test_novelty", None)


def test_register_task_without_name_raises():
    with pytest.raises(ValueError):
        @register_task
        class Unnamed(UnderstandingTask):
            template_kind = "multiple_choice"
            task_instruction = "x"

            def build(self, corpus, *, n, seed):
                return []

            def score(self, bench, predictions):
                return {}


def test_register_task_collision_raises():
    with pytest.raises(KeyError):
        @register_task
        class Duplicate(UnderstandingTask):
            name = "bar_count"
            template_kind = "structured_output"
            task_instruction = "x"

            def build(self, corpus, *, n, seed):
                return []

            def score(self, bench, predictions):
                return {}
