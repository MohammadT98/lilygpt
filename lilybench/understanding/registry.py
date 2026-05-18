"""Registry for understanding tasks.

Use :func:`register_task` as a decorator on an :class:`UnderstandingTask`
subclass. The registered class is instantiated lazily so subclasses can
hold cheap, picklable state in ``__init__`` if needed.
"""

from __future__ import annotations

from typing import Iterator

from lilybench.understanding.base import UnderstandingTask


TASK_REGISTRY: dict[str, type[UnderstandingTask]] = {}


def register_task(cls: type[UnderstandingTask]) -> type[UnderstandingTask]:
    """Decorator: add ``cls`` to the registry under ``cls.name``."""
    if not cls.name:
        raise ValueError(f"task class {cls.__name__} lacks a non-empty `name`")
    if cls.name in TASK_REGISTRY:
        raise KeyError(f"task {cls.name!r} already registered")
    TASK_REGISTRY[cls.name] = cls
    return cls


def get_task(name: str) -> UnderstandingTask:
    """Instantiate the registered task class for ``name``."""
    if name not in TASK_REGISTRY:
        known = ", ".join(sorted(TASK_REGISTRY))
        raise KeyError(f"unknown task {name!r}; known: {known}")
    return TASK_REGISTRY[name]()


def iter_tasks() -> Iterator[UnderstandingTask]:
    """Yield one instance per registered task, in name order."""
    for name in sorted(TASK_REGISTRY):
        yield TASK_REGISTRY[name]()
