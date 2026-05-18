"""Understanding benchmark: ten ABC-Eval-adapted tasks on LilyPond.

The subpackage exposes a small task registry so new tasks can be added
without modifying the runner:

.. code-block:: python

    from lilybench.understanding import register_task, UnderstandingTask

    @register_task
    class MyTask(UnderstandingTask):
        name = "my_task"
        template_kind = "multiple_choice"
        task_instruction = "..."
        def build(self, corpus, *, n, seed): ...
        def score(self, bench, predictions): ...

Importing the subpackage triggers registration of the ten tasks shipped
with the paper.
"""

from lilybench.understanding.base import (
    PROMPT_TEMPLATES,
    UnderstandingRecord,
    UnderstandingTask,
    format_mc_prompt,
    format_structured_prompt,
)
from lilybench.understanding.registry import (
    TASK_REGISTRY,
    get_task,
    iter_tasks,
    register_task,
)

# Trigger task registration on import.
from lilybench.understanding.tasks import _ensure_registered  # noqa: F401

__all__ = [
    "UnderstandingTask",
    "UnderstandingRecord",
    "PROMPT_TEMPLATES",
    "TASK_REGISTRY",
    "register_task",
    "get_task",
    "iter_tasks",
    "format_mc_prompt",
    "format_structured_prompt",
]
