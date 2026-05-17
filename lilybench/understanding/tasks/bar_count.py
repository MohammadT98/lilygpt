"""Count the number of bars in a LilyPond score (structured output)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from lilybench.data.types import CorpusEntry
from lilybench.understanding._helpers import pick_subset, task_rng
from lilybench.understanding.bar_utils import count_bars
from lilybench.understanding.base import (
    UnderstandingRecord,
    UnderstandingTask,
    format_structured_prompt,
)
from lilybench.understanding.registry import register_task


@register_task
class BarCount(UnderstandingTask):
    name = "bar_count"
    template_kind = "structured_output"
    default_n = 100
    task_instruction = (
        "Count the number of bars in the LilyPond score above. "
        "Output a single integer."
    )
    structured_output_template = "<integer>"

    def build(
        self, corpus: Sequence[CorpusEntry], *, n: int, seed: int
    ) -> list[UnderstandingRecord]:
        eligible = [c for c in corpus if count_bars(c.text) >= 1]
        rng = task_rng(seed, self.name)
        records: list[UnderstandingRecord] = []
        for i, entry in enumerate(pick_subset(eligible, n, rng)):
            bars = count_bars(entry.text)
            prompt = format_structured_prompt(
                input_content=entry.text,
                task_instruction=self.task_instruction,
                structured_output_template=self.structured_output_template,
            )
            records.append(UnderstandingRecord(
                task=self.name,
                id=f"{self.name}_{i:04d}",
                source_file=entry.source_file,
                input_content=entry.text,
                task_instruction=self.task_instruction,
                prompt=prompt,
                template_kind=self.template_kind,
                gold=str(bars),
                structured_output_template=self.structured_output_template,
            ))
        return records

    def score(
        self, bench: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        from lilybench.metrics.understanding import accuracy, bar_count_tolerance

        by_id = {r["id"]: r for r in bench}
        aligned = [(by_id[p["id"]], p) for p in predictions if p["id"] in by_id]
        if not aligned:
            return {"task": self.name, "n": 0, "accuracy": 0.0}
        preds = [p["parsed_answer"] for _, p in aligned]
        golds = [b["gold"] for b, _ in aligned]
        return {
            "task": self.name,
            "n": len(aligned),
            "accuracy": accuracy(preds, golds),
            "n_parsed": sum(1 for s in preds if s.isdigit()),
            "tolerance": bar_count_tolerance(preds, golds),
        }
