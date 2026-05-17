"""Reconstruct the correct order of four shuffled bars (structured output)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from lilybench.data.types import CorpusEntry
from lilybench.understanding._helpers import align_by_id, pick_subset, task_rng
from lilybench.understanding.bar_utils import count_bars, split_bars
from lilybench.understanding.base import (
    UnderstandingRecord,
    UnderstandingTask,
    format_structured_prompt,
)
from lilybench.understanding.registry import register_task


@register_task
class BarSequencing(UnderstandingTask):
    name = "bar_sequencing"
    template_kind = "structured_output"
    default_n = 119
    task_instruction = (
        "The bars below are shuffled and labelled with indices 0..3. "
        "Output the indices in the correct sequential order as a single "
        "digit string (for example, 0312)."
    )
    structured_output_template = "<four-digit index permutation, no separator>"

    def build(
        self, corpus: Sequence[CorpusEntry], *, n: int, seed: int
    ) -> list[UnderstandingRecord]:
        rng = task_rng(seed, self.name)
        eligible = [c for c in corpus if count_bars(c.text) >= 4]
        records: list[UnderstandingRecord] = []
        for i, entry in enumerate(pick_subset(eligible, n, rng)):
            bars = split_bars(entry.text)[:4]
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [bars[j] for j in order]
            gold = "".join(str(order.index(j)) for j in range(4))
            input_content = "\n".join(f"{idx}. {seg}" for idx, seg in enumerate(shuffled))
            prompt = format_structured_prompt(
                input_content=input_content,
                task_instruction=self.task_instruction,
                structured_output_template=self.structured_output_template,
            )
            records.append(UnderstandingRecord(
                task=self.name,
                id=f"{self.name}_{i:04d}",
                source_file=entry.source_file,
                input_content=input_content,
                task_instruction=self.task_instruction,
                prompt=prompt,
                template_kind=self.template_kind,
                gold=gold,
                structured_output_template=self.structured_output_template,
                extras={"shuffled_indices": order},
            ))
        return records

    def score(
        self, bench: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        from lilybench.metrics.understanding import bar_sequencing_score

        aligned = align_by_id(bench, predictions)
        if not aligned:
            return {"task": self.name, "n": 0, "score": 0.0, "n_valid": 0}
        per_item = [
            bar_sequencing_score(pred=p["parsed_answer"], gold=b["gold"])
            for b, p in aligned
        ]
        return {
            "task": self.name,
            "n": len(aligned),
            "score": sum(per_item) / len(per_item),
            "n_valid": sum(1 for s in per_item if s > 0),
        }
