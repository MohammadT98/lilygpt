"""Pick the bar that continues a four-bar context (4-way MC)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from lilybench.data.types import CorpusEntry
from lilybench.understanding._helpers import align_by_id, pick_subset, task_rng
from lilybench.understanding.bar_utils import count_bars, split_bars
from lilybench.understanding.base import (
    UnderstandingRecord,
    UnderstandingTask,
    format_mc_prompt,
)
from lilybench.understanding.registry import register_task


CONTEXT_BARS = 4


@register_task
class NextBarPrediction(UnderstandingTask):
    name = "next_bar_prediction"
    template_kind = "multiple_choice"
    default_n = 119
    task_instruction = (
        "Given the opening bars of the score, select the option that is "
        "most likely to be the next bar."
    )

    def build(
        self, corpus: Sequence[CorpusEntry], *, n: int, seed: int
    ) -> list[UnderstandingRecord]:
        rng = task_rng(seed, self.name)
        eligible = [c for c in corpus if count_bars(c.text) >= CONTEXT_BARS + 4]
        records: list[UnderstandingRecord] = []
        for i, entry in enumerate(pick_subset(eligible, n, rng)):
            bars = split_bars(entry.text)
            context = bars[:CONTEXT_BARS]
            gold_bar = bars[CONTEXT_BARS]
            pool = bars[CONTEXT_BARS + 1 :]
            if len(pool) < 3:
                continue
            distractors = rng.sample(pool, 3)
            options = distractors + [gold_bar]
            rng.shuffle(options)
            gold_index = options.index(gold_bar)
            input_content = "\n".join(context)
            prompt = format_mc_prompt(
                input_content=input_content,
                task_instruction=self.task_instruction,
                options=options,
            )
            records.append(UnderstandingRecord(
                task=self.name,
                id=f"{self.name}_{i:04d}",
                source_file=entry.source_file,
                input_content=input_content,
                task_instruction=self.task_instruction,
                prompt=prompt,
                template_kind=self.template_kind,
                options=options,
                gold=str(gold_index),
                gold_index=gold_index,
            ))
        return records

    def score(
        self, bench: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        from lilybench.metrics.understanding import accuracy

        aligned = align_by_id(bench, predictions)
        if not aligned:
            return {"task": self.name, "n": 0, "accuracy": 0.0}
        preds = [p["parsed_answer"] for _, p in aligned]
        golds = [str(b["gold_index"]) for b, _ in aligned]
        return {
            "task": self.name,
            "n": len(aligned),
            "accuracy": accuracy(preds, golds),
            "n_parsed": sum(1 for s in preds if s in {"0", "1", "2", "3"}),
        }
