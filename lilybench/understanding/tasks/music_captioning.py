"""Identify the score's title from four candidate titles (4-way MC)."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from lilybench.data.types import CorpusEntry
from lilybench.understanding._helpers import (
    align_by_id,
    pick_subset,
    sample_distractors,
    task_rng,
)
from lilybench.understanding.base import (
    UnderstandingRecord,
    UnderstandingTask,
    format_mc_prompt,
)
from lilybench.understanding.registry import register_task


_TITLE_FIELD_RE = re.compile(r'(title\s*=\s*)"[^"]*"')


def _strip_title(ly_text: str) -> str:
    return _TITLE_FIELD_RE.sub(r'\1""', ly_text)


@register_task
class MusicCaptioning(UnderstandingTask):
    name = "music_captioning"
    template_kind = "multiple_choice"
    default_n = 60
    task_instruction = (
        "Select the most likely title for the score from the four options."
    )

    def build(
        self, corpus: Sequence[CorpusEntry], *, n: int, seed: int
    ) -> list[UnderstandingRecord]:
        rng = task_rng(seed, self.name)
        titled = [c for c in corpus if c.title]
        pool = list(dict.fromkeys(c.title for c in titled if c.title))
        records: list[UnderstandingRecord] = []
        for i, entry in enumerate(pick_subset(titled, n, rng)):
            gold = entry.title
            if not gold or len([t for t in pool if t != gold]) < 3:
                continue
            distractors = sample_distractors(pool, gold, rng, k=3)
            options = distractors + [gold]
            rng.shuffle(options)
            gold_index = options.index(gold)
            stripped = _strip_title(entry.text)
            prompt = format_mc_prompt(
                input_content=stripped,
                task_instruction=self.task_instruction,
                options=options,
            )
            records.append(UnderstandingRecord(
                task=self.name,
                id=f"{self.name}_{i:04d}",
                source_file=entry.source_file,
                input_content=stripped,
                task_instruction=self.task_instruction,
                prompt=prompt,
                template_kind=self.template_kind,
                options=options,
                gold=gold,
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
