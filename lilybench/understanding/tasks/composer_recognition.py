"""Identify the score's composer from four candidates (4-way MC)."""

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


_COMPOSER_FIELD_RE = re.compile(r'(composer\s*=\s*)"[^"]*"')


def _strip_composer(ly_text: str) -> str:
    return _COMPOSER_FIELD_RE.sub(r'\1""', ly_text)


@register_task
class ComposerRecognition(UnderstandingTask):
    name = "composer_recognition"
    template_kind = "multiple_choice"
    default_n = 96
    task_instruction = (
        "Select the most likely composer of the score from the four options."
    )

    def build(
        self, corpus: Sequence[CorpusEntry], *, n: int, seed: int
    ) -> list[UnderstandingRecord]:
        rng = task_rng(seed, self.name)
        pool = list(dict.fromkeys(c.composer for c in corpus if c.composer))
        if len(pool) < 4:
            return []
        records: list[UnderstandingRecord] = []
        for i, entry in enumerate(pick_subset(corpus, n, rng)):
            gold = entry.composer
            if not gold:
                continue
            try:
                distractors = sample_distractors(pool, gold, rng, k=3)
            except ValueError:
                continue
            options = distractors + [gold]
            rng.shuffle(options)
            gold_index = options.index(gold)
            stripped = _strip_composer(entry.text)
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
