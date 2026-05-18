"""Detect synthetically corrupted bars (structured output)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from lilybench.data.types import CorpusEntry
from lilybench.understanding._helpers import align_by_id, task_rng
from lilybench.understanding.bar_utils import count_bars
from lilybench.understanding.base import (
    UnderstandingRecord,
    UnderstandingTask,
    format_structured_prompt,
)
from lilybench.understanding.corruptor import ERROR_CATEGORIES, inject
from lilybench.understanding.registry import register_task


@register_task
class ErrorDetection(UnderstandingTask):
    name = "error_detection"
    template_kind = "structured_output"
    default_n = 220
    task_instruction = (
        "The score below may contain one or more erroneous bars "
        "(invalid metadata, garbage content, wrong bar duration, "
        "implausible melodic leap, or accidental outside the declared key). "
        "Identify the 1-indexed bar numbers where the errors occur. "
        "Output the numbers as a space-separated list (e.g. '3 7 12'). "
        "If no errors are present, output 'none'."
    )
    structured_output_template = "<space-separated integer bar numbers, or 'none'>"

    def build(
        self, corpus: Sequence[CorpusEntry], *, n: int, seed: int
    ) -> list[UnderstandingRecord]:
        rng = task_rng(seed, self.name)
        eligible = [c for c in corpus if count_bars(c.text) >= 4]
        if len(eligible) < n:
            n = len(eligible)
        per_cat = max(1, n // len(ERROR_CATEGORIES))
        rng.shuffle(eligible)
        pool = list(eligible)
        idx = 0
        out: list[UnderstandingRecord] = []
        for category in ERROR_CATEGORIES:
            emitted = 0
            attempts = 0
            max_attempts = 8 * per_cat
            while emitted < per_cat and idx < len(pool) and attempts < max_attempts:
                entry = pool[idx]
                idx += 1
                attempts += 1
                corruption = inject(entry.text, category, rng=rng)
                if corruption is None:
                    continue
                prompt = format_structured_prompt(
                    input_content=corruption.text,
                    task_instruction=self.task_instruction,
                    structured_output_template=self.structured_output_template,
                )
                rec_idx = len(out)
                out.append(UnderstandingRecord(
                    task=self.name,
                    id=f"{self.name}_{rec_idx:04d}",
                    source_file=entry.source_file,
                    input_content=corruption.text,
                    task_instruction=self.task_instruction,
                    prompt=prompt,
                    template_kind=self.template_kind,
                    gold=None,
                    structured_output_template=self.structured_output_template,
                    extras={
                        "gold_bars": list(corruption.error_bars),
                        "category": corruption.category,
                    },
                ))
                emitted += 1
        return out

    def score(
        self, bench: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        from lilybench.metrics.understanding import error_detection_f1, parse_bar_list

        aligned = align_by_id(bench, predictions)
        if not aligned:
            return {"task": self.name, "n": 0, "macro_f1": 0.0}
        per_record: list[float] = []
        by_cat: dict[str, list[float]] = defaultdict(list)
        for b, p in aligned:
            gold = set(b.get("gold_bars") or [])
            raw = p.get("parsed_answer") or p.get("raw_output", "")
            pred = set(parse_bar_list(raw))
            f1 = error_detection_f1(pred=pred, gold=gold)
            per_record.append(f1)
            by_cat[b.get("category", "unknown")].append(f1)
        per_cat_f1 = {c: (sum(v) / len(v) if v else 0.0) for c, v in by_cat.items()}
        macro = sum(per_cat_f1.values()) / len(per_cat_f1) if per_cat_f1 else 0.0
        return {
            "task": self.name,
            "n": len(aligned),
            "macro_f1": macro,
            "mean_f1": sum(per_record) / len(per_record),
            "per_category_f1": per_cat_f1,
        }
