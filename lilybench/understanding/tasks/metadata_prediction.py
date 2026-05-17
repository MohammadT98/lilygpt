"""Predict a masked metadata field from musical content (4-way MC)."""

from __future__ import annotations

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
from lilybench.understanding.score_metadata import mask_field
from lilybench.understanding.tasks.metadata_qa import METADATA_FIELDS


@register_task
class MetadataPrediction(UnderstandingTask):
    name = "metadata_prediction"
    template_kind = "multiple_choice"
    default_n = 60
    task_instruction = (
        "The score has one metadata field masked. Predict the value of "
        "that field from the four options based on the musical content."
    )

    def build(
        self, corpus: Sequence[CorpusEntry], *, n: int, seed: int
    ) -> list[UnderstandingRecord]:
        rng = task_rng(seed, self.name)
        pools: dict[str, list[str]] = {f: [] for f in METADATA_FIELDS}
        for e in corpus:
            for f in METADATA_FIELDS:
                v = getattr(e, f)
                if v:
                    pools[f].append(v)
        pools = {f: list(dict.fromkeys(v)) for f, v in pools.items()}

        eligible = [c for c in corpus if any(getattr(c, f) for f in METADATA_FIELDS)]
        records: list[UnderstandingRecord] = []
        for i, entry in enumerate(pick_subset(eligible, n, rng)):
            field = METADATA_FIELDS[i % len(METADATA_FIELDS)]
            gold = getattr(entry, field)
            if not gold:
                for alt in METADATA_FIELDS:
                    if getattr(entry, alt):
                        field, gold = alt, getattr(entry, alt)
                        break
                else:
                    continue
            pool = pools[field]
            if len([x for x in pool if x != gold]) < 3:
                continue
            distractors = sample_distractors(pool, gold, rng, k=3)
            options = distractors + [gold]
            rng.shuffle(options)
            masked = mask_field(entry.text, field)
            task_instruction = f"{self.task_instruction} Masked field: {field}."
            prompt = format_mc_prompt(
                input_content=masked,
                task_instruction=task_instruction,
                options=options,
            )
            records.append(UnderstandingRecord(
                task=self.name,
                id=f"{self.name}_{i:04d}",
                source_file=entry.source_file,
                input_content=masked,
                task_instruction=task_instruction,
                prompt=prompt,
                template_kind=self.template_kind,
                options=options,
                gold=gold,
                gold_index=options.index(gold),
                extras={"question_field": field},
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
