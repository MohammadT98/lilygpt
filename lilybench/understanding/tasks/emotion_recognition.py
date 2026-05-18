"""Predict Russell valence-arousal quadrant (Q1..Q4) from a LilyPond clip."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from lilybench.data.types import CorpusEntry
from lilybench.understanding._helpers import align_by_id, task_rng
from lilybench.understanding.base import (
    UnderstandingRecord,
    UnderstandingTask,
    format_mc_prompt,
)
from lilybench.understanding.registry import register_task


EMOTION_QUADRANTS = ("Q1", "Q2", "Q3", "Q4")


@register_task
class EmotionRecognition(UnderstandingTask):
    name = "emotion_recognition"
    template_kind = "multiple_choice"
    default_n = 120
    corpus_kind = "emopia"
    task_instruction = (
        "Select the most likely Russell valence-arousal quadrant for the score "
        "from the four options. "
        "Q1 = high valence + high arousal, "
        "Q2 = low valence + high arousal, "
        "Q3 = low valence + low arousal, "
        "Q4 = high valence + low arousal."
    )

    def build(
        self, corpus: Sequence[CorpusEntry], *, n: int, seed: int
    ) -> list[UnderstandingRecord]:
        rng = task_rng(seed, self.name)
        per_q = max(1, n // len(EMOTION_QUADRANTS))
        by_q: dict[str, list[CorpusEntry]] = defaultdict(list)
        for e in corpus:
            label = (e.extras or {}).get("label")
            if label in EMOTION_QUADRANTS:
                by_q[label].append(e)

        records: list[UnderstandingRecord] = []
        idx = 0
        for q in EMOTION_QUADRANTS:
            bucket = by_q.get(q, [])
            if not bucket:
                continue
            picked = rng.sample(bucket, min(per_q, len(bucket)))
            for entry in picked:
                options = list(EMOTION_QUADRANTS)
                rng.shuffle(options)
                gold_index = options.index(q)
                prompt = format_mc_prompt(
                    input_content=entry.text,
                    task_instruction=self.task_instruction,
                    options=options,
                )
                records.append(UnderstandingRecord(
                    task=self.name,
                    id=f"{self.name}_{idx:04d}",
                    source_file=entry.source_file,
                    input_content=entry.text,
                    task_instruction=self.task_instruction,
                    prompt=prompt,
                    template_kind=self.template_kind,
                    options=options,
                    gold=q,
                    gold_index=gold_index,
                    extras={
                        "clip_id": entry.extras.get("clip_id"),
                        "song_id": entry.extras.get("song_id"),
                    },
                ))
                idx += 1
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
        # Confusion matrix Q-gold → Q-predicted.
        quad_set = set(EMOTION_QUADRANTS)
        matrix = {g: {p: 0 for p in EMOTION_QUADRANTS} for g in EMOTION_QUADRANTS}
        n_off_grid = 0
        for b, p in aligned:
            gold = b.get("gold")
            if gold not in quad_set:
                continue
            ans = p.get("parsed_answer", "")
            pred_label = None
            if ans in {"0", "1", "2", "3"}:
                try:
                    pred_label = b["options"][int(ans)]
                except (KeyError, IndexError, ValueError):
                    pred_label = None
            if pred_label in quad_set:
                matrix[gold][pred_label] += 1
            else:
                n_off_grid += 1
        return {
            "task": self.name,
            "n": len(aligned),
            "accuracy": accuracy(preds, golds),
            "n_parsed": sum(1 for s in preds if s in {"0", "1", "2", "3"}),
            "confusion": {"matrix": matrix, "n_off_grid": n_off_grid},
        }
