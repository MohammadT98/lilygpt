"""Greedy-decoding runner for the understanding benchmark.

Reads a bench JSONL produced by the per-task builders, prompts the
selected backbone with the chat template, decodes ``max_new_tokens``
tokens greedily, and parses the answer according to ``template_kind``
(MC index for multiple choice, integer / digit-string / bar list for
structured output). One JSONL per task is written under
``output_dir/<model_id>/<task>.jsonl``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from lilybench.models import ModelSpec, load_backbone


_DIGIT03_RE = re.compile(r"[0-3]")
_DIGIT_RUN_RE = re.compile(r"\d+")


def parse_answer(raw: str, *, task: str, template_kind: str) -> str:
    """Extract the predicted answer from the model's raw output."""
    text = raw.strip()
    if template_kind == "multiple_choice":
        m = _DIGIT03_RE.search(text)
        return m.group(0) if m else ""
    if task == "bar_count":
        m = _DIGIT_RUN_RE.search(text)
        return m.group(0) if m else ""
    runs = _DIGIT_RUN_RE.findall(text)
    return max(runs, key=len) if runs else ""


@dataclass(frozen=True)
class UnderstandingConfig:
    output_dir: Path
    max_new_tokens: int = 20
    seed: int = 1234
    device_map: str = "auto"
    quantization: str | None = None
    max_input_chars: int | None = None
    extra: dict = field(default_factory=dict)


_SYSTEM_PROMPT = (
    "You are a music-theory assistant. Read the LilyPond score in the "
    "Input section and answer the multiple-choice question exactly as "
    "instructed."
)


def _build_chat_prompt(body: str, tokenizer) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return f"### System:\n{_SYSTEM_PROMPT}\n\n### User:\n{body}\n\n### Assistant:\n"
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _maybe_truncate(rec: dict, limit: int | None) -> str:
    """Truncate ``input_content`` and re-render so question/template stay."""
    body = rec["prompt"]
    if not limit:
        return body
    inp = rec.get("input_content", "")
    if len(inp) <= limit:
        return body
    truncated = inp[:limit] + "\n% ...truncated...\n"
    from lilybench.understanding.base import (
        format_mc_prompt, format_structured_prompt,
    )
    if rec.get("template_kind") == "multiple_choice":
        return format_mc_prompt(
            input_content=truncated,
            task_instruction=rec["task_instruction"],
            options=rec["options"],
        )
    return format_structured_prompt(
        input_content=truncated,
        task_instruction=rec["task_instruction"],
        structured_output_template=rec.get("structured_output_template", ""),
    )


def run_understanding(
    *,
    spec: ModelSpec,
    bench: Iterable[dict],
    cfg: UnderstandingConfig,
) -> Path:
    """Run greedy decoding over ``bench`` and write per-task JSONLs.

    Returns the per-model output root (``output_dir / spec.model_id``).
    """
    out_root = Path(cfg.output_dir).expanduser().resolve() / spec.model_id
    out_root.mkdir(parents=True, exist_ok=True)

    records = list(bench)
    print(f"[lilybench.understanding] model={spec.model_id} n={len(records)} "
          f"tasks={sorted({r['task'] for r in records})}")

    model, tokenizer = load_backbone(
        spec, device_map=cfg.device_map, quantization=cfg.quantization
    )
    device = next(model.parameters()).device
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    end_marker = spec.generation_end_marker

    handles: dict[str, Any] = {}
    try:
        for i, rec in enumerate(records):
            task = rec["task"]
            template_kind = rec.get("template_kind") or "multiple_choice"
            body = _maybe_truncate(rec, cfg.max_input_chars)
            prompt = _build_chat_prompt(body, tokenizer)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            raw = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
            if end_marker and end_marker in raw:
                raw = raw.split(end_marker, 1)[0]
            parsed = parse_answer(raw, task=task, template_kind=template_kind)

            if task not in handles:
                handles[task] = (out_root / f"{task}.jsonl").open("w", encoding="utf-8")
            handles[task].write(json.dumps({
                "id": rec["id"],
                "raw_output": raw,
                "parsed_answer": parsed,
                "latency_ms": latency_ms,
            }, ensure_ascii=False) + "\n")

            if (i + 1) % 25 == 0 or i + 1 == len(records):
                print(f"[lilybench.understanding] {i + 1}/{len(records)} "
                      f"({task}: parsed={parsed!r})")
    finally:
        for fh in handles.values():
            fh.close()
    return out_root
