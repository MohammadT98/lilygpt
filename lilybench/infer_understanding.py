from __future__ import annotations

"""Zero-shot inference for the music-understanding benchmark.

Mirrors :mod:`lilybench.infer` but emits short answers (one digit for MC
tasks, an integer for ``bar_count``, a 4-digit permutation for
``bar_sequencing``) instead of full LilyPond scores. Temperature is fixed at
zero (greedy decoding) to match the ABC-Eval paper.

Invocation::

    python -m lilybench.infer_understanding model=phi4 \\
        bench_path=data/understanding/bench.jsonl \\
        output_dir=data/understanding/predictions

Sweep four models on the cluster::

    python -m lilybench.infer_understanding --multirun \\
        model=phi4,qwen-coder,codestral,deepseek-coder \\
        hydra/launcher=slurm_infer
"""

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from lilybench.hf_cache import apply_hf_env
from lilybench.infer import (  # reuse cache compat shims + chat fallback
    _DTYPE_MAP,  # noqa: F401  (imported for side-effect via lilybench.infer)
)
from lilybench.models import get_spec

BANNER_LINE = "=" * 80
_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _parse_tasks(value: str | list[str] | None) -> set[str] | None:
    """Return the set of task names to run, or ``None`` for all."""
    if value is None or value == "all":
        return None
    if isinstance(value, str):
        return {t.strip() for t in value.split(",") if t.strip()}
    return set(value)


def _load_bench(path: Path, only: set[str] | None, limit: int | None) -> list[dict]:
    records: list[dict] = []
    per_task: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            task = r.get("task")
            if only is not None and task not in only:
                continue
            if limit is not None and per_task[task] >= limit:
                continue
            per_task[task] += 1
            records.append(r)
    return records


def _build_chat_prompt(prompt_body: str, tokenizer) -> str:
    """Wrap the prompt in the model's chat template, falling back when absent."""
    system = (
        "You are a music-theory assistant. Read the LilyPond score in the "
        "Input section and answer the multiple-choice question exactly as "
        "instructed."
    )
    if not getattr(tokenizer, "chat_template", None):
        return f"### System:\n{system}\n\n### User:\n{prompt_body}\n\n### Assistant:\n"
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt_body},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


_DIGIT03_RE = re.compile(r"[0-3]")
_DIGIT_RUN_RE = re.compile(r"\d+")


def _parse_answer(raw: str, *, task: str, template_kind: str) -> str:
    """Extract the predicted answer from the raw model output."""
    text = raw.strip()
    if template_kind == "multiple_choice":
        m = _DIGIT03_RE.search(text)
        return m.group(0) if m else ""
    if task == "bar_count":
        m = _DIGIT_RUN_RE.search(text)
        return m.group(0) if m else ""
    if task == "bar_sequencing":
        runs = _DIGIT_RUN_RE.findall(text)
        return max(runs, key=len) if runs else ""
    # Generic fallback: longest digit run.
    runs = _DIGIT_RUN_RE.findall(text)
    return max(runs, key=len) if runs else ""


@hydra.main(config_path="../configs", config_name="infer_understanding", version_base=None)
def main(cfg: DictConfig) -> int:
    apply_hf_env(cfg.get("hf"))

    bench_path = Path(cfg.bench_path).expanduser().resolve()
    if not bench_path.exists():
        print(f"[infer-und] bench not found: {bench_path}", file=sys.stderr)
        return 2

    only = _parse_tasks(cfg.get("tasks"))
    limit = cfg.get("limit")
    if limit is not None:
        limit = int(limit)
    records = _load_bench(bench_path, only=only, limit=limit)
    if not records:
        print("[infer-und] bench is empty after filtering", file=sys.stderr)
        return 2

    spec = get_spec(cfg.model.id)
    out_root = Path(cfg.output_dir).expanduser().resolve() / spec.model_id
    out_root.mkdir(parents=True, exist_ok=True)

    print(BANNER_LINE)
    print("LilyBench understanding inference")
    print(BANNER_LINE)
    print(OmegaConf.to_yaml(cfg))
    print(f"[infer-und] {len(records)} records over {len({r['task'] for r in records})} tasks")
    print(BANNER_LINE)

    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, use_fast=True, trust_remote_code=spec.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _DTYPE[spec.dtype]
    print(f"[infer-und] loading model: {spec.hf_id} ({spec.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
    )
    model.eval()
    device = next(model.parameters()).device

    max_new_tokens = int(cfg.max_new_tokens)
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # One JSONL per task. Open lazily to avoid empty files for filtered-out tasks.
    file_handles: dict[str, Any] = {}
    try:
        for i, rec in enumerate(records):
            task = rec["task"]
            template_kind = rec.get("template_kind") or "multiple_choice"
            prompt_body = rec["prompt"]
            prompt = _build_chat_prompt(prompt_body, tokenizer)

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            latency_ms = int((time.perf_counter() - t0) * 1000)

            gen_ids = out[0][prompt_len:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
            marker = spec.generation_end_marker
            if marker and marker in raw:
                raw = raw.split(marker, 1)[0]
            parsed = _parse_answer(raw, task=task, template_kind=template_kind)

            if task not in file_handles:
                file_handles[task] = (out_root / f"{task}.jsonl").open(
                    "w", encoding="utf-8"
                )
            file_handles[task].write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "raw_output": raw,
                        "parsed_answer": parsed,
                        "latency_ms": latency_ms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if (i + 1) % 25 == 0 or i + 1 == len(records):
                print(f"[infer-und] {i + 1}/{len(records)} ({task}: parsed={parsed!r})")
    finally:
        for fh in file_handles.values():
            fh.close()

    print(f"[infer-und] wrote predictions to {out_root}")
    return 0


if __name__ == "__main__":
    main()
