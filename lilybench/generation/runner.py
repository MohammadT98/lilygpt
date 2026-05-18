"""Run a registered backbone over a prompt bank.

Outputs are written to ``output_dir/samples/sample_####.ly`` (cleaned
text, with a minimal ``\\version`` / ``\\language`` header prepended when
the model omitted them) and ``output_dir/samples/raw_####.txt`` (raw
decoded tokens before fence-stripping). One sample per bank record.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from lilybench.generation.prompt_bank import Prompt
from lilybench.generation.regimes import Regime
from lilybench.models import ModelSpec, load_backbone


@dataclass(frozen=True)
class GenerationConfig:
    output_dir: Path
    max_new_tokens: int = 3000
    temperature: float = 0.7
    top_p: float = 0.9
    seed_base: int = 1234
    lilypond_version: str = "2.24.4"
    lilypond_language: str = "nederlands"
    device_map: str = "auto"
    quantization: str | None = None
    extra: dict = field(default_factory=dict)


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)\n```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n", re.MULTILINE)


def _strip_markdown_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    stripped = _OPEN_FENCE_RE.sub("", text, count=1)
    return stripped.strip() if stripped != text else text


def _ensure_lily_headers(text: str, *, version: str, language: str) -> str:
    body = text.rstrip()
    header_lines: list[str] = []
    if "\\version" not in body:
        header_lines.append(f'\\version "{version}"')
    if "\\language" not in body:
        header_lines.append(f'\\language "{language}"')
    header = ("\n".join(header_lines) + "\n") if header_lines else ""
    return header + body + "\n"


def generate(
    *,
    spec: ModelSpec,
    regime: Regime,
    bank: Iterable[Prompt],
    cfg: GenerationConfig,
) -> Path:
    """Run ``spec`` over ``bank`` under ``regime``; return the samples dir."""
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_backbone(
        spec, device_map=cfg.device_map, quantization=cfg.quantization
    )
    device = next(model.parameters()).device
    end_marker = spec.generation_end_marker

    bank_list = list(bank)
    n = len(bank_list)
    print(f"[lilybench.generate] model={spec.model_id} regime={regime.name} n={n}")

    for i, prompt in enumerate(bank_list):
        seed = cfg.seed_base + i
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        text_in = regime.build_prompt(prompt, tokenizer=tokenizer)
        inputs = tokenizer(text_in, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        if end_marker and end_marker in raw:
            raw = raw.split(end_marker, 1)[0]
        raw = raw.rstrip()
        cleaned = _strip_markdown_fences(raw)
        final = _ensure_lily_headers(
            cleaned.strip(),
            version=cfg.lilypond_version,
            language=cfg.lilypond_language,
        )

        (samples_dir / f"raw_{i:04d}.txt").write_text(raw + "\n", encoding="utf-8")
        (samples_dir / f"sample_{i:04d}.ly").write_text(final, encoding="utf-8")

        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"[lilybench.generate] {i + 1}/{n}")

    return samples_dir
