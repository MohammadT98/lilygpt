from __future__ import annotations

"""Generate LilyPond samples from a registered model (zero/few/lora regime).

Hydra entry point. Run with ``python -m lilybench.infer`` and override config
values on the command line, e.g.::

    python -m lilybench.infer model=phi4 regime=zero num_samples=50
    python -m lilybench.infer model=qwen-coder regime=lora regime.path=runs/qwen/final
    python -m lilybench.infer model=phi4 regime=few regime.fewshot_file=configs/fewshot/phi4.txt

Submit sweeps to SLURM via the submitit launcher::

    python -m lilybench.infer --multirun model=phi4,qwen-coder regime=zero \\
        hydra/launcher=slurm_infer
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from lilybench.hf_cache import apply_hf_env
from lilybench.models import get_spec

# DeepSeek-Coder-V2's vendored modeling_deepseek.py imports
# `is_torch_fx_available` from transformers.utils.import_utils, which was
# removed in transformers 5.x. Re-add it as a module-level alias so the
# trust_remote_code import succeeds.
try:
    import torch.fx as _torch_fx
    import transformers.utils.import_utils as _tx_import_utils
    if not hasattr(_tx_import_utils, "is_torch_fx_available"):
        _tx_import_utils.is_torch_fx_available = lambda: True
except Exception:
    pass

# DeepSeek-Coder-V2's vendored modeling_deepseek.py reads three attributes on
# DynamicCache that newer transformers (>=4.50) removed:
#   - seen_tokens         (now get_seq_length())
#   - get_max_length()    (no replacement; DynamicCache is unbounded)
#   - get_usable_length() (was on Cache base class)
# Shim all three so trust_remote_code models work without forking.
try:
    from transformers.cache_utils import DynamicCache as _DynamicCache
    if not hasattr(_DynamicCache, "seen_tokens"):
        _DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(_DynamicCache, "get_max_length"):
        _DynamicCache.get_max_length = lambda self: None
    if not hasattr(_DynamicCache, "get_usable_length"):
        _DynamicCache.get_usable_length = (
            lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
        )
except Exception:
    pass

BANNER_LINE = "=" * 80
_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

_METADATA_FIELDS = ("composer", "period", "musical_form", "ensemble", "part")


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _load_prompt(cfg: DictConfig) -> str:
    prompt_file = cfg.get("prompt_file")
    if prompt_file:
        return Path(prompt_file).expanduser().read_text(encoding="utf-8").strip()
    return str(cfg.default_prompt).strip()


def _load_prompt_bank(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL prompt bank. Each record must carry at minimum
    ``user_prompt`` (for zero/few) and/or ``metadata`` (for lora)."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _render_metadata_block(metadata: dict[str, Any] | None) -> str:
    """Render a ``%% === METADATA ===`` block matching the training format.

    ``ensemble`` is normalised to a comma-separated string when supplied as a
    list (training uses the comma-joined form, see
    ``lilybench/preprocess/metadata_header.py``). Fields with ``None``/empty
    values are omitted so the block matches a field-dropout variant.
    """
    lines = ["%% === METADATA ==="]
    if metadata:
        for key in _METADATA_FIELDS:
            val = metadata.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, (list, tuple)):
                if not val:
                    continue
                val = ", ".join(str(v) for v in val)
            lines.append(f"%% {key}: {val}")
    lines.append("%% === END METADATA ===")
    return "\n".join(lines) + "\n"


def _build_lora_preamble(
    *, version: str, language: str, metadata: dict[str, Any] | None = None
) -> str:
    """Raw preamble mimicking the training distribution: metadata block, then
    \\version and \\language. The LoRA was trained to continue such text, so
    inference must present the same surface form. When ``metadata`` is None the
    block is empty (back-compat with the single-prompt default)."""
    return (
        _render_metadata_block(metadata)
        + f'\\version "{version}"\n'
        + f'\\language "{language}"\n'
    )


def _build_prompt(
    *,
    regime: str,
    user_prompt_text: str,
    fewshot_text: str,
    tokenizer,
    lilypond_version: str,
    lilypond_language: str,
    model_id: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    if regime == "lora":
        return _build_lora_preamble(
            version=lilypond_version, language=lilypond_language, metadata=metadata
        )

    system = (
        "You are a LilyPond assistant. Output only valid LilyPond code, "
        "no prose, no markdown."
    )
    user = user_prompt_text
    if metadata:
        user = _render_metadata_block(metadata) + user
    if regime == "few":
        user = f"{fewshot_text}\n\n{user}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if not getattr(tokenizer, "chat_template", None):
        # Base (non-instruct) models like gemma-2-27b ship no chat template.
        # Use a generic instruction format so zero/few-shot still produce
        # comparable prompts. Quality is expected to be lower than an
        # instruction-tuned variant; report this in the paper alongside the
        # affected models.
        return (
            f"### System:\n{system}\n\n"
            f"### User:\n{user}\n\n"
            f"### Assistant:\n"
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)\n```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n", re.MULTILINE)


def _strip_markdown_fences(text: str) -> str:
    """Extract LilyPond from a ``` ... ``` fenced block when present.

    Chat-templated zero/few regimes often wrap output in ```lilypond\n...\n```
    despite the system prompt forbidding markdown. Keep the inner content so
    evaluators see raw LilyPond (brace balance, LilyBERT tokenization, MIDI
    conversion). Unterminated fences — from a hit token cap — drop just the
    opening line.
    """
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    stripped = _OPEN_FENCE_RE.sub("", text, count=1)
    if stripped != text:
        return stripped.strip()
    return text


def _wrap_score(text: str, *, version: str, language: str) -> str:
    """Prepend minimal \\version / \\language headers when missing.

    Leaves the body untouched: if the model generated a partial fragment, a
    broken trailing statement, or a complete \\score block, the result is
    preserved verbatim. Wrapping partial output in \\score { ... } can both
    mask broken LilyPond and silently change what the output means, so we
    stop doing that here.
    """
    body = text.rstrip()
    header_lines: list[str] = []
    if "\\version" not in body:
        header_lines.append(f'\\version "{version}"')
    if "\\language" not in body:
        header_lines.append(f'\\language "{language}"')
    header = ("\n".join(header_lines) + "\n") if header_lines else ""
    return header + body + "\n"


@hydra.main(config_path="../configs", config_name="infer", version_base=None)
def main(cfg: DictConfig) -> int:
    apply_hf_env(cfg.get("hf"))

    regime = str(cfg.regime.name)
    if regime not in {"zero", "few", "lora"}:
        print(f"[infer] unknown regime: {regime}", file=sys.stderr)
        return 2

    lora_path = None
    fewshot_file = None
    if regime == "lora":
        lora_path = cfg.regime.get("path")
        if not lora_path:
            print("[infer] regime=lora requires regime.path", file=sys.stderr)
            return 2
        lora_path = _resolve_path(lora_path)
    elif regime == "few":
        fewshot_file = cfg.regime.get("fewshot_file")
        if not fewshot_file:
            print("[infer] regime=few requires regime.fewshot_file", file=sys.stderr)
            return 2
        fewshot_file = _resolve_path(fewshot_file)

    output_dir = _resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    print(BANNER_LINE)
    print("LilyBench inference")
    print(BANNER_LINE)
    print(OmegaConf.to_yaml(cfg))
    print(BANNER_LINE)

    spec = get_spec(cfg.model.id)
    print(f"[infer] model registry: id={spec.model_id} hf_id={spec.hf_id} family={spec.family} regime={regime}")

    user_prompt_text = _load_prompt(cfg)
    fewshot_text = ""
    if regime == "few":
        fewshot_text = fewshot_file.read_text(encoding="utf-8").strip()

    prompts_file = cfg.get("prompts_file")
    prompt_bank: list[dict[str, Any]] = []
    if prompts_file:
        prompt_bank = _load_prompt_bank(_resolve_path(prompts_file))
        if not prompt_bank:
            print(f"[infer] prompts_file is empty: {prompts_file}", file=sys.stderr)
            return 2
        print(f"[infer] loaded prompt bank with {len(prompt_bank)} records from {prompts_file}")

    if regime == "lora":
        tokenizer = AutoTokenizer.from_pretrained(
            str(lora_path), use_fast=True, local_files_only=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id, use_fast=True, trust_remote_code=spec.trust_remote_code
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[infer] loading base model: {spec.hf_id}")
    dtype = _DTYPE_MAP[spec.dtype]
    base = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
    )

    if regime == "lora":
        from peft import PeftModel

        print(f"[infer] loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(base, str(lora_path), local_files_only=True)
    else:
        model = base
    model.eval()

    version = str(cfg.lilypond_version)
    language = str(cfg.lilypond_language)

    device = next(model.parameters()).device
    seed_base = int(cfg.seed_base)
    max_new_tokens = int(cfg.max_new_tokens)
    temperature = float(cfg.temperature)
    top_p = float(cfg.top_p)

    if prompt_bank:
        num_samples = len(prompt_bank)
    else:
        num_samples = int(cfg.num_samples)
        preview_prompt = _build_prompt(
            regime=regime,
            user_prompt_text=user_prompt_text,
            fewshot_text=fewshot_text,
            tokenizer=tokenizer,
            lilypond_version=version,
            lilypond_language=language,
            model_id=spec.model_id,
        )
        print(f"\n[infer] prompt preview (first 200 chars): {preview_prompt[:200]!r}")

    for i in range(num_samples):
        seed = seed_base + i
        print(f"\n[infer] sample {i + 1}/{num_samples} (seed={seed})")

        if prompt_bank:
            record = prompt_bank[i]
            record_metadata = record.get("metadata")
            record_user = record.get("user_prompt", user_prompt_text)
            prompt = _build_prompt(
                regime=regime,
                user_prompt_text=record_user,
                fewshot_text=fewshot_text,
                tokenizer=tokenizer,
                lilypond_version=version,
                lilypond_language=language,
                model_id=spec.model_id,
                metadata=record_metadata,
            )
        else:
            prompt = preview_prompt

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = out[0][prompt_len:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        marker = spec.generation_end_marker
        if marker and marker in raw_text:
            raw_text = raw_text.split(marker, 1)[0]
        raw_text = raw_text.rstrip()
        cleaned_text = _strip_markdown_fences(raw_text)
        if regime == "lora":
            cleaned_text = f"{prompt}{cleaned_text}"

        final_text = _wrap_score(cleaned_text.strip(), version=version, language=language)

        raw_path = samples_dir / f"raw_{i:04d}.txt"
        ly_path = samples_dir / f"sample_{i:04d}.ly"
        raw_path.write_text(raw_text + "\n", encoding="utf-8")
        ly_path.write_text(final_text, encoding="utf-8")

        print(BANNER_LINE)
        print(f"Raw Output {i + 1}:")
        print(BANNER_LINE)
        print(raw_text)
        print(BANNER_LINE)
        print(f"Detokenized Output {i + 1}:")
        print(BANNER_LINE)
        print(final_text)
        print(BANNER_LINE)

    print(f"\n[infer] wrote {num_samples} samples to {samples_dir}")
    return 0


if __name__ == "__main__":
    main()
