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

import random
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from lilybench.models import get_spec

BANNER_LINE = "=" * 80
_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _load_prompt(cfg: DictConfig) -> str:
    prompt_file = cfg.get("prompt_file")
    if prompt_file:
        return Path(prompt_file).expanduser().read_text(encoding="utf-8").strip()
    return str(cfg.default_prompt).strip()


def _build_prompt(
    *,
    regime: str,
    user_prompt_text: str,
    fewshot_text: str,
    tokenizer,
) -> str:
    if regime == "lora":
        return "\\relative do'' {\n"

    system = (
        "You are a LilyPond assistant. Output only valid LilyPond code, "
        "no prose, no markdown."
    )
    user = user_prompt_text
    if regime == "few":
        user = f"{fewshot_text}\n\n{user_prompt_text}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{system}\n\n{user}\n"


def _wrap_score(text: str, *, version: str, language: str) -> str:
    body = text.rstrip()
    if "\\score" in body:
        return body
    header_lines = []
    if "\\version" not in body:
        header_lines.append(f'\\version "{version}"')
    if "\\language" not in body:
        header_lines.append(f'\\language "{language}"')
    header = ("\n".join(header_lines) + "\n") if header_lines else ""
    return header + "\\score {\n" + body + "\n\\layout {}\n\\midi {}\n}\n"


@hydra.main(config_path="../configs", config_name="infer", version_base=None)
def main(cfg: DictConfig) -> int:
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

    prompt = _build_prompt(
        regime=regime,
        user_prompt_text=user_prompt_text,
        fewshot_text=fewshot_text,
        tokenizer=tokenizer,
    )
    print(f"\n[infer] prompt preview (first 200 chars): {prompt[:200]!r}")

    device = next(model.parameters()).device
    seed_base = int(cfg.seed_base)
    num_samples = int(cfg.num_samples)
    max_new_tokens = int(cfg.max_new_tokens)
    temperature = float(cfg.temperature)
    top_p = float(cfg.top_p)
    version = str(cfg.lilypond_version)
    language = str(cfg.lilypond_language)

    for i in range(num_samples):
        seed = seed_base + i
        print(f"\n[infer] sample {i + 1}/{num_samples} (seed={seed})")

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw_text = tokenizer.decode(out[0], skip_special_tokens=False)
        marker = spec.generation_end_marker
        if marker and marker in raw_text:
            raw_text = raw_text.split(marker, 1)[0]
        raw_text = raw_text.rstrip()

        final_text = _wrap_score(raw_text.strip(), version=version, language=language)

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
