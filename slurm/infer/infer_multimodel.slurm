#!/bin/bash
#SBATCH --job-name=lilybench-infer
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/infer-%x-%j.out
#SBATCH --error=logs/infer-%x-%j.err

# Parametrized multi-model, multi-regime inference.
# Required env vars:
#   MODEL_ID       registry id (gpt-oss | phi4 | qwen-coder | deepseek-coder | codestral)
#   REGIME         zero | few | lora
# Required for REGIME=lora:
#   LORA_PATH      path to LoRA adapter folder (e.g. runs/.../final)
# Required for REGIME=few:
#   FEWSHOT_FILE   text file with three curated demonstrations separated by a blank line
# Optional:
#   PROMPT_FILE    path to a prompt text file (user instruction). If absent, a default
#                  monophonic prompt is used.
#   NUM_SAMPLES    default 100
#   TEMPERATURE    default 0.7
#   TOP_P          default 0.9
#   MAX_NEW_TOKENS default 3000
#
# Example:
#   MODEL_ID=qwen-coder REGIME=zero NUM_SAMPLES=50 \
#     sbatch --job-name=qwen-zero slurm/infer/infer_multimodel.slurm

set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required}"
: "${REGIME:?REGIME is required (zero|few|lora)}"

export MODEL_ID REGIME
export LORA_PATH="${LORA_PATH:-}"
export FEWSHOT_FILE="${FEWSHOT_FILE:-}"
export PROMPT_FILE="${PROMPT_FILE:-}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.9}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3000}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$HOME/lilybench/.venv/bin/activate"

SCRATCH_ROOT="${TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_ROOT/$USER-$SLURM_JOB_ID"
mkdir -p "$SCRATCH"

export TMPDIR="$SCRATCH/tmp"
export TORCH_HOME="$SCRATCH/torch"
export TRITON_CACHE_DIR="$SCRATCH/triton"

export HF_HOME="$SCRATCH/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

mkdir -p "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

python - <<'PY'
import os
import random
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lilybench.models import get_spec

MODEL_ID = os.environ["MODEL_ID"]
REGIME = os.environ["REGIME"]
LORA_PATH = os.environ.get("LORA_PATH") or None
FEWSHOT_FILE = os.environ.get("FEWSHOT_FILE") or None
PROMPT_FILE = os.environ.get("PROMPT_FILE") or None
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "100"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "3000"))

LILYPOND_VERSION = "2.24.4"
LILYPOND_LANGUAGE = "italiano"

spec = get_spec(MODEL_ID)
print("=" * 60)
print(f"Multi-model inference  model={spec.model_id} hf={spec.hf_id} regime={REGIME}")
print("=" * 60)

if REGIME not in {"zero", "few", "lora"}:
    print(f"[infer] unknown regime: {REGIME}", file=sys.stderr)
    sys.exit(2)
if REGIME == "lora" and not LORA_PATH:
    print("[infer] REGIME=lora requires LORA_PATH", file=sys.stderr)
    sys.exit(2)
if REGIME == "few" and not FEWSHOT_FILE:
    print("[infer] REGIME=few requires FEWSHOT_FILE", file=sys.stderr)
    sys.exit(2)

DEFAULT_PROMPT = (
    "Write a short LilyPond fragment, four bars long, in A minor, 4/4. "
    "Use Italian note names (do re mi fa sol la si) and lowercase relative "
    "notation. Output only the LilyPond code; do not add prose."
)
if PROMPT_FILE:
    with open(PROMPT_FILE, "r", encoding="utf-8") as fh:
        user_prompt_text = fh.read().strip()
else:
    user_prompt_text = DEFAULT_PROMPT

fewshot_text = ""
if REGIME == "few":
    with open(FEWSHOT_FILE, "r", encoding="utf-8") as fh:
        fewshot_text = fh.read().strip()

# --- tokenizer / model load ---
if REGIME == "lora":
    tokenizer_src = LORA_PATH
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_src, use_fast=True, local_files_only=True
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, use_fast=True, trust_remote_code=spec.trust_remote_code
    )
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[infer] loading base model: {spec.hf_id}")
dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[spec.dtype]
base = AutoModelForCausalLM.from_pretrained(
    spec.hf_id,
    device_map="auto",
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=spec.trust_remote_code,
)

if REGIME == "lora":
    from peft import PeftModel
    print(f"[infer] loading LoRA adapter: {LORA_PATH}")
    model = PeftModel.from_pretrained(base, LORA_PATH, local_files_only=True)
else:
    model = base
model.eval()


# --- prompt assembly ---
def _build_prompt() -> str:
    """Build the model-ready prompt string for the current regime.

    For zero/few-shot, use the tokenizer's chat template when available.
    For lora, reuse the continuation-style seed used by the original pipeline.
    """
    if REGIME == "lora":
        return "\\relative do'' {\n"

    system = (
        "You are a LilyPond assistant. Output only valid LilyPond code, "
        "no prose, no markdown."
    )
    user = user_prompt_text
    if REGIME == "few":
        user = f"{fewshot_text}\n\n{user_prompt_text}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # Rely on the model's built-in chat template when present.
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback: plain concatenation.
    return f"{system}\n\n{user}\n"


prompt = _build_prompt()
print(f"\n[infer] prompt preview (first 200 chars): {prompt[:200]!r}")

device = next(model.parameters()).device


def _generate_once(seed: int) -> str:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw_text = tokenizer.decode(out[0], skip_special_tokens=False)
    # Strip any model-specific end marker (e.g. <|return|> for gpt-oss).
    marker = spec.generation_end_marker
    if marker and marker in raw_text:
        raw_text = raw_text.split(marker, 1)[0]
    return raw_text.rstrip()


def _wrap_score(text: str) -> str:
    body = text.rstrip()
    if "\\score" in body:
        return body
    header_lines = []
    if "\\version" not in body:
        header_lines.append(f'\\version "{LILYPOND_VERSION}"')
    if "\\language" not in body:
        header_lines.append(f'\\language "{LILYPOND_LANGUAGE}"')
    header = ("\n".join(header_lines) + "\n") if header_lines else ""
    return header + "\\score {\n" + body + "\n\\layout {}\n\\midi {}\n}\n"


for i in range(NUM_SAMPLES):
    print(f"\n[infer] sample {i + 1}/{NUM_SAMPLES}")
    raw_text = _generate_once(seed=1234 + i)
    print("=" * 60)
    print(f"Raw Output {i + 1}:")
    print("=" * 60)
    print(raw_text)
    print("=" * 60)

    final_text = _wrap_score(raw_text.strip())
    print("=" * 60)
    print(f"Detokenized Output {i + 1}:")
    print("=" * 60)
    print(final_text)
    print("=" * 60)
PY
