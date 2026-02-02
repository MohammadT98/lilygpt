#!/bin/bash
#SBATCH --job-name=lilygpt-exp10-2voice
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp10-2voice-infer-%j.out
#SBATCH --error=logs/exp10-2voice-infer-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$HOME/lilygpt/.venv/bin/activate"

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
import re
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "openai/gpt-oss-20b"
LORA_PATH = "runs/lora_exp10_voice_tokens/final"
LILYPOND_VERSION = "2.24.4"
LILYPOND_LANGUAGE = "italiano"
NUM_SAMPLES = 100

print("=" * 60)
print("Testing exp10 two-voice prompt (dual <VOICE> markers)")
print("=" * 60)

# Load tokenizer from LoRA path (has special tokens)
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, use_fast=True, local_files_only=True)

# Load base model
print("\nLoading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# Resize embeddings to match tokenizer with special tokens
base.resize_token_embeddings(len(tokenizer))

# Load LoRA weights
print("Loading LoRA weights from exp10...")
model = PeftModel.from_pretrained(base, LORA_PATH, local_files_only=True)
model.eval()

# Two-voice seed for testing voice-token behavior across two assignments
user_prompt = """<VOICE> V1 = \\relative do'' {
<KEY:do_major><TIME:4/4>
do4 re mi fa sol4 la si do
}
<VOICE> V2 = \\relative do' {
<KEY:do_major><TIME:4/4>
do4 si la sol fa4 mi re do
"""

prompt = user_prompt

print(f"\nPrompt: {repr(prompt)}")

device = next(model.parameters()).device

def _generate_once(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=3000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw_text = tokenizer.decode(out[0], skip_special_tokens=False)
    return raw_text.split("<|return|>", 1)[0].rstrip()

def _clean_payload(payload: str) -> str:
    return payload.strip().lstrip("\\")

def _detokenize(text: str) -> str:
    def _repl_key(match: re.Match) -> str:
        payload = _clean_payload(match.group(1))
        if "_" in payload:
            key, mode = payload.split("_", 1)
            return f"\\key {key} \\{mode}\n"
        return f"\\key {payload}\n"

    def _repl_time(match: re.Match) -> str:
        return f"\\time {_clean_payload(match.group(1))}\n"

    def _repl_tempo(match: re.Match) -> str:
        return f"\\tempo {_clean_payload(match.group(1))}\n"

    text = re.sub(r"<KEY:([^>]+)>[ \t]*\r?\n?", _repl_key, text)
    text = re.sub(r"<TIME:([^>]+)>[ \t]*\r?\n?", _repl_time, text)
    text = re.sub(r"<TEMPO:([^>]+)>[ \t]*\r?\n?", _repl_tempo, text)
    text = text.replace("<VOICE>", "")
    text = re.sub(r"(\\key[^\n]*\n)\n+", r"\1", text)
    text = re.sub(r"(\\time[^\n]*\n)\n+", r"\1", text)
    text = re.sub(r"(\\tempo[^\n]*\n)\n+", r"\1", text)
    return text.strip()

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
    return (
        header
        + "\\score {\n"
        + body
        + "\n\\layout {}\n\\midi {}\n}\n"
    )

for i in range(NUM_SAMPLES):
    print(f"\nGenerating sample {i + 1}/{NUM_SAMPLES}...")
    raw_text = _generate_once(seed=1234 + i)

    print("\n" + "=" * 60)
    print(f"Raw Output {i + 1}:")
    print("=" * 60)
    print(raw_text)
    print("=" * 60)

    clean_text = _detokenize(raw_text)
    final_text = _wrap_score(clean_text)

    print("\n" + "=" * 60)
    print(f"Detokenized Output {i + 1}:")
    print("=" * 60)
    print(final_text)
    print("=" * 60)
PY
