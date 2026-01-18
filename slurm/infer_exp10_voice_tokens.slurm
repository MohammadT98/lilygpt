#!/bin/bash
#SBATCH --job-name=lilygpt-exp10-infer
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp10-infer-%j.out
#SBATCH --error=logs/exp10-infer-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$HOME/lilygpt/.venv/bin/activate"

python - <<'PY'
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "openai/gpt-oss-20b"
LORA_PATH = "runs/lora_exp10_voice_tokens/final"

print("=" * 60)
print("Testing exp10 (voice labels + structural tokens)")
print("=" * 60)

# Load tokenizer from LoRA path (has special tokens)
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, use_fast=True)

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

# User-facing prompt (voice label injected internally)
user_prompt = """\\relative do'' {
"""

if user_prompt.lstrip().startswith("<VOICE>"):
    prompt = user_prompt
else:
    prompt = f"<VOICE> {user_prompt}"

print(f"\nPrompt: {repr(prompt)}")
print("\nGenerating...")

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

# Generate
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

def _detokenize(text: str) -> str:
    def _repl_key(match: re.Match) -> str:
        payload = match.group(1).strip()
        if "_" in payload:
            key, mode = payload.split("_", 1)
            return f"\\\\key {key} \\\\{mode}"
        return f"\\\\key {payload}"

    text = re.sub(r"<KEY:([^>]+)>", _repl_key, text)
    text = re.sub(r"<TIME:([^>]+)>", r"\\\\time \\1", text)
    text = re.sub(r"<TEMPO:([^>]+)>", r"\\\\tempo \\1", text)
    text = text.replace("<BAR>", "|")
    text = text.replace("<VOICE>", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()

# Decode
text = tokenizer.decode(out[0], skip_special_tokens=False)
text = text.split("<|return|>", 1)[0].rstrip()
text = _detokenize(text)

print("\n" + "=" * 60)
print("Generated Output:")
print("=" * 60)
print(text)
print("=" * 60)
PY
