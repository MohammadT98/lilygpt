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

# Prompt with voice token
prompt = """<VOICE> \\relative do'' {
"""

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

# Decode
text = tokenizer.decode(out[0], skip_special_tokens=False)
text = text.split("<|return|>", 1)[0].rstrip()

print("\n" + "=" * 60)
print("Generated Output:")
print("=" * 60)
print(text)
print("=" * 60)
PY
