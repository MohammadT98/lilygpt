#!/bin/bash
#SBATCH --job-name=exp4-infer
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/exp4-infer-%j.out
#SBATCH --error=logs/exp4-infer-%j.err

# Usage: Edit CHECKPOINT_NUM below to test different checkpoints
# Example: CHECKPOINT_NUM=1750 tests checkpoint-1750

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/lora_exp4_standard

source "$HOME/lilygpt/.venv/bin/activate"

# EDIT THIS to test different checkpoints
CHECKPOINT_NUM=1750

python - <<'PY'
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "openai/gpt-oss-20b"
CHECKPOINT_NUM = int(os.environ.get('CHECKPOINT_NUM', '1750'))
LORA_PATH = f"runs/lora_exp4_standard/checkpoint-{CHECKPOINT_NUM}"
OUT_PATH = f"runs/lora_exp4_standard/infer_ckpt{CHECKPOINT_NUM}.ly"

print(f"Testing checkpoint-{CHECKPOINT_NUM}")
print(f"LoRA path: {LORA_PATH}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base, LORA_PATH)
model.eval()

prompt = """violinoI = \\relative do'' {
do4 re mi
"""

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=1.0,
        top_p=0.92,
        repetition_penalty=1.15,
        no_repeat_ngram_size=6,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(out[0], skip_special_tokens=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(text + "\n")

print(text)
print(f"\n[SAVED] {OUT_PATH}")
PY
