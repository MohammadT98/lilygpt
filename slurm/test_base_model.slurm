#!/bin/bash
#SBATCH --job-name=test-base-model
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/test-base-%j.out
#SBATCH --error=logs/test-base-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

source "$HOME/lilygpt/.venv/bin/activate"

python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "openai/gpt-oss-20b"
OUT_PATH   = "runs/base_model_test.ly"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
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
