#!/bin/bash
#SBATCH --job-name=20b-lora-infer
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/20b-lora-infer-%j.out
#SBATCH --error=logs/20b-lora-infer-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

source "$HOME/lilygpt/.venv/bin/activate"

python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "openai/gpt-oss-20b"
LORA_PATH  = "runs/lora_exp3_len2048/checkpoint-2250"  # Testing exp3 checkpoint-2250
OUT_PATH   = "runs/lora_exp3_len2048/infer_ckpt2250.ly"

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
        max_new_tokens=500,       # don't let it run forever
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
