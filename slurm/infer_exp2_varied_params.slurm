#!/bin/bash
#SBATCH --job-name=test-gen-params
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test-gen-params-%j.out
#SBATCH --error=logs/test-gen-params-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/lora_exp2_lr5e5

source "$HOME/lilygpt/.venv/bin/activate"

python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "openai/gpt-oss-20b"
LORA_PATH  = "runs/lora_exp2_lr5e5/final"

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

# Test different generation strategies
configs = [
    {
        "name": "greedy",
        "params": {
            "max_new_tokens": 200,
            "do_sample": False,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
    },
    {
        "name": "low_temp",
        "params": {
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
    },
    {
        "name": "high_penalty",
        "params": {
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "repetition_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
    },
    {
        "name": "nucleus_sampling",
        "params": {
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.92,
            "repetition_penalty": 1.3,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
    },
]

for config in configs:
    print(f"\n{'='*60}")
    print(f"CONFIG: {config['name']}")
    print(f"{'='*60}\n")

    with torch.no_grad():
        out = model.generate(**inputs, **config["params"])
        text = tokenizer.decode(out[0], skip_special_tokens=True)

    print(text)

    # Save to file
    out_path = f"runs/lora_exp2_lr5e5/infer_{config['name']}.ly"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[SAVED] {out_path}")

print("\n" + "="*60)
print("ALL TESTS COMPLETE")
print("="*60)
PY
