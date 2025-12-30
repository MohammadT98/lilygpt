#!/bin/bash
#SBATCH --job-name=base-vs-finetuned
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diagnostic-base-vs-finetuned-%j.out
#SBATCH --error=logs/diagnostic-base-vs-finetuned-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/diagnostics

source "$HOME/lilygpt/.venv/bin/activate"

python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "openai/gpt-oss-20b"

# Test configurations
LORA_CHECKPOINTS = [
    ("Base Model (no LoRA)", None),
    ("exp3 checkpoint-1750", "runs/lora_exp3_len2048/checkpoint-1750"),
    ("exp3 checkpoint-2250", "runs/lora_exp3_len2048/checkpoint-2250"),
]

prompt = """violinoI = \\relative do'' {
do4 re mi
"""

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

# Generation parameters (consistent across all tests)
gen_params = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.92,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 6,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.eos_token_id,
}

print("="*80)
print("DIAGNOSTIC TEST: Base Model vs Fine-Tuned Models")
print("="*80)
print(f"\nPrompt:\n{prompt}")
print(f"\nGeneration params: {gen_params}")
print("\n" + "="*80)

for name, lora_path in LORA_CHECKPOINTS:
    print(f"\n{'='*80}")
    print(f"MODEL: {name}")
    print(f"{'='*80}\n")

    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Optionally load LoRA adapter
    if lora_path:
        print(f"Loading LoRA from: {lora_path}")
        model = PeftModel.from_pretrained(base, lora_path)
    else:
        print("Using base model without LoRA")
        model = base

    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        out = model.generate(**inputs, **gen_params)
        text = tokenizer.decode(out[0], skip_special_tokens=True)

    print(text)

    # Save to file
    safe_name = name.replace(" ", "_").replace("/", "_")
    out_path = f"runs/diagnostics/{safe_name}.ly"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[SAVED] {out_path}")

    # Clean up to free memory
    del model
    del base
    torch.cuda.empty_cache()

print("\n" + "="*80)
print("DIAGNOSTIC TEST COMPLETE")
print("="*80)
print("\nOutput files saved to: runs/diagnostics/")
print("  - Base_Model_(no_LoRA).ly")
print("  - exp3_checkpoint-1750.ly")
print("  - exp3_checkpoint-2250.ly")
print("\nCompare these files to see the effect of fine-tuning.")
PY
