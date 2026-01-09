#!/bin/bash
#SBATCH --job-name=exp6b-infer
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/exp6b-infer-%j.out
#SBATCH --error=logs/exp6b-infer-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output

source "$HOME/lilygpt/.venv/bin/activate"

python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "openai/gpt-oss-20b"
LORA_PATH = "output/exp6b_weight5"
OUT_PATH = "output/exp6b_inference_test.ly"

print("="*60)
print("Testing exp6b (5x weighted loss)")
print("="*60)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

# Load base model
print("\nLoading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# Load LoRA weights
print("Loading LoRA weights from exp6b...")
model = PeftModel.from_pretrained(base, LORA_PATH)
model.eval()

# Use training-style prompt (IvlIn format works best)
prompt = """IvlIn = \\relative do'' {
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
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=6,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode
text = tokenizer.decode(out[0], skip_special_tokens=True)

# Save
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(text + "\n")

print("\n" + "="*60)
print("Generated Output:")
print("="*60)
print(text)

# Count closing braces
closing_braces = text.count("}")

print("\n" + "="*60)
print("Analysis:")
print("="*60)
print(f"Closing braces found: {closing_braces}")

if closing_braces == 0:
    print("❌ FAIL: No closing brace (same as exp5)")
    print("   5x weight too weak, model didn't learn to close")
elif closing_braces == 1:
    print("✅ SUCCESS: Exactly one closing brace!")
    print("   5x is the perfect sweet spot!")
elif closing_braces <= 4:
    print(f"⚠️  PARTIAL: {closing_braces} closing braces")
    print(f"   Better than exp6's 12, but not perfect")
    print("   Consider trying exp6c (10x weight)")
else:
    print(f"⚠️  OVER-GENERATION: {closing_braces} closing braces")
    print("   5x still too strong, but better than 20x")

# Check musical content
has_notes = any(note in text for note in ['do', 're', 'mi', 'fa', 'sol', 'la', 'si'])
has_rhythms = any(rhythm in text for rhythm in ['4', '8', '16', '2'])

print(f"\nMusical content check:")
print(f"  - Contains notes: {'✅' if has_notes else '❌'}")
print(f"  - Contains rhythms: {'✅' if has_rhythms else '❌'}")

print(f"\n[SAVED] {OUT_PATH}")
print("="*60)

PY
