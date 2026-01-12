#!/bin/bash
#SBATCH --job-name=exp9-infer
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/exp9-infer-%j.out
#SBATCH --error=logs/exp9-infer-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output

source "$HOME/lilygpt/.venv/bin/activate"

python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

BASE_MODEL = "openai/gpt-oss-20b"
LORA_PATH = "runs/lora_exp9_unweighted_eos_no_mask/final"
OUT_PATH = "output/exp9_inference_test.ly"

print("="*60)
print("Testing exp9 (unweighted + EOS + NO input masking)")
print("="*60)

class BraceStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_text):
        self.tokenizer = tokenizer
        self.initial_depth = prompt_text.count("{") - prompt_text.count("}")

    def __call__(self, input_ids, scores, **kwargs):
        if self.initial_depth <= 0:
            return False
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        depth = text.count("{") - text.count("}")
        return depth < self.initial_depth

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
print("Loading LoRA weights from exp9...")
model = PeftModel.from_pretrained(base, LORA_PATH)
model.eval()

# Use training-style prompt
prompt = """IvlIn = \\relative do'' {
"""

print(f"\nPrompt: {repr(prompt)}")
print("\nGenerating...")

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

stopping = StoppingCriteriaList([BraceStoppingCriteria(tokenizer, prompt)])

# Generate
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping,
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

# Check musical content
has_notes = any(note in text for note in ['do', 're', 'mi', 'fa', 'sol', 'la', 'si'])
has_rhythms = any(rhythm in text for rhythm in ['4', '8', '16', '2'])

print(f"\nMusical content check:")
print(f"  - Contains notes: {'OK' if has_notes else 'NO'}")
print(f"  - Contains rhythms: {'OK' if has_rhythms else 'NO'}")

print(f"\n[SAVED] {OUT_PATH}")
print("="*60)

PY
