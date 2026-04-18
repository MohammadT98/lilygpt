#!/bin/bash
#SBATCH --job-name=lilybench-exp12-infer
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/exp12-infer-%j.out
#SBATCH --error=/tmp/exp12-infer-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

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

# Copy logs back at end
copy_logs() {
    cp "/tmp/exp12-infer-$SLURM_JOB_ID.out" "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true
    cp "/tmp/exp12-infer-$SLURM_JOB_ID.err" "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true
    rm -f "/tmp/exp12-infer-$SLURM_JOB_ID.out" "/tmp/exp12-infer-$SLURM_JOB_ID.err" || true
}
trap copy_logs EXIT

source "$HOME/lilybench/.venv/bin/activate"

python - <<'PY'
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "openai/gpt-oss-20b"
LORA_PATH = "runs/lora_exp12_voice_tokens_eff4/final"
LILYPOND_VERSION = "2.24.4"
LILYPOND_LANGUAGE = "italiano"
NUM_SAMPLES = 100

print("=" * 60)
print("Testing exp12 (voice labels + structural tokens, eff4)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, use_fast=True, local_files_only=True)

print("\nLoading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

print("Loading LoRA weights from exp12...")
model = PeftModel.from_pretrained(base, LORA_PATH, local_files_only=True)
model.eval()

user_prompt = """\\relative do'' {
"""
prompt = user_prompt if user_prompt.lstrip().startswith("<VOICE>") else f"<VOICE> {user_prompt}"

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

    clean_text = raw_text.strip()
    final_text = _wrap_score(clean_text)

    print("\n" + "=" * 60)
    print(f"Detokenized Output {i + 1}:")
    print("=" * 60)
    print(final_text)
    print("=" * 60)
PY
