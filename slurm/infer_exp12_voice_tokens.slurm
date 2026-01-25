#!/bin/bash
#SBATCH --job-name=lilygpt-exp12-infer
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/tmp/exp12-infer-%j.out
#SBATCH --error=/tmp/exp12-infer-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Copy logs back at end
copy_logs() {
    cp "/tmp/exp12-infer-$SLURM_JOB_ID.out" "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true
    cp "/tmp/exp12-infer-$SLURM_JOB_ID.err" "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true
    rm -f "/tmp/exp12-infer-$SLURM_JOB_ID.out" "/tmp/exp12-infer-$SLURM_JOB_ID.err" || true
}
trap copy_logs EXIT

source "$HOME/lilygpt/.venv/bin/activate"

python - <<'PY'
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "openai/gpt-oss-20b"
LORA_PATH = "runs/lora_exp12_voice_tokens_eff4/final"
LILYPOND_VERSION = "2.24.4"
LILYPOND_LANGUAGE = "italiano"

print("=" * 60)
print("Testing exp12 (voice labels + structural tokens, eff4)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<KEY:", "<TIME:", "<TEMPO:", "<VOICE>"]}
)

print("\nLoading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
base.resize_token_embeddings(len(tokenizer))

print("Loading LoRA weights from exp12...")
model = PeftModel.from_pretrained(base, LORA_PATH, local_files_only=True)
model.eval()

user_prompt = """\\relative do'' {
"""
prompt = user_prompt if user_prompt.lstrip().startswith("<VOICE>") else f"<VOICE> {user_prompt}"

print(f"\nPrompt: {repr(prompt)}")
print("\nGenerating...")

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

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
raw_text = raw_text.split("<|return|>", 1)[0].rstrip()

print("\n" + "=" * 60)
print("Raw Output:")
print("=" * 60)
print(raw_text)
print("=" * 60)

def _clean_payload(payload: str) -> str:
    return payload.strip().lstrip("\\")

def _detokenize(text: str) -> str:
    def _repl_key(match: re.Match) -> str:
        payload = _clean_payload(match.group(1))
        if "_" in payload:
            key, mode = payload.split("_", 1)
            return f"\\key {key} \\{mode}\n"
        return f"\\key {payload}\n"

    def _repl_time(match: re.Match) -> str:
        return f"\\time {_clean_payload(match.group(1))}\n"

    def _repl_tempo(match: re.Match) -> str:
        return f"\\tempo {_clean_payload(match.group(1))}\n"

    text = re.sub(r"<KEY:([^>]+)>[ \t]*\r?\n?", _repl_key, text)
    text = re.sub(r"<TIME:([^>]+)>[ \t]*\r?\n?", _repl_time, text)
    text = re.sub(r"<TEMPO:([^>]+)>[ \t]*\r?\n?", _repl_tempo, text)
    text = text.replace("<VOICE>", "")
    text = re.sub(r"(\\key[^\n]*\n)\n+", r"\1", text)
    text = re.sub(r"(\\time[^\n]*\n)\n+", r"\1", text)
    text = re.sub(r"(\\tempo[^\n]*\n)\n+", r"\1", text)
    return text.strip()

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

clean_text = _detokenize(raw_text)
final_text = _wrap_score(clean_text)

print("\n" + "=" * 60)
print("Detokenized Output:")
print("=" * 60)
print(final_text)
print("=" * 60)
PY
