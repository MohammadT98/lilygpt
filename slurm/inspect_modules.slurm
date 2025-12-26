#!/bin/bash
#SBATCH --job-name=inspect-mods
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/inspect-%j.out
#SBATCH --error=logs/inspect-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/lilygpt/.venv/bin/activate"

python - <<'PY'
from transformers import AutoModelForCausalLM
model_id="openai/gpt-oss-20b"
m = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", low_cpu_mem_usage=True)

names=set()
for n,_ in m.named_modules():
    # collect last component names
    last = n.split(".")[-1]
    if any(x in last for x in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","c_attn","c_proj"]):
        names.add(last)

print("FOUND CANDIDATE MODULE NAMES:", sorted(names))

# also print a few full module paths containing proj to confirm
hits=[]
for n,_ in m.named_modules():
    if any(x in n for x in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]):
        hits.append(n)
print("\nSAMPLE FULL PATHS:")
for x in hits[:40]:
    print(x)
PY
