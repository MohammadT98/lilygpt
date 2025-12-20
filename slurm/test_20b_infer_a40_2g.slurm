cat > slurm/test_20b_infer_a40_2g.slurm <<'EOF'
#!/bin/bash
#SBATCH --job-name=20b-infer
#SBATCH --partition=allgroups
#SBATCH --qos=allgroups_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/20b-infer-%j.out
#SBATCH --error=logs/20b-infer-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/lilygpt/.venv/bin/activate"

echo "== hostname =="
hostname
echo "== nvidia-smi =="
nvidia-smi

python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "openai/gpt-oss-20b"

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",              # <-- shards model across the 2 GPUs
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

prompt = "Write a short LilyPond snippet for a simple melody in C major:\n"
inputs = tok(prompt, return_tensors="pt")

# Send inputs to the first device the model lives on
first_device = next(model.parameters()).device
inputs = {k: v.to(first_device) for k, v in inputs.items()}

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128)
print(tok.decode(out[0], skip_special_tokens=True))
PY
EOF
