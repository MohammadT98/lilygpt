# lilynorm

**Fine-tune GPT models on LilyPond music notation**

Complete pipeline for processing LilyPond scores, training GPT models with LoRA adapters, and generating new music. Developed for master's thesis research.

## 🎵 Features

- **LilyPond Processing**: Normalize, clean, and tokenize LilyPond scores
- **Data Pipeline**: Train/validation/test splitting with JSONL format
- **LoRA Training**: Efficient fine-tuning of large GPT models
- **Music Generation**: Generate LilyPond continuations from prompts
- **Evaluation**: Syntax validation and music statistics

## 📁 Project Structure

```
lilynorm/
├── src/lilynorm/              # Main package
│   ├── stages/
│   │   ├── preprocessing/     # Normalize & clean LilyPond
│   │   ├── tokenization/      # GPT tokenization
│   │   ├── splitting/         # Train/val/test splits
│   │   └── training/          # LoRA training utilities
│   ├── inference/             # Music generation
│   └── evaluation/            # Metrics & validation
├── examples/                  # Usage examples
│   ├── train_model.py         # Training example
│   └── generate_music.py      # Generation example
├── scripts/                   # Pipeline scripts
│   └── process_dataset.py     # Main data processor
├── data/                      # Data storage
│   ├── raw/                   # Original .ly files
│   ├── normalized_dataset/    # Processed files
│   ├── splits/                # train/val/test.jsonl
│   └── fine_tuning/           # Model checkpoints
└── configs/                   # Configuration files
```

## 🚀 Quick Start

### Prerequisites

* Python 3.10+
* [uv](https://github.com/astral-sh/uv) (recommended) or pip
* LilyPond (for evaluation/compilation)

### Installation

```bash
# Clone the repository
git clone https://github.com/MohammadT98/lilygpt.git
cd lilynorm

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## 📚 Complete Pipeline

### 1️⃣ Data Processing

Process raw LilyPond files into normalized, tokenized datasets:

```bash
# Process all .ly files in data/raw/
python -m scripts.process_dataset --input "data/raw"

# This will:
# - Normalize LilyPond syntax
# - Extract single voices
# - Tokenize for GPT
# - Create train/val/test splits
```

**Output**: `data/splits/train.jsonl`, `val.jsonl`, `test.jsonl`

### 2️⃣ Training

Fine-tune a GPT model with LoRA adapters:

```bash
# Use the example script
python examples/train_model.py

# Or use the training module directly
python -m lilynorm.stages.training.train \
    --train data/splits/train.jsonl \
    --val data/splits/val.jsonl \
    --model-name openai/gpt-oss-20b \
    --output-dir data/fine_tuning/checkpoints \
    --epochs 3
```

**Output**: LoRA checkpoint in `data/fine_tuning/checkpoints/`

### 3️⃣ Generation

Generate new LilyPond music from your trained model:

```bash
# Use the example script
python examples/generate_music.py

# Or use the CLI directly
python -m lilynorm.inference.generator \
    --model openai/gpt-oss-20b \
    --adapter data/fine_tuning/checkpoints/checkpoint-1000 \
    --simple-prompt \
    --output generated.ly \
    --num-variations 3
```

**Output**: Generated `.ly` files in `data/generated_music/`

### 4️⃣ Evaluation

Evaluate generated music with automatic metrics:

```bash
# Evaluate all generated files
python -m lilynorm.evaluation.metrics data/generated_music/*.ly

# This will check:
# - Syntax validity (can LilyPond compile it?)
# - Note statistics (count, range, intervals)
# - Repetition metrics
```

## 💡 Examples

### Training Example

```python
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from lilynorm.stages.tokenization import LilyTokensDataset, collate_batch

# Load datasets
train_dataset = LilyTokensDataset("data/splits/train.jsonl", tokenizer, max_length=1024)
val_dataset = LilyTokensDataset("data/splits/val.jsonl", tokenizer, max_length=1024)

# Apply LoRA
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32)
model = get_peft_model(model, lora_config)

# Train
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
```

See `examples/train_model.py` for complete example.

### Generation Example

```python
from lilynorm.inference import MusicGenerator, GenerationConfig, create_simple_prompt

# Load your fine-tuned model
generator = MusicGenerator(
    model_name_or_path="openai/gpt-oss-20b",
    adapter_path="data/fine_tuning/checkpoints/checkpoint-1000"
)

# Create a prompt
prompt = create_simple_prompt(key="do", mode="major", time_signature="4/4")

# Generate music
generated = generator.generate_and_save(
    prompt=prompt,
    output_path="my_music.ly",
    config=GenerationConfig(max_length=512, temperature=1.0)
)
```

See `examples/generate_music.py` for complete example.

### Evaluation Example

```python
from lilynorm.evaluation import LilyPondMetrics, evaluate_generated_files, print_evaluation_summary

# Evaluate multiple files
summary = evaluate_generated_files(["file1.ly", "file2.ly", "file3.ly"])
print_evaluation_summary(summary)

# Output:
# ============================================================
# EVALUATION SUMMARY
# ============================================================
# Total files evaluated: 3
# Valid syntax (compilable): 2 (66.7%)
# 
# Average statistics:
#   - Total notes: 45.3
#   - Unique pitches: 5.7
#   - Repetition ratio: 0.23
#   - Average interval: 1.8 steps
# ============================================================
```

## 🔧 Configuration

Configuration files in `configs/`:

- `defaults.yaml` - Default processing settings
- `profiles/` - Processing profiles (strict, keep_engraving, etc.)
- `prompts/` - Prompt templates (zero-shot, few-shot)

## 📊 For Your Thesis

### Key Metrics to Report

1. **Dataset Statistics**:
   - Training samples: `len(train_dataset)`
   - Validation samples: `len(val_dataset)`
   - Average sequence length

2. **Model Statistics**:
   - Total parameters vs trainable (LoRA efficiency)
   - Training loss curves
   - Best validation loss

3. **Generation Quality**:
   - Syntax validity rate (% compilable)
   - Average note count
   - Pitch range distribution
   - Repetition metrics

4. **Qualitative Evaluation**:
   - Musical plausibility (1-5 rating)
   - Listen to MIDI output
   - Visual inspection of scores

### Quick Commands for Thesis Work

```bash
# Complete pipeline in one go
python -m scripts.process_dataset --input data/raw
python examples/train_model.py
python examples/generate_music.py

# Generate thesis figures
python -m lilynorm.evaluation.metrics data/generated_music/*.ly > results.txt
```

## 🛠️ Advanced Usage

### Custom Prompt Generation

```python
# Custom LilyPond prompt
custom_prompt = """\\version "2.24.0"
\\language "italiano"
{
  \\key sol \\major
  \\time 3/4
  \\relative do' {
    sol4 la si |
"""

generated = generator.generate(custom_prompt)
```

### Batch Generation

```python
prompts = [create_simple_prompt(key=k) for k in ["do", "re", "mi", "fa"]]
results = generator.generate_batch(prompts)
```

### Training with Custom Settings

See `src/lilynorm/stages/training/train.py` for all available options:
- LoRA rank, alpha, dropout
- Learning rate, batch size
- FP16/BF16 training
- Gradient accumulation

## 📖 Documentation

- **Data Processing**: `src/lilynorm/stages/preprocessing/`
- **Training**: `src/lilynorm/stages/training/`
- **Generation**: `src/lilynorm/inference/`
- **Evaluation**: `src/lilynorm/evaluation/`

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- HuggingFace Transformers for model infrastructure
- PEFT for LoRA implementation
- LilyPond for music notation

---

**Built for master's thesis research | University of Padova**
