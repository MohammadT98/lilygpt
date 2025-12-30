"""Dataset loader for STANDARD causal LM training (no loss masking).

This implements the standard domain adaptation approach used for code/music:
- Train on FULL sequences (all tokens contribute to loss)
- No artificial input/output split
- Model learns patterns throughout the entire sequence

This is different from dataset_continuation.py which used instruction-tuning
style masking (appropriate for ChatGPT, but not for domain adaptation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class StandardSample:
    """A single training example with full sequence training."""
    id: str
    source_file: str
    var_name: str
    full_text: str
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]


class LilyStandardDataset(Dataset):
    """
    Dataset for STANDARD causal LM training on LilyPond.

    Uses the same continuation examples from dataset_continuation.py,
    but trains on the FULL sequence (no loss masking).

    For causal LM training:
    - input_ids = tokenize(input_text + output_text)
    - labels = input_ids  (SAME - no masking!)

    This is the standard approach for:
    - Code generation (GitHub Copilot, CodeLlama)
    - Domain adaptation (teaching LLMs specialized languages)
    - Text completion (not instruction following)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with continuation examples
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self.samples: List[StandardSample] = []
        self._load_and_tokenize()

    def _load_and_tokenize(self):
        """Load examples from JSONL and tokenize them."""
        with self.path.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj: Dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue

                # Extract fields
                example_id = obj.get('id', f'example_{line_num}')
                source_file = obj.get('source_file', '')
                var_name = obj.get('var_name', '')
                input_text = obj.get('input', '')
                output_text = obj.get('output', '')

                # Concatenate to get full sequence
                full_text = input_text + output_text

                # Tokenize full text
                input_ids = self.tokenizer.encode(
                    full_text,
                    add_special_tokens=False,
                    max_length=self.max_length,
                    truncation=True,
                )

                # STANDARD APPROACH: Labels = input_ids (no masking)
                # Model learns from ALL tokens, not just the continuation
                labels = input_ids.copy()

                # Skip very short examples (less than 10 tokens)
                if len(input_ids) < 10:
                    print(f"Warning: Skipping example {example_id} - too short ({len(input_ids)} tokens)")
                    continue

                # Attention mask: 1 for all tokens
                attention_mask = [1] * len(input_ids)

                sample = StandardSample(
                    id=example_id,
                    source_file=source_file,
                    var_name=var_name,
                    full_text=full_text,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )

                self.samples.append(sample)

        print(f"Loaded {len(self.samples)} training examples (standard approach, no masking)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> StandardSample:
        return self.samples[idx]


def collate_standard_batch(
    batch: List[StandardSample],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for standard training examples.

    Pads sequences to the longest in the batch.
    Returns tensors ready for causal LM training.
    """
    # Find max length in batch
    max_len = max(len(sample.input_ids) for sample in batch)

    batch_size = len(batch)

    # Initialize tensors
    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )

    labels = torch.full(
        (batch_size, max_len),
        fill_value=-100,  # Padding positions still ignored
        dtype=torch.long,
    )

    # Fill in the tensors
    for i, sample in enumerate(batch):
        seq_len = len(sample.input_ids)

        input_ids[i, :seq_len] = torch.tensor(sample.input_ids, dtype=torch.long)
        attention_mask[i, :seq_len] = torch.tensor(sample.attention_mask, dtype=torch.long)
        labels[i, :seq_len] = torch.tensor(sample.labels, dtype=torch.long)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
