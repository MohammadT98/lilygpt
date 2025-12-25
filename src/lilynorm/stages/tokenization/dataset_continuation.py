"""Dataset loader for continuation-style training examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class ContinuationSample:
    """A single continuation training example."""
    id: str
    source_file: str
    var_name: str
    input_text: str
    output_text: str
    split_point: str
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]


class LilyContinuationDataset(Dataset):
    """
    Dataset for continuation-style LilyPond training.

    Each example has:
    - input_text: The prefix (e.g., "violinoI = {\\ndo4 re mi")
    - output_text: The continuation (e.g., " fa sol la si do'\\n}")

    For causal LM training:
    - input_ids = tokenize(input_text + output_text)
    - labels = [-100] * len(tokenize(input_text)) + tokenize(output_text)

    This way, loss is only computed on the continuation part.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
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

        self.samples: List[ContinuationSample] = []
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
                split_point = obj.get('split_point', 'unknown')

                # Tokenize input and output separately
                input_ids_input = self.tokenizer.encode(
                    input_text,
                    add_special_tokens=False,
                )
                input_ids_output = self.tokenizer.encode(
                    output_text,
                    add_special_tokens=False,
                )

                # Combine: input_ids = input + output
                input_ids = input_ids_input + input_ids_output

                # Labels: mask input part with -100, keep output part
                # -100 is the ignore_index in cross-entropy loss
                labels = [-100] * len(input_ids_input) + input_ids_output

                # Truncate if too long
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    labels = labels[:self.max_length]

                # Attention mask: 1 for all tokens (we'll handle padding in collate)
                attention_mask = [1] * len(input_ids)

                sample = ContinuationSample(
                    id=example_id,
                    source_file=source_file,
                    var_name=var_name,
                    input_text=input_text,
                    output_text=output_text,
                    split_point=split_point,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )

                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ContinuationSample:
        return self.samples[idx]


def collate_continuation_batch(
    batch: List[ContinuationSample],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for continuation examples.

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
        fill_value=-100,  # Ignore index for loss
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
