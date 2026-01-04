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
    id: str
    source_file: str
    var_name: str
    full_text: str
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]


class LilyStandardDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
    ):
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self.samples: List[StandardSample] = []
        self._load_and_tokenize()

    def _load_and_tokenize(self):
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

                example_id = obj.get('id', f'example_{line_num}')
                source_file = obj.get('source_file', '')
                var_name = obj.get('var_name', '')
                input_text = obj.get('input', '')
                output_text = obj.get('output', '')

                full_text = input_text + output_text

                input_ids = self.tokenizer.encode(
                    full_text,
                    add_special_tokens=False,
                    max_length=self.max_length,
                    truncation=True,
                )

                labels = input_ids.copy()

                if len(input_ids) < 10:
                    print(f"Warning: Skipping example {example_id} - too short ({len(input_ids)} tokens)")
                    continue

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
    max_len = max(len(sample.input_ids) for sample in batch)

    batch_size = len(batch)

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
