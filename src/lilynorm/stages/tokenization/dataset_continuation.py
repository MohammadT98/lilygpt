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
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
    ):
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self.samples: List[ContinuationSample] = []
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
                split_point = obj.get('split_point', 'unknown')

                full_text = input_text + output_text
                input_ids = self.tokenizer.encode(
                    full_text,
                    add_special_tokens=False,
                )

                input_ids_only = self.tokenizer.encode(
                    input_text,
                    add_special_tokens=False,
                )
                input_len = len(input_ids_only)

                if input_len >= self.max_length:
                    print(f"Warning: Skipping example {example_id} - input ({input_len} tokens) >= max_length ({self.max_length})")
                    continue

                if len(input_ids) > self.max_length:
                    max_output_tokens = self.max_length - input_len
                    if max_output_tokens < 10:
                        print(f"Warning: Skipping example {example_id} - insufficient output tokens after truncation")
                        continue

                    input_ids = input_ids[:self.max_length]

                labels = [-100] * input_len + input_ids[input_len:]

                if len(labels) > len(input_ids):
                    labels = labels[:len(input_ids)]

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
        fill_value=-100,  # Ignore index for loss
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
