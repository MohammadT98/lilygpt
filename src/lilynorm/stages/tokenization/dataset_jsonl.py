from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader

from lilynorm.stages.tokenization.tokenize_gpt import DEFAULT_MAX_LENGTH


@dataclass
class JsonlSample:
    id: str
    rel_path: str
    input_ids: List[int]
    attention_mask: List[int] | None = None
    token_count: int | None = None
    truncated: bool | None = None


class LilyTokensDataset(Dataset):
    """
    Dataset that loads pre-tokenized LilyPond pieces from a JSONL split
    (train.jsonl / val.jsonl / test.jsonl).

    Each line should contain at least:
        {"id": ..., "rel_path": ..., "input_ids": [...]}
    """

    def __init__(self, jsonl_path: str | Path) -> None:
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.samples: List[JsonlSample] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj: Dict[str, Any] = json.loads(line)

                input_ids = obj.get("input_ids")
                if not isinstance(input_ids, list):
                    raise ValueError(f"{self.path}: line without 'input_ids' list")

                sample = JsonlSample(
                    id=obj.get("id", ""),
                    rel_path=obj.get("rel_path", ""),
                    input_ids=input_ids,
                    attention_mask=obj.get("attention_mask"),
                    token_count=obj.get("token_count"),
                    truncated=obj.get("truncated"),
                )
                self.samples.append(sample)

        if not self.samples:
            raise RuntimeError(f"{self.path} is empty or invalid")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> JsonlSample:
        return self.samples[idx]


def collate_batch(
    batch: List[JsonlSample],
    *,
    pad_token_id: int,
    max_length: int | None = None,
) -> Dict[str, torch.Tensor | List[str]]:
    """
    Convert a list of JsonlSample into padded tensors for GPT training.

    - Pads to the longest sequence in the batch, or to max_length if provided.
    - attention_mask: 1 for real tokens, 0 for padding.
    - labels = input_ids (causal LM).
    
    Note: Padding tokens in labels should be ignored during loss computation
    by setting ignore_index=pad_token_id (or -100) in your loss function.
    """
    # lengths before trunc / pad
    lengths = [len(s.input_ids) for s in batch]

    if max_length is None:
        target_len = max(lengths)
    else:
        target_len = min(max(lengths), max_length)

    batch_size = len(batch)

    input_ids = torch.full(
        (batch_size, target_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (batch_size, target_len),
        dtype=torch.long,
    )

    sample_ids = []
    for i, sample in enumerate(batch):
        ids = sample.input_ids[:target_len]
        seq_len = len(ids)
        input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :seq_len] = 1
        sample_ids.append(sample.id)

    labels = input_ids.clone()
    # Ignore padding in the loss: set pad positions to -100
    labels[input_ids == pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_ids": sample_ids,
    }


def create_dataloader(
    jsonl_path: str | Path,
    *,
    pad_token_id: int,
    batch_size: int = 1,
    shuffle: bool = True,
    max_length: int | None = DEFAULT_MAX_LENGTH,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience helper: build a DataLoader over one split.
    """
    dataset = LilyTokensDataset(jsonl_path)

    def _collate_fn(batch: List[JsonlSample]) -> Dict[str, torch.Tensor | List[str]]:
        return collate_batch(
            batch,
            pad_token_id=pad_token_id,
            max_length=max_length,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        num_workers=num_workers,
    )
