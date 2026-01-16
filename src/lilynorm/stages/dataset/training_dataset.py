from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from lilynorm.stages.tokenization.special_tokens import (
    add_structural_tokens,
    add_voice_label,
)


@dataclass
class StandardSample:
    """Container for a single tokenized training sample."""

    id: str
    source_file: str
    var_name: str
    full_text: str
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]


class LilyStandardDataset(Dataset):
    """Dataset that loads JSONL samples and tokenizes them for training."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        mask_input: bool = False,
        use_structural_tokens: bool = True,
    ):
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_input = mask_input
        self.use_structural_tokens = use_structural_tokens

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self.samples: List[StandardSample] = []
        self._load_and_tokenize()

    def _load_and_tokenize(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj: Dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {exc}")
                    continue

                sample = self._build_sample(obj, line_num)
                if sample is not None:
                    self.samples.append(sample)

        print(f"Loaded {len(self.samples)} training examples (standard approach, no masking)")

    def _build_sample(self, obj: Dict[str, Any], line_num: int) -> StandardSample | None:
        example_id = obj.get("id", f"example_{line_num}")
        source_file = obj.get("source_file", "")
        var_name = obj.get("var_name", "")
        input_text = obj.get("input", "")
        output_text = obj.get("output", "")

        input_text, output_text = self._apply_tokens(input_text, output_text, var_name)
        full_text = input_text + output_text

        if self.mask_input and input_text and output_text:
            input_ids, labels = self._encode_masked(input_text, output_text)
        else:
            input_ids, labels = self._encode_full(full_text)

        if len(input_ids) < 10:
            print(f"Warning: Skipping example {example_id} - too short ({len(input_ids)} tokens)")
            return None

        attention_mask = [1] * len(input_ids)

        return StandardSample(
            id=example_id,
            source_file=source_file,
            var_name=var_name,
            full_text=full_text,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    def _apply_tokens(
        self,
        input_text: str,
        output_text: str,
        var_name: str,
    ) -> tuple[str, str]:
        if not self.use_structural_tokens:
            return input_text, output_text

        if input_text:
            input_text = add_voice_label(input_text, var_name)
        elif output_text:
            output_text = add_voice_label(output_text, var_name)

        if input_text:
            input_text = add_structural_tokens(input_text)
        if output_text:
            output_text = add_structural_tokens(output_text)

        return input_text, output_text

    def _encode_masked(self, input_text: str, output_text: str) -> tuple[List[int], List[int]]:
        input_ids_prefix = self.tokenizer.encode(
            input_text,
            add_special_tokens=False,
        )
        output_ids = self.tokenizer.encode(
            output_text,
            add_special_tokens=False,
        )
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (not output_ids or output_ids[-1] != eos_id):
            output_ids.append(eos_id)

        if self.max_length and self.max_length > 0:
            allowed_out = self.max_length - len(input_ids_prefix)
            if allowed_out <= 0:
                input_ids = input_ids_prefix[:self.max_length]
                labels = [-100] * len(input_ids)
            else:
                output_ids = output_ids[:allowed_out]
                input_ids = input_ids_prefix + output_ids
                labels = ([-100] * len(input_ids_prefix)) + output_ids
        else:
            input_ids = input_ids_prefix + output_ids
            labels = ([-100] * len(input_ids_prefix)) + output_ids

        return input_ids, labels

    def _encode_full(self, full_text: str) -> tuple[List[int], List[int]]:
        input_ids = self.tokenizer.encode(
            full_text,
            add_special_tokens=False,
        )
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            if self.max_length and self.max_length > 0:
                if len(input_ids) >= self.max_length:
                    input_ids = input_ids[:max(0, self.max_length - 1)]
            if not input_ids or input_ids[-1] != eos_id:
                input_ids.append(eos_id)
        if self.max_length and len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        return input_ids, input_ids.copy()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> StandardSample:
        return self.samples[idx]


def collate_standard_batch(
    batch: List[StandardSample],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Pad and collate a batch of StandardSample objects."""
    max_len = max(len(sample.input_ids) for sample in batch)
    batch_size = len(batch)

    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full(
        (batch_size, max_len),
        fill_value=-100,
        dtype=torch.long,
    )

    for i, sample in enumerate(batch):
        seq_len = len(sample.input_ids)
        input_ids[i, :seq_len] = torch.tensor(sample.input_ids, dtype=torch.long)
        attention_mask[i, :seq_len] = torch.tensor(sample.attention_mask, dtype=torch.long)
        labels[i, :seq_len] = torch.tensor(sample.labels, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
