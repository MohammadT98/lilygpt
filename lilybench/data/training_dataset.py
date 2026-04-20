from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class StandardSample:
    """Container for a single tokenized training sample."""

    id: str
    source_file: str
    full_text: str
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_file": self.source_file,
            "full_text": self.full_text,
            "input_ids": self.input_ids,
            "labels": self.labels,
            "attention_mask": self.attention_mask,
        }


class LilyStandardDataset(Dataset):
    """Dataset that loads the full-file JSONL schema and tokenizes for training.

    Each record must carry ``full_text`` + ``label_mask_char_ranges``. Every char
    range in the list is converted to a token range and those labels are set to
    ``-100`` so the model conditions on them (metadata header + prelude) without
    being trained to reproduce them.
    """

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

    def _load_and_tokenize(self) -> None:
        n_masked = 0
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
                    if any(label == -100 for label in sample.labels):
                        n_masked += 1

        print(
            f"Loaded {len(self.samples)} training examples "
            f"({n_masked} with char-range loss masking)"
        )

    def _build_sample(self, obj: Dict[str, Any], line_num: int) -> StandardSample | None:
        example_id = obj.get("id", f"example_{line_num}")
        source_file = obj.get("source_file", "")

        full_text = obj["full_text"]
        mask_ranges = obj.get("label_mask_char_ranges") or []
        input_ids, labels = self._encode_with_char_mask(full_text, mask_ranges)

        if len(input_ids) < 10:
            print(f"Warning: Skipping example {example_id} - too short ({len(input_ids)} tokens)")
            return None

        attention_mask = [1] * len(input_ids)

        return StandardSample(
            id=example_id,
            source_file=source_file,
            full_text=full_text,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    def _encode_with_char_mask(
        self, full_text: str, mask_ranges: List[List[int]]
    ) -> Tuple[List[int], List[int]]:
        """Tokenize ``full_text`` and build labels with ``mask_ranges`` set to ``-100``.

        Uses the tokenizer's offset-mapping (fast tokenizers only) to map
        character ranges to token indices. Falls back to a prefix-tokenization
        count when offsets aren't available.
        """
        encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=self.tokenizer.is_fast,
            truncation=False,
        )
        input_ids: List[int] = list(encoding["input_ids"])
        eos_id = self.tokenizer.eos_token_id

        if self.tokenizer.is_fast:
            offsets: List[Tuple[int, int]] = list(encoding["offset_mapping"])
            token_mask_flags = [False] * len(input_ids)
            for char_start, char_end in mask_ranges:
                for t_idx, (tok_s, tok_e) in enumerate(offsets):
                    if tok_e == tok_s:
                        continue
                    if tok_e <= char_start or tok_s >= char_end:
                        continue
                    token_mask_flags[t_idx] = True
        else:
            token_mask_flags = self._mask_flags_via_prefix(full_text, mask_ranges)

        labels = [
            -100 if flag else tok
            for tok, flag in zip(input_ids, token_mask_flags)
        ]

        if eos_id is not None and (not input_ids or input_ids[-1] != eos_id):
            input_ids.append(eos_id)
            labels.append(eos_id)

        if self.max_length and len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        return input_ids, labels

    def _mask_flags_via_prefix(
        self, full_text: str, mask_ranges: List[List[int]]
    ) -> List[bool]:
        """Fallback when the tokenizer lacks offset mapping.

        For each ``[char_start, char_end]``, tokenize ``full_text[:char_start]``
        and ``full_text[:char_end]`` to derive the corresponding token indices.
        O(len(mask_ranges)) extra tokenizations — acceptable since the file is
        small relative to the overall tokenization cost.
        """
        full_len = len(
            self.tokenizer.encode(full_text, add_special_tokens=False)
        )
        flags = [False] * full_len
        for char_start, char_end in mask_ranges:
            if char_end <= char_start:
                continue
            prefix_tokens = len(
                self.tokenizer.encode(
                    full_text[:char_start], add_special_tokens=False
                )
            )
            to_end_tokens = len(
                self.tokenizer.encode(
                    full_text[:char_end], add_special_tokens=False
                )
            )
            for i in range(prefix_tokens, min(to_end_tokens, full_len)):
                flags[i] = True
        return flags

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx].as_dict()


def collate_standard_batch(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Pad and collate a batch of tokenized samples (dict items)."""
    max_len = max(len(sample["input_ids"]) for sample in batch)
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
        seq_len = len(sample["input_ids"])
        input_ids[i, :seq_len] = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask[i, :seq_len] = torch.tensor(sample["attention_mask"], dtype=torch.long)
        labels[i, :seq_len] = torch.tensor(sample["labels"], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
