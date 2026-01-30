from __future__ import annotations

"""Evaluate a LoRA adapter on a held-out split (test/val)."""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from lilynorm.stages.dataset.training_dataset import LilyStandardDataset, collate_standard_batch
from lilynorm.stages.tokenization.special_tokens import apply_special_tokens, build_special_tokens

BANNER_LINE = "=" * 80


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter on a dataset split.")
    parser.add_argument("--data", required=True, help="Path to JSONL split (e.g., test.jsonl).")
    parser.add_argument(
        "--model-name",
        default="openai/gpt-oss-20b",
        help="Base HuggingFace model id (default: openai/gpt-oss-20b).",
    )
    parser.add_argument(
        "--lora-path",
        required=True,
        help="Path to LoRA adapter folder (e.g., runs/exp/final).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device eval batch size (default: 1).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048).",
    )
    parser.add_argument(
        "--mask-input",
        action="store_true",
        help="Mask input tokens (loss only on completion).",
    )
    parser.add_argument(
        "--no-structural-tokens",
        action="store_true",
        help="Disable structural token injection.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 dtype for base model.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 dtype for base model.",
    )
    return parser


def _torch_dtype(fp16: bool, bf16: bool) -> torch.dtype:
    if fp16:
        return torch.float16
    if bf16:
        return torch.bfloat16
    return torch.float32


def main() -> int:
    args = build_arg_parser().parse_args()

    data_path = Path(args.data).expanduser().resolve()
    lora_path = Path(args.lora_path).expanduser().resolve()

    if not data_path.exists():
        print(f"[eval_lora] data not found: {data_path}", file=sys.stderr)
        return 2
    if not lora_path.exists():
        print(f"[eval_lora] lora path not found: {lora_path}", file=sys.stderr)
        return 2

    print(BANNER_LINE)
    print("EVALUATION: LoRA adapter")
    print(BANNER_LINE)

    print(f"[eval_lora] loading tokenizer from: {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(lora_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_structural_tokens = not args.no_structural_tokens
    if use_structural_tokens:
        apply_special_tokens(tokenizer)

    pad_token_id = tokenizer.pad_token_id

    print("[eval_lora] loading dataset...")
    dataset = LilyStandardDataset(
        data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mask_input=args.mask_input,
        use_structural_tokens=use_structural_tokens,
    )
    print(f"[eval_lora] samples: {len(dataset)}")

    print(f"[eval_lora] loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=_torch_dtype(args.fp16, args.bf16),
        device_map="auto",
        trust_remote_code=True,
    )

    # Make sure embeddings match tokenizer (special tokens)
    model.resize_token_embeddings(len(tokenizer))

    print(f"[eval_lora] loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, str(lora_path), local_files_only=True)
    model.eval()

    def collate_fn(batch):
        return collate_standard_batch(batch, pad_token_id=pad_token_id)

    eval_args = TrainingArguments(
        output_dir=str(lora_path / "eval"),
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=dataset,
        data_collator=collate_fn,
    )

    print("[eval_lora] evaluating...")
    metrics = trainer.evaluate()
    print("[eval_lora] metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
