from __future__ import annotations

"""Train a causal language model on LilyPond data using LoRA adapters."""

import argparse
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from lilybench.models import get_spec, list_model_ids
from lilybench.stages.dataset.training_dataset import (
    LilyStandardDataset,
    collate_standard_batch,
)

BANNER_LINE = "=" * 80


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for LoRA training."""
    parser = argparse.ArgumentParser(
        description="Train a causal LM on LilyPond data with LoRA (STANDARD approach - no masking)."
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Path to train.jsonl split.",
    )
    parser.add_argument(
        "--val",
        required=True,
        help="Path to val.jsonl split.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        choices=list_model_ids(),
        help=(
            "Short id from the lilybench model registry "
            f"({', '.join(list_model_ids())}). When set, overrides --model-name."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="openai/gpt-oss-20b",
        help=(
            "HuggingFace model name. Used when --model-id is not provided, "
            "or as an explicit override (default: openai/gpt-oss-20b)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/fine_tuning/checkpoints",
        help="Directory to save checkpoints (default: data/fine_tuning/checkpoints).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train batch size (default: 1).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Gradient accumulation steps (default: 32).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024).",
    )
    parser.add_argument(
        "--mask-input",
        action="store_true",
        help="Mask input tokens so loss is only on the output/completion.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training (fp16).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision (recommended for Ampere GPUs).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps (default: 500).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps (default: 10).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory.",
    )

    return parser


def _print_training_banner(mask_input: bool) -> None:
    print(BANNER_LINE)
    print("STANDARD TRAINING APPROACH (No Loss Masking)")
    print(BANNER_LINE)
    if mask_input:
        print("This trains with INPUT MASKING (loss only on the output/completion).")
    else:
        print("This trains on FULL sequences (all tokens contribute to loss).")
    print("This is the standard approach for domain adaptation and code generation.")
    print(BANNER_LINE)
    print()


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _torch_dtype(fp16: bool, bf16: bool) -> torch.dtype:
    if fp16:
        return torch.float16
    if bf16:
        return torch.bfloat16
    return torch.float32


def main() -> int:
    """Run LoRA training for a LilyPond dataset split."""
    args = build_arg_parser().parse_args()

    train_path = _resolve_path(args.train)
    val_path = _resolve_path(args.val)
    output_dir = _resolve_path(args.output_dir)

    if not train_path.exists():
        print(f"[train_lora] train split not found: {train_path}", file=sys.stderr)
        return 2
    if not val_path.exists():
        print(f"[train_lora] val split not found: {val_path}", file=sys.stderr)
        return 2

    _print_training_banner(args.mask_input)

    spec = get_spec(args.model_id) if args.model_id else None
    model_name = spec.hf_id if spec else args.model_name
    trust_remote_code = spec.trust_remote_code if spec else True
    if spec is not None:
        print(f"[train_lora] model registry: id={spec.model_id} hf_id={spec.hf_id} family={spec.family}")

    print(f"[train_lora] loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id
    print(f"[train_lora] pad_token_id: {pad_token_id}")

    print("[train_lora] loading datasets...")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")

    train_dataset = LilyStandardDataset(
        train_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mask_input=args.mask_input,
    )
    val_dataset = LilyStandardDataset(
        val_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mask_input=args.mask_input,
    )

    print(f"[train_lora] train samples: {len(train_dataset)}")
    print(f"[train_lora] val samples:   {len(val_dataset)}")

    print(f"[train_lora] loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_torch_dtype(args.fp16, args.bf16),
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )

    lora_targets = spec.lora_target_modules if spec else "all-linear"
    print(
        "[train_lora] applying LoRA "
        f"(r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, targets={lora_targets})"
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    def collate_fn(batch):
        return collate_standard_batch(batch, pad_token_id=pad_token_id)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="no",  # Disable checkpoints to save disk space
        load_best_model_at_end=False,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "logs"),
        ddp_find_unused_parameters=False,
    )

    print("[train_lora] initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print("[train_lora] starting training...")
    if args.resume_from_checkpoint:
        print(f"[train_lora] resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    final_dir = output_dir / "final"
    print(f"[train_lora] saving final model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("[train_lora] training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
