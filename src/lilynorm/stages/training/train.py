"""
Train GPT-OSS-20B on LilyPond dataset using LoRA.

Usage:
    python -m lilynorm.stages.train --train data/splits/train.jsonl --val data/splits/val.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from lilynorm.stages.tokenization.dataset_continuation import (
    LilyContinuationDataset,
    collate_continuation_batch
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train GPT-OSS-20B on LilyPond data with LoRA."
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
        "--model-name",
        default="openai/gpt-oss-20b",
        help="HuggingFace model name (default: openai/gpt-oss-20b).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/fine_tuning/checkpoints",
        help="Directory to save checkpoints (default: data/fine_tuning/checkpoints).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device train batch size (default: 4).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4).",
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


def main() -> int:
    args = build_arg_parser().parse_args()

    train_path = Path(args.train).expanduser().resolve()
    val_path = Path(args.val).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not train_path.exists():
        print(f"[train] train split not found: {train_path}", file=sys.stderr)
        return 2
    if not val_path.exists():
        print(f"[train] val split not found: {val_path}", file=sys.stderr)
        return 2

    # Load tokenizer FIRST (needed for dataset)
    print(f"[train] loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id
    print(f"[train] pad_token_id: {pad_token_id}")

    print(f"[train] loading datasets...")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")

    # Load datasets (continuation-style)
    train_dataset = LilyContinuationDataset(
        train_path,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    val_dataset = LilyContinuationDataset(
        val_path,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    print(f"[train] train samples: {len(train_dataset)}")
    print(f"[train] val samples:   {len(val_dataset)}")

    # Load model
    print(f"[train] loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    print(f"[train] applying LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn", "c_proj"],  # GPT-OSS specific; adjust if needed
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Custom collate function for continuation examples
    def collate_fn(batch):
        return collate_continuation_batch(batch, pad_token_id=pad_token_id)

    # Training arguments
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
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=0,  # Set >0 if you have CPU cores to spare
        remove_unused_columns=False,  # Important: our dataset returns custom dicts
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "logs"),
        ddp_find_unused_parameters=False,
    )

    # Trainer
    print("[train] initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Train
    print("[train] starting training...")
    if args.resume_from_checkpoint:
        print(f"[train] resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    print(f"[train] saving final model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("[train] ✅ training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
