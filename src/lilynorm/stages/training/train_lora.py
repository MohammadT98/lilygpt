from __future__ import annotations

"""Train a causal language model on LilyPond data using LoRA adapters."""

import argparse
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from lilynorm.stages.dataset.training_dataset import (
    LilyStandardDataset,
    collate_standard_batch,
)
from lilynorm.stages.tokenization.special_tokens import (
    apply_special_tokens,
    build_special_tokens,
)

BANNER_LINE = "=" * 80


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for LoRA training."""
    parser = argparse.ArgumentParser(
        description="Train GPT-OSS-20B on LilyPond data with LoRA (STANDARD approach - no masking)."
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
        "--no-structural-tokens",
        action="store_true",
        help="Disable structural token injection for key/time/tempo/voice.",
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

    print(f"[train_lora] loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_structural_tokens = not args.no_structural_tokens
    special_tokens_added = 0
    trainable_token_ids: list[int] = []
    if use_structural_tokens:
        special_tokens_added = apply_special_tokens(tokenizer)
        added_vocab = tokenizer.get_added_vocab()
        trainable_tokens = [t for t in build_special_tokens() if t in added_vocab]
        trainable_token_ids = [added_vocab[t] for t in trainable_tokens]
        if special_tokens_added:
            print(f"[train_lora] added {special_tokens_added} special tokens: {trainable_tokens}")
        elif trainable_tokens:
            print(f"[train_lora] using existing special tokens: {trainable_tokens}")
        if trainable_token_ids:
            print(f"[train_lora] special token ids: {trainable_token_ids}")

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
        use_structural_tokens=use_structural_tokens,
    )
    val_dataset = LilyStandardDataset(
        val_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mask_input=args.mask_input,
        use_structural_tokens=use_structural_tokens,
    )

    print(f"[train_lora] train samples: {len(train_dataset)}")
    print(f"[train_lora] val samples:   {len(val_dataset)}")

    print(f"[train_lora] loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=_torch_dtype(args.fp16, args.bf16),
        device_map="auto",
        trust_remote_code=True,
    )
    if special_tokens_added:
        model.resize_token_embeddings(len(tokenizer))
        print(f"[train_lora] resized token embeddings to {len(tokenizer)}")

    print(
        "[train_lora] applying LoRA "
        f"(r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})"
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",  # Apply to all linear layers.
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    def _mask_embedding_grads(embedding, token_ids: list[int], label: str) -> None:
        if embedding is None or not token_ids:
            return
        weight = embedding.weight
        weight.requires_grad = True
        keep_ids = torch.tensor(sorted(set(token_ids)), dtype=torch.long)

        def _hook(grad):
            if grad is None:
                return None
            ids = keep_ids.to(grad.device)
            kept = grad.index_select(0, ids)
            grad.zero_()
            grad.index_copy_(0, ids, kept)
            return grad

        weight.register_hook(_hook)
        print(f"[train_lora] training {len(keep_ids)} token rows in {label} embeddings")

    if trainable_token_ids:
        input_emb = model.get_input_embeddings()
        output_emb = model.get_output_embeddings()
        _mask_embedding_grads(input_emb, trainable_token_ids, "input")
        if output_emb is not None and output_emb is not input_emb and output_emb.weight is not input_emb.weight:
            _mask_embedding_grads(output_emb, trainable_token_ids, "output")

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

    print("[train_lora] ✅ training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
