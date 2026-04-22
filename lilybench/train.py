from __future__ import annotations

"""Train a causal language model on LilyPond data using LoRA adapters.

Hydra entry point. Run with ``python -m lilybench.train`` and override
config values on the command line, e.g.::

    python -m lilybench.train model=qwen-coder epochs=1 bf16=true

Submit sweeps to SLURM via the submitit launcher::

    python -m lilybench.train --multirun model=phi4,qwen-coder \\
        hydra/launcher=slurm_train
"""

import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from lilybench.data.training_dataset import (
    LilyStandardDataset,
    collate_standard_batch,
)
from lilybench.hf_cache import apply_hf_env
from lilybench.models import get_spec

BANNER_LINE = "=" * 80


def _torch_dtype(fp16: bool, bf16: bool) -> torch.dtype:
    if fp16:
        return torch.float16
    if bf16:
        return torch.bfloat16
    return torch.float32


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> int:
    apply_hf_env(cfg.get("hf"))

    train_path = _resolve_path(cfg.data.train)
    val_path = _resolve_path(cfg.data.val)
    output_dir = _resolve_path(cfg.output_dir)

    if not train_path.exists():
        print(f"[train] train split not found: {train_path}", file=sys.stderr)
        return 2
    if not val_path.exists():
        print(f"[train] val split not found: {val_path}", file=sys.stderr)
        return 2

    print(BANNER_LINE)
    print("LilyBench LoRA training")
    print(BANNER_LINE)
    print(OmegaConf.to_yaml(cfg))
    print(BANNER_LINE)

    wandb_cfg = cfg.get("wandb") or {}
    if wandb_cfg.get("project"):
        os.environ["WANDB_PROJECT"] = str(wandb_cfg.project)
    if wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(wandb_cfg.entity)
    if wandb_cfg.get("mode"):
        os.environ["WANDB_MODE"] = str(wandb_cfg.mode)
    wandb_enabled = str(wandb_cfg.get("mode") or "online").lower() != "disabled"
    wandb_run_name = wandb_cfg.get("run_name")

    tb_cfg = cfg.get("tensorboard") or {}
    tb_enabled = bool(tb_cfg.get("enabled", True))
    tb_log_dir = str(tb_cfg.get("log_dir") or (output_dir / "logs"))

    report_to = []
    if tb_enabled:
        report_to.append("tensorboard")
    if wandb_enabled:
        report_to.append("wandb")

    spec = get_spec(cfg.model.id)
    print(f"[train] model registry: id={spec.model_id} hf_id={spec.hf_id} family={spec.family}")

    print(f"[train] loading tokenizer: {spec.hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, use_fast=True, trust_remote_code=spec.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id
    print(f"[train] pad_token_id: {pad_token_id}")

    print("[train] loading datasets...")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")

    train_dataset = LilyStandardDataset(
        train_path, tokenizer=tokenizer, max_length=cfg.max_length
    )
    val_dataset = LilyStandardDataset(
        val_path, tokenizer=tokenizer, max_length=cfg.max_length
    )

    print(f"[train] train samples: {len(train_dataset)}")
    print(f"[train] val samples:   {len(val_dataset)}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    compute_dtype = _torch_dtype(cfg.fp16, cfg.bf16)
    use_qlora = bool(cfg.get("qlora", True))

    # If the checkpoint is already quantized on the Hub (e.g. gpt-oss MXFP4),
    # transformers refuses to stack a BitsAndBytesConfig on top. Load with the
    # native quantization instead — the model is still k-bit so prepare_for_kbit
    # + LoRA still apply.
    hub_config = AutoConfig.from_pretrained(
        spec.hf_id, trust_remote_code=spec.trust_remote_code
    )
    native_quantized = getattr(hub_config, "quantization_config", None) is not None

    print(
        f"[train] loading model: {spec.hf_id} "
        f"(qlora={use_qlora}, native_quantized={native_quantized}, local_rank={local_rank})"
    )
    model_kwargs = {
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": local_rank},
    }
    if use_qlora and not native_quantized:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif not native_quantized:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **model_kwargs)

    gc_enabled = bool(cfg.get("gradient_checkpointing", True))
    if use_qlora and not native_quantized:
        # Disable gc here so we can enable it once with use_reentrant=False below
        # (avoids double-enable + reentrant mismatch with Trainer).
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )
    elif native_quantized:
        # MXFP4 / natively-quantized checkpoints: prepare_model_for_kbit_training
        # upcasts non-quantized params to fp32 which OOMs on 20B+ models. LoRA
        # only needs gradient flow through the frozen quant weights — do just
        # that minimum.
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if gc_enabled:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    print(
        f"[train] applying LoRA "
        f"(r={cfg.lora.r}, alpha={cfg.lora.alpha}, dropout={cfg.lora.dropout}, "
        f"targets={spec.lora_target_modules})"
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=spec.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    model.print_trainable_parameters()

    def collate_fn(batch):
        return collate_standard_batch(batch, pad_token_id=pad_token_id)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 8)),
        dataloader_pin_memory=True,
        group_by_length=bool(cfg.get("group_by_length", True)),
        remove_unused_columns=False,
        report_to=report_to,
        run_name=wandb_run_name,
        logging_dir=tb_log_dir,
        ddp_find_unused_parameters=False,
    )

    print("[train] initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print("[train] starting training...")
    if cfg.resume_from_checkpoint:
        print(f"[train] resuming from checkpoint: {cfg.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    else:
        trainer.train()

    final_dir = output_dir / "final"
    print(f"[train] saving final model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("[train] training complete!")
    return 0


if __name__ == "__main__":
    main()
