from __future__ import annotations

"""Evaluate a LoRA adapter on a held-out split using held-out loss.

Hydra entry point. Run with ``python -m lilybench.evaluate.loss`` and
override config values on the command line, e.g.::

    python -m lilybench.evaluate.loss model=phi4 lora_path=runs/phi4_lora/final \\
        data=data/splits_full/test.jsonl
"""

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from lilybench.data.training_dataset import LilyStandardDataset, collate_standard_batch
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


@hydra.main(config_path="../../configs", config_name="evaluate/loss", version_base=None)
def main(cfg: DictConfig) -> int:
    apply_hf_env(cfg.get("hf"))

    data_path = _resolve_path(cfg.data)
    lora_path = _resolve_path(cfg.lora_path)

    if not data_path.exists():
        print(f"[eval_loss] data not found: {data_path}", file=sys.stderr)
        return 2
    if not lora_path.exists():
        print(f"[eval_loss] lora path not found: {lora_path}", file=sys.stderr)
        return 2

    print(BANNER_LINE)
    print("LilyBench LoRA loss evaluation")
    print(BANNER_LINE)
    print(OmegaConf.to_yaml(cfg))
    print(BANNER_LINE)

    spec = get_spec(cfg.model.id)
    print(f"[eval_loss] model registry: id={spec.model_id} hf_id={spec.hf_id} family={spec.family}")

    print(f"[eval_loss] loading tokenizer from: {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(lora_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id

    print("[eval_loss] loading dataset...")
    dataset = LilyStandardDataset(
        data_path,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    print(f"[eval_loss] samples: {len(dataset)}")

    print(f"[eval_loss] loading base model: {spec.hf_id}")
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        torch_dtype=_torch_dtype(cfg.fp16, cfg.bf16),
        device_map="auto",
        trust_remote_code=spec.trust_remote_code,
    )

    print(f"[eval_loss] loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, str(lora_path), local_files_only=True)
    model.eval()

    def collate_fn(batch):
        return collate_standard_batch(batch, pad_token_id=pad_token_id)

    eval_args = TrainingArguments(
        output_dir=str(lora_path / "eval"),
        per_device_eval_batch_size=cfg.batch_size,
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

    print("[eval_loss] evaluating...")
    metrics = trainer.evaluate()
    print("[eval_loss] metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    main()
