#!/usr/bin/env python
"""
Example: Train a GPT model on LilyPond data with LoRA.

This demonstrates the complete training workflow:
1. Load and prepare datasets
2. Configure model and training
3. Train with LoRA adapters
4. Save checkpoint
5. (Optional) Generate samples during training

Perfect for your thesis - clean, documented, reproducible!
"""

from pathlib import Path
import logging
import sys

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from lilynorm.stages.tokenization import LilyTokensDataset, collate_batch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training example."""
    
    # =========================================================================
    # CONFIGURATION - Adjust these for your setup
    # =========================================================================
    
    # Data paths
    TRAIN_DATA = Path("data/splits/train.jsonl")
    VAL_DATA = Path("data/splits/val.jsonl")
    
    # Model configuration
    BASE_MODEL = "openai/gpt-oss-20b"  # or use a smaller model for testing
    MAX_LENGTH = 1024
    
    # Training configuration
    OUTPUT_DIR = Path("data/fine_tuning/checkpoints")
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    
    # LoRA configuration
    LORA_R = 8  # rank
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    # Hardware
    USE_FP16 = torch.cuda.is_available()
    
    # =========================================================================
    # STEP 1: Validate data files
    # =========================================================================
    
    logger.info("="*60)
    logger.info("STEP 1: Validating data files")
    logger.info("="*60)
    
    if not TRAIN_DATA.exists():
        logger.error(f"Training data not found: {TRAIN_DATA}")
        logger.error("Please run the data processing pipeline first!")
        sys.exit(1)
    
    if not VAL_DATA.exists():
        logger.error(f"Validation data not found: {VAL_DATA}")
        logger.error("Please run the data processing pipeline first!")
        sys.exit(1)
    
    logger.info(f"✓ Training data: {TRAIN_DATA}")
    logger.info(f"✓ Validation data: {VAL_DATA}")
    
    # =========================================================================
    # STEP 2: Load tokenizer
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Loading tokenizer")
    logger.info("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")
    
    logger.info(f"✓ Tokenizer loaded: {BASE_MODEL}")
    logger.info(f"  Vocab size: {len(tokenizer)}")
    
    # =========================================================================
    # STEP 3: Load datasets
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Loading datasets")
    logger.info("="*60)
    
    train_dataset = LilyTokensDataset(
        jsonl_path=str(TRAIN_DATA),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    
    val_dataset = LilyTokensDataset(
        jsonl_path=str(VAL_DATA),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    
    logger.info(f"✓ Training samples: {len(train_dataset)}")
    logger.info(f"✓ Validation samples: {len(val_dataset)}")
    
    # =========================================================================
    # STEP 4: Load base model
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Loading base model")
    logger.info("="*60)
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if USE_FP16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    logger.info(f"✓ Base model loaded: {BASE_MODEL}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")
    
    # =========================================================================
    # STEP 5: Apply LoRA
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Applying LoRA adapters")
    logger.info("="*60)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["c_attn", "c_proj"],  # Adjust for your model
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable_params / total_params
    
    logger.info(f"✓ LoRA applied (r={LORA_R}, alpha={LORA_ALPHA})")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable %: {trainable_pct:.2f}%")
    
    # =========================================================================
    # STEP 6: Configure training
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Configuring training")
    logger.info("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        
        # Training schedule
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # Optimization
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        weight_decay=0.01,
        
        # Logging and saving
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        
        # Evaluation
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Hardware
        fp16=USE_FP16,
        
        # Other
        report_to=["tensorboard"],  # or "none" if you don't want tensorboard
        dataloader_num_workers=0,  # Adjust based on your system
    )
    
    logger.info(f"✓ Training configuration:")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  FP16: {USE_FP16}")
    logger.info(f"  Output dir: {OUTPUT_DIR}")
    
    # =========================================================================
    # STEP 7: Create trainer and train!
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("STEP 7: Starting training")
    logger.info("="*60)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda batch: collate_batch(batch, tokenizer),
    )
    
    # Train!
    logger.info("Starting training... This may take a while!")
    logger.info("You can monitor progress with: tensorboard --logdir " + str(OUTPUT_DIR))
    
    trainer.train()
    
    # =========================================================================
    # STEP 8: Save final model
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("STEP 8: Saving final model")
    logger.info("="*60)
    
    final_model_path = OUTPUT_DIR / "final_model"
    final_model_path.mkdir(exist_ok=True)
    
    # Save LoRA adapter
    model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    
    logger.info(f"✓ Model saved to: {final_model_path}")
    
    # =========================================================================
    # DONE!
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"Checkpoints in: {OUTPUT_DIR}")
    logger.info("\nNext steps:")
    logger.info("1. Check the tensorboard logs")
    logger.info("2. Use the model for generation (see generate_music.py)")
    logger.info("3. Evaluate generated music")
    logger.info("4. Report results in your thesis!")
    
    logger.info("\n" + "="*60)
    logger.info("Thesis tip: Document these numbers!")
    logger.info("="*60)
    logger.info(f"- Training samples: {len(train_dataset)}")
    logger.info(f"- Validation samples: {len(val_dataset)}")
    logger.info(f"- Model parameters: {total_params:,}")
    logger.info(f"- Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    logger.info(f"- Training epochs: {NUM_EPOCHS}")
    logger.info(f"- Final eval loss: {trainer.state.best_metric:.4f}")


if __name__ == "__main__":
    main()
