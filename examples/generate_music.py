#!/usr/bin/env python
"""
Example: Generate LilyPond music from a fine-tuned model.

This demonstrates the complete generation workflow:
1. Load your fine-tuned model
2. Create prompts
3. Generate music
4. Save to .ly files
5. (Optional) Evaluate results

Perfect for your thesis demonstration!
"""

from pathlib import Path
import logging

from lilynorm.inference import MusicGenerator, GenerationConfig, create_simple_prompt
from lilynorm.evaluation import evaluate_generated_files, print_evaluation_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main generation example."""
    
    # =========================================================================
    # CONFIGURATION - Adjust these for your setup
    # =========================================================================
    
    # Model paths
    BASE_MODEL = "openai/gpt-oss-20b"  # or your chosen base model
    ADAPTER_PATH = "data/fine_tuning/checkpoints/checkpoint-1000"  # your trained adapter
    
    # Output directory
    OUTPUT_DIR = Path("data/generated_music")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generation settings
    NUM_PIECES = 5
    MAX_LENGTH = 512
    TEMPERATURE = 1.0  # Higher = more creative, lower = more conservative
    
    # =========================================================================
    # STEP 1: Load the fine-tuned model
    # =========================================================================
    
    logger.info("="*60)
    logger.info("STEP 1: Loading fine-tuned model")
    logger.info("="*60)
    
    generator = MusicGenerator(
        model_name_or_path=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
        device="cuda",  # or "cpu" if no GPU
    )
    
    # =========================================================================
    # STEP 2: Create prompts
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Creating prompts")
    logger.info("="*60)
    
    # Example 1: Simple auto-generated prompts
    prompts = [
        create_simple_prompt(key="do", mode="major", time_signature="4/4"),
        create_simple_prompt(key="sol", mode="major", time_signature="3/4"),
        create_simple_prompt(key="re", mode="minor", time_signature="4/4"),
    ]
    
    # Example 2: Custom prompts with actual music
    custom_prompt = """\\version "2.24.0"
\\language "italiano"

{
  \\key do \\major
  \\time 4/4
  \\relative do' {
    do4 re mi fa |
    sol2 sol2 |
"""
    prompts.append(custom_prompt)
    
    # Add more if requested
    while len(prompts) < NUM_PIECES:
        prompts.append(create_simple_prompt())
    
    prompts = prompts[:NUM_PIECES]
    
    logger.info(f"Created {len(prompts)} prompts")
    
    # =========================================================================
    # STEP 3: Generate music
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Generating music")
    logger.info("="*60)
    
    config = GenerationConfig(
        max_length=MAX_LENGTH,
        temperature=TEMPERATURE,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
    )
    
    generated_files = []
    
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"\nGenerating piece {i}/{len(prompts)}...")
        
        output_path = OUTPUT_DIR / f"generated_{i:03d}.ly"
        
        try:
            generated = generator.generate_and_save(
                prompt=prompt,
                output_path=output_path,
                config=config,
            )
            
            logger.info(f"✓ Generated {len(generated)} characters")
            logger.info(f"  Saved to: {output_path}")
            generated_files.append(output_path)
            
        except Exception as e:
            logger.error(f"✗ Failed to generate piece {i}: {e}")
    
    logger.info(f"\nSuccessfully generated {len(generated_files)}/{len(prompts)} pieces")
    
    # =========================================================================
    # STEP 4: Generate variations (optional)
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Generating variations (optional)")
    logger.info("="*60)
    
    if prompts:
        logger.info("Generating 3 variations of the first prompt...")
        
        variations = generator.generate_variations(
            prompt=prompts[0],
            num_variations=3,
            config=config,
        )
        
        for i, variation in enumerate(variations, 1):
            var_path = OUTPUT_DIR / f"variation_{i}.ly"
            with open(var_path, "w", encoding="utf-8") as f:
                f.write(variation)
            logger.info(f"✓ Saved variation {i} to {var_path}")
            generated_files.append(var_path)
    
    # =========================================================================
    # STEP 5: Evaluate generated music (optional but recommended!)
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Evaluating generated music")
    logger.info("="*60)
    
    try:
        summary = evaluate_generated_files(
            file_paths=generated_files,
            lilypond_command="lilypond",  # adjust if needed
        )
        
        print_evaluation_summary(summary)
        
        # Save evaluation results
        import json
        eval_path = OUTPUT_DIR / "evaluation_summary.json"
        
        # Convert stats to serializable format
        serializable_summary = {
            "total_files": summary["total_files"],
            "valid_syntax_count": summary["valid_syntax_count"],
            "valid_syntax_rate": summary["valid_syntax_rate"],
            "avg_total_notes": summary["avg_total_notes"],
            "avg_unique_pitches": summary["avg_unique_pitches"],
            "avg_repetition_ratio": summary["avg_repetition_ratio"],
            "avg_interval": summary["avg_interval"],
        }
        
        with open(eval_path, "w") as f:
            json.dump(serializable_summary, f, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_path}")
        
    except FileNotFoundError:
        logger.warning("LilyPond not found - skipping syntax validation")
        logger.warning("Install LilyPond to enable full evaluation")
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
    
    # =========================================================================
    # DONE!
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("GENERATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"Generated files saved to: {OUTPUT_DIR}")
    logger.info(f"Total files: {len(generated_files)}")
    logger.info("\nNext steps:")
    logger.info("1. Review the generated .ly files")
    logger.info("2. Compile them with LilyPond to create PDFs")
    logger.info("3. Listen to the MIDI output")
    logger.info("4. Use these results in your thesis!")


if __name__ == "__main__":
    main()
