"""
Music generation from fine-tuned LilyPond models.

This module provides simple, practical music generation for thesis demonstration.
No fancy features - just: load model, generate, save.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for music generation.
    
    Keep it simple - just the essentials for thesis work.
    """
    max_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1


class MusicGenerator:
    """Generate LilyPond music from fine-tuned GPT models.
    
    Minimal, practical implementation for thesis demonstration:
    - Load fine-tuned model (with LoRA adapters)
    - Generate continuations from prompts
    - Save to .ly files
    
    That's it. No bells and whistles.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the generator.
        
        Args:
            model_name_or_path: Base model (e.g., "openai/gpt-oss-20b")
            adapter_path: Path to LoRA adapter checkpoint (if used)
            device: Device to use ("cuda" or "cpu")
        """
        # Import at runtime to avoid requiring torch/peft for non-inference tasks
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel  # type: ignore[import-not-found]
        
        # Store for use in other methods
        self._torch = torch
        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoTokenizer = AutoTokenizer
        self._PeftModel = PeftModel
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info(f"Loading base model: {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        # Load LoRA adapter if provided
        if adapter_path:
            logger.info(f"Loading LoRA adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()  # Merge for faster inference
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded and ready for generation")
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate LilyPond music from a prompt.
        
        Args:
            prompt: LilyPond prompt (e.g., opening bars, key signature)
            config: Generation configuration
            
        Returns:
            Generated LilyPond text
        """
        config = config or GenerationConfig()
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        logger.info(f"Generating with max_length={config.max_length}, temp={config.temperature}")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=config.max_length,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                num_return_sequences=config.num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """Generate music from multiple prompts.
        
        Args:
            prompts: List of LilyPond prompts
            config: Generation configuration
            
        Returns:
            List of generated LilyPond texts
        """
        return [self.generate(prompt, config) for prompt in prompts]
    
    def generate_and_save(
        self,
        prompt: str,
        output_path: str | Path,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate music and save to .ly file.
        
        Args:
            prompt: LilyPond prompt
            output_path: Path to save .ly file
            config: Generation configuration
            
        Returns:
            Generated LilyPond text
        """
        generated = self.generate(prompt, config)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generated)
        
        logger.info(f"Saved generated music to {output_path}")
        return generated
    
    def generate_variations(
        self,
        prompt: str,
        num_variations: int = 3,
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """Generate multiple variations from the same prompt.
        
        Just run generation multiple times with sampling.
        Simple but effective for thesis demonstrations.
        
        Args:
            prompt: LilyPond prompt
            num_variations: Number of variations to generate
            config: Generation configuration
            
        Returns:
            List of generated variations
        """
        config = config or GenerationConfig()
        config.do_sample = True  # Ensure sampling for variation
        
        variations = []
        for i in range(num_variations):
            logger.info(f"Generating variation {i+1}/{num_variations}")
            variation = self.generate(prompt, config)
            variations.append(variation)
        
        return variations


def create_simple_prompt(
    key: str = "c",
    mode: str = "major",
    time_signature: str = "4/4",
) -> str:
    """Create a simple LilyPond prompt for generation.
    
    Helper function for quick testing.
    
    Args:
        key: Key (c, g, d, etc.)
        mode: Mode (major or minor)
        time_signature: Time signature
        
    Returns:
        LilyPond prompt string
    """
    mode_str = "\\major" if mode == "major" else "\\minor"
    
    prompt = f"""\\version "2.24.0"
\\language "italiano"

{{
  \\key {key} {mode_str}
  \\time {time_signature}
  \\relative do' {{
    """
    
    return prompt


# Simple CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LilyPond music")
    parser.add_argument(
        "--model",
        required=True,
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter",
        help="LoRA adapter checkpoint path",
    )
    parser.add_argument(
        "--prompt",
        help="LilyPond prompt (or use --simple-prompt)",
    )
    parser.add_argument(
        "--simple-prompt",
        action="store_true",
        help="Use a simple auto-generated prompt",
    )
    parser.add_argument(
        "--output",
        default="generated.ly",
        help="Output .ly file",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=1,
        help="Number of variations to generate",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create generator
    generator = MusicGenerator(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
    )
    
    # Get prompt
    if args.simple_prompt:
        prompt = create_simple_prompt()
        print("Using simple prompt:")
        print(prompt)
    elif args.prompt:
        prompt = args.prompt
    else:
        print("Error: Must provide --prompt or --simple-prompt")
        exit(1)
    
    # Generate config
    config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature,
    )
    
    # Generate
    if args.num_variations > 1:
        variations = generator.generate_variations(
            prompt,
            num_variations=args.num_variations,
            config=config,
        )
        for i, variation in enumerate(variations):
            output_path = Path(args.output).stem + f"_var{i+1}.ly"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(variation)
            print(f"Saved variation {i+1} to {output_path}")
    else:
        generated = generator.generate_and_save(prompt, args.output, config)
        print(f"\nGenerated {len(generated)} characters")
        print(f"Saved to {args.output}")
