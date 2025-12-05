"""Training stage: fine-tuning with LoRA."""

# Lazy load to avoid importing peft unless explicitly needed
def __getattr__(name):
    if name == "train":
        from .train import main, build_arg_parser
        return type("train", (), {"main": main, "build_arg_parser": build_arg_parser})()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["train"]
