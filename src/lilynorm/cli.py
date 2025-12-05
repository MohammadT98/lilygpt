from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List

from lilynorm.utils.options import NormOptions
from lilynorm.stages import preprocessing as preparse_module
from lilynorm.stages import preprocessing
from lilynorm.stages import tokenization as tokenize_gpt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> str:
    """
    Build a visual banner string for stage headings.
    """
    return f"\n{'=' * 12} {title} {'=' * 12}\n"


def _read(path: Path) -> str:
    """
    Read UTF-8 text from a file, ignoring decoding errors.
    """
    return path.read_text(encoding="utf-8", errors="ignore")


def _write(path: Path, content: str) -> None:
    """
    Write UTF-8 text to a file, creating parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _iter_input_files(in_path: Path) -> Iterable[Path]:
    """
    Yield all .ly files to process.

    If in_path is a file, yield it.
    If in_path is a directory, recursively yield all *.ly files.
    """
    if in_path.is_file():
        yield in_path
    else:
        yield from in_path.rglob("*.ly")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_text(
    text: str,
    opts: NormOptions,
    *,
    show_stages: bool = False,
    stop_at: int = 0,
    skip_tokenize: bool = False,
    tokenizer_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the in-process normalization pipeline on a single LilyPond string.

    Stages:
      1) preparse
      2) normalize
      3) engrave_strip (optional, depending on keep_engraving)
      4) tokenize_gpt (optional)

    Returns a dict with keys among:
      "preparse", "normalize", "final", "tokens"
    depending on stop_at/skip_tokenize.
    """
    stage_outputs: Dict[str, Any] = {}

    # --- Stage 1: PREPARSE ---
    print(banner("STAGE 1 | PREPARSE"))
    preparse_text = preprocessing.preparse.run(text, opts)
    if show_stages:
        print(preparse_text.rstrip())
    stage_outputs["preparse"] = preparse_text
    if stop_at == 1:
        return stage_outputs

    # --- Stage 2: NORMALIZE ---
    print(banner("STAGE 2 | NORMALIZE"))
    normalized_text = preprocessing.normalize.run(preparse_text, opts)
    if show_stages:
        print(normalized_text.rstrip())
    stage_outputs["normalize"] = normalized_text
    if stop_at == 2:
        return stage_outputs

    # --- Stage 3: ENGRAVING (strip or keep) ---
    print(banner("STAGE 3 | ENGRAVING"))
    final_text = (
        normalized_text if opts.keep_engraving else preprocessing.engrave_strip.run(normalized_text, opts)
    )
    if show_stages:
        print(final_text.rstrip())
    stage_outputs["final"] = final_text
    if stop_at == 3 or skip_tokenize:
        return stage_outputs

    # --- Stage 4: GPT TOKENIZE ---
    print(banner("STAGE 4 | GPT TOKENIZE"))
    token_ids: List[int] = tokenize_gpt.run(final_text, model_name=tokenizer_model)
    stage_outputs["tokens"] = token_ids

    if show_stages:
        preview = token_ids[:24]
        suffix = "..." if len(token_ids) > len(preview) else ""
        print(f"{preview}{suffix}")

    return stage_outputs


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_inspect(file: str) -> int:
    """
    Inspect a single .ly file: print basic stats (bytes, lines, brace counts).
    """
    path = Path(file)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 2

    text = _read(path)
    lines = text.count(chr(10)) + 1
    left_braces = text.count("{")
    right_braces = text.count("}")

    print(f"File: {path.name}")
    print(
        f"bytes: {len(text)} | "
        f"lines: {lines} | "
        f"braces: {{ {left_braces} , {right_braces} }}"
    )
    return 0


def cmd_run(
    input_path: str,
    out_root: str,
    keep_engraving: bool,
    show_stages: bool,
    stop_at: int,
    *,
    save_intermediate: bool = False,
    skip_tokenize: bool = False,
    tokenizer_model: str = tokenize_gpt.DEFAULT_MODEL_NAME,
) -> int:
    """
    Run the full normalization pipeline on a file or folder of LilyPond sources.

    Writes:
      - normalized/*.ly       (final normalized output)
      - tokenized/*.tokens.json (if tokenization is enabled)
      - cleaned/*.ly          (optional, Stage 1)
      - normalized_stage2/*.ly (optional, Stage 2)
    """
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        print(f"ERROR: input path not found: {input_path_obj}")
        return 2

    out_root_path = Path(out_root)
    (out_root_path / "normalized").mkdir(parents=True, exist_ok=True)
    if not skip_tokenize:
        (out_root_path / "tokenized").mkdir(parents=True, exist_ok=True)

    if save_intermediate:
        (out_root_path / "cleaned").mkdir(parents=True, exist_ok=True)
        (out_root_path / "normalized_stage2").mkdir(parents=True, exist_ok=True)

    opts = NormOptions(keep_engraving=keep_engraving)

    files = [input_path_obj] if input_path_obj.is_file() else list(
        _iter_input_files(input_path_obj)
    )
    if not files:
        print(f"[warn] No .ly files found under: {input_path_obj}")
        return 0

    for file_path in files:
        print(banner(f"PROCESSING: {file_path}"))
        raw_text = _read(file_path)

        stage_outputs = process_text(
            raw_text,
            opts,
            show_stages=show_stages,
            stop_at=stop_at,
            skip_tokenize=skip_tokenize or stop_at >= 4,
            tokenizer_model=tokenizer_model,
        )

        # Optionally save intermediate stages
        if save_intermediate and "preparse" in stage_outputs:
            _write(
                out_root_path / "cleaned" / file_path.name,
                stage_outputs["preparse"],  # type: ignore[arg-type]
            )
        if save_intermediate and "normalize" in stage_outputs:
            _write(
                out_root_path / "normalized_stage2" / file_path.name,
                stage_outputs["normalize"],  # type: ignore[arg-type]
            )

        # Final normalized output fallback chain
        if "final" in stage_outputs:
            _write(
                out_root_path / "normalized" / file_path.name,
                stage_outputs["final"],  # type: ignore[arg-type]
            )
        elif "normalize" in stage_outputs:
            _write(
                out_root_path / "normalized" / file_path.name,
                stage_outputs["normalize"],  # type: ignore[arg-type]
            )
        elif "preparse" in stage_outputs:
            _write(
                out_root_path / "normalized" / file_path.name,
                stage_outputs["preparse"],  # type: ignore[arg-type]
            )

        # Token IDs (if present and not skipped)
        if not skip_tokenize and "tokens" in stage_outputs:
            token_ids = stage_outputs["tokens"]  # type: ignore[assignment]
            tok_path = out_root_path / "tokenized" / (file_path.stem + ".tokens.json")
            tok_path.parent.mkdir(parents=True, exist_ok=True)
            tok_path.write_text(
                json.dumps({"input_ids": token_ids}) + "\n",
                encoding="utf-8",
            )
            print(f"[tokenize] wrote {tok_path} ({len(token_ids)} tokens)")

    print(banner("DONE"))
    return 0


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    """
    Build the top-level argument parser for the CLI.
    """
    parser = argparse.ArgumentParser(
        description="LilyPond normalization CLI (in-process).",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run full pipeline on file or folder.",
    )
    run_parser.add_argument(
        "--input",
        default="data/raw",
        help="Path to .ly file or folder.",
    )
    run_parser.add_argument(
        "--out-root",
        default="data",
        help="Output root folder.",
    )
    run_parser.add_argument(
        "--keep-engraving",
        action="store_true",
        help="Keep engraving/layout elements (skip strip).",
    )
    run_parser.add_argument(
        "--show-stages",
        action="store_true",
        help="Print each stage output.",
    )
    run_parser.add_argument(
        "--stop-at",
        type=int,
        default=0,
        help="Stop after this stage (1..4).",
    )
    run_parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Also write Stage 1/2 outputs.",
    )
    run_parser.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Skip GPT tokenization stage.",
    )
    run_parser.add_argument(
        "--tokenizer-model",
        default=tokenize_gpt.DEFAULT_MODEL_NAME,
        help=(
            "HuggingFace tokenizer name to use for Stage 4 "
            "(default: EleutherAI/gpt-neox-20b)."
        ),
    )

    # inspect subcommand
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Quick stats for a single file.",
    )
    inspect_parser.add_argument(
        "--file",
        required=True,
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """
    Main entry point for the CLI.

    Returns an exit code.
    """
    parsed_args = make_parser().parse_args(argv)

    if parsed_args.cmd == "inspect":
        return cmd_inspect(parsed_args.file)

    if parsed_args.cmd == "run":
        return cmd_run(
            parsed_args.input,
            parsed_args.out_root,
            parsed_args.keep_engraving,
            parsed_args.show_stages,
            parsed_args.stop_at,
            save_intermediate=getattr(parsed_args, "save_intermediate", False),
            skip_tokenize=getattr(parsed_args, "skip_tokenize", False),
            tokenizer_model=getattr(
                parsed_args,
                "tokenizer_model",
                tokenize_gpt.DEFAULT_MODEL_NAME,
            ),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())