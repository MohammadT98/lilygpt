#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
cli.py - Run normalization/tokenization stages in-process (string in -> string out).

Stages contract (each exposes run(text, opts) -> str or List[int]):
- preparse.run
- normalize.run
- engrave_strip.run (skipped if keep_engraving=True)
- tokenize_gpt.run (returns token ids)

Usage examples:
  python src/lilynorm/cli.py run --input data/raw --out-root data
  python src/lilynorm/cli.py run --input C:/Users/Navid/Desktop/13.ly --show-stages
  python src/lilynorm/cli.py inspect --file C:/Users/Navid/Desktop/13.ly
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from lilynorm.utils.options import NormOptions
from lilynorm.stages import preparse, normalize, engrave_strip, tokenize_gpt

BANNER = lambda title: f"\n{'='*12} {title} {'='*12}\n"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _write(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def process_text(
    text: str,
    opts: NormOptions,
    *,
    show_stages: bool = False,
    stop_at: int = 0,
    skip_tokenize: bool = False,
    tokenizer_model: Optional[str] = None,
):
    """
    Run the pipeline and return a dictionary with stage outputs.
    """
    stage: dict[str, object] = {}

    print(BANNER("STAGE 1 | PREPARSE"))
    t1 = preparse.run(text, opts)
    if show_stages:
        print(t1.rstrip())
    stage["preparse"] = t1
    if stop_at == 1:
        return stage

    print(BANNER("STAGE 2 | NORMALIZE"))
    t2 = normalize.run(t1, opts)
    if show_stages:
        print(t2.rstrip())
    stage["normalize"] = t2
    if stop_at == 2:
        return stage

    print(BANNER("STAGE 3 | ENGRAVING"))
    t3 = t2 if opts.keep_engraving else engrave_strip.run(t2, opts)
    if show_stages:
        print(t3.rstrip())
    stage["final"] = t3
    if stop_at == 3 or skip_tokenize:
        return stage

    print(BANNER("STAGE 4 | GPT TOKENIZE"))
    tok_ids = tokenize_gpt.run(t3, model_name=tokenizer_model)
    stage["tokens"] = tok_ids
    if show_stages:
        preview = tok_ids[:24]
        suffix = "..." if len(tok_ids) > len(preview) else ""
        print(f"{preview}{suffix}")
    return stage


def _iter_input_files(in_path: Path) -> Iterable[Path]:
    if in_path.is_file():
        yield in_path
    else:
        yield from in_path.rglob("*.ly")


def cmd_inspect(file: str) -> int:
    p = Path(file)
    if not p.exists():
        print(f"ERROR: file not found: {p}")
        return 2
    txt = _read(p)
    print(f"File: {p.name}")
    print(f"bytes: {len(txt)} | lines: {txt.count(chr(10))+1} | braces: {{ {txt.count('{')} , {txt.count('}')} }}")
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
    in_p = Path(input_path)
    if not in_p.exists():
        print(f"ERROR: input path not found: {in_p}")
        return 2

    out_root_p = Path(out_root)
    (out_root_p / "normalized").mkdir(parents=True, exist_ok=True)
    if not skip_tokenize:
        (out_root_p / "tokenized").mkdir(parents=True, exist_ok=True)

    if save_intermediate:
        (out_root_p / "cleaned").mkdir(parents=True, exist_ok=True)
        (out_root_p / "normalized_stage2").mkdir(parents=True, exist_ok=True)

    opts = NormOptions(keep_engraving=keep_engraving)

    files = [in_p] if in_p.is_file() else list(in_p.rglob("*.ly"))
    if not files:
        print(f"[warn] No .ly files found under: {in_p}")
        return 0

    for f in files:
        print(BANNER(f"PROCESSING: {f}"))
        raw = _read(f)
        outs = process_text(
            raw,
            opts,
            show_stages=show_stages,
            stop_at=stop_at,
            skip_tokenize=skip_tokenize or stop_at >= 4,
            tokenizer_model=tokenizer_model,
        )

        if save_intermediate and "preparse" in outs:
            _write(out_root_p / "cleaned" / f.name, outs["preparse"])  # type: ignore[arg-type]
        if save_intermediate and "normalize" in outs:
            _write(out_root_p / "normalized_stage2" / f.name, outs["normalize"])  # type: ignore[arg-type]

        if "final" in outs:
            _write(out_root_p / "normalized" / f.name, outs["final"])  # type: ignore[arg-type]
        elif "normalize" in outs:
            _write(out_root_p / "normalized" / f.name, outs["normalize"])  # type: ignore[arg-type]
        elif "preparse" in outs:
            _write(out_root_p / "normalized" / f.name, outs["preparse"])  # type: ignore[arg-type]

        if not skip_tokenize and "tokens" in outs:
            tok_ids = outs["tokens"]  # type: ignore[assignment]
            tok_path = out_root_p / "tokenized" / (f.stem + ".tokens.json")
            tok_path.parent.mkdir(parents=True, exist_ok=True)
            tok_path.write_text(json.dumps({"input_ids": tok_ids}) + "\n", encoding="utf-8")
            print(f"[tokenize] wrote {tok_path} ({len(tok_ids)} tokens)")

    print(BANNER("DONE"))
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LilyPond normalization CLI (in-process).")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run full pipeline on file or folder.")
    r.add_argument("--input", default="data/raw", help="Path to .ly file or folder.")
    r.add_argument("--out-root", default="data", help="Output root folder.")
    r.add_argument("--keep-engraving", action="store_true", help="Keep engraving/layout elements (skip strip).")
    r.add_argument("--show-stages", action="store_true", help="Print each stage output.")
    r.add_argument("--stop-at", type=int, default=0, help="Stop after this stage (1..4).")
    r.add_argument("--save-intermediate", action="store_true", help="Also write Stage 1/2 outputs.")
    r.add_argument("--skip-tokenize", action="store_true", help="Skip GPT tokenization stage.")
    r.add_argument(
        "--tokenizer-model",
        default=tokenize_gpt.DEFAULT_MODEL_NAME,
        help="HuggingFace tokenizer name to use for Stage 4 (default: EleutherAI/gpt-neox-20b).",
    )

    i = sub.add_parser("inspect", help="Quick stats for a single file.")
    i.add_argument("--file", required=True)

    return p


def main(argv=None) -> int:
    ap = make_parser().parse_args(argv)

    if ap.cmd == "inspect":
        return cmd_inspect(ap.file)

    if ap.cmd == "run":
        return cmd_run(
            ap.input,
            ap.out_root,
            ap.keep_engraving,
            ap.show_stages,
            ap.stop_at,
            save_intermediate=getattr(ap, "save_intermediate", False),
            skip_tokenize=getattr(ap, "skip_tokenize", False),
            tokenizer_model=getattr(ap, "tokenizer_model", tokenize_gpt.DEFAULT_MODEL_NAME),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
