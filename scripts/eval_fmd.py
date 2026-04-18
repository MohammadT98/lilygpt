#!/usr/bin/env python3
"""Compute Fréchet Music Distance (FMD) between generated and reference LilyPond.

FMD (Retkowski et al., 2024) adapts Fréchet Inception Distance to symbolic music:
given embeddings X (generations) and Y (reference), fit Gaussians N(mu_x, Sigma_x)
and N(mu_y, Sigma_y) and compute

    FMD = ||mu_x - mu_y||^2 + Tr(Sigma_x + Sigma_y - 2 * sqrtm(Sigma_x * Sigma_y))

This script uses LilyBERT (CSCPadova/lilybert) as the symbolic-music embedding
model, applied directly to LilyPond source text (no MIDI round-trip). Report
FMD against two reference sets per the LilyBench protocol:
  - in-domain  : held-out test split (test.jsonl 'output' fields)
  - out-of-domain : Mutopia LilyPond corpus (directory of .ly files)

Usage:
  python scripts/eval_fmd.py \
    --generations-dir data/inference/samples/phi4_zero \
    --reference-kind test \
    --reference-path data/splits_full/test.jsonl \
    --embedder-checkpoint /path/to/lilybert \
    --out data/inference/sample_eval/fmd_phi4_zero.json

  python scripts/eval_fmd.py \
    --generations-dir data/inference/samples/phi4_zero \
    --reference-kind mutopia \
    --reference-path data/mutopia_ly \
    --embedder-checkpoint /path/to/lilybert \
    --out data/inference/sample_eval/fmd_phi4_zero_mutopia.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import linalg
from transformers import AutoModel, AutoTokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generations-dir",
        required=True,
        help="Directory containing generated .ly files (e.g. from extract_detokenized.py).",
    )
    parser.add_argument(
        "--reference-kind",
        required=True,
        choices=["test", "mutopia"],
        help="Reference set type. 'test' expects a JSONL split; 'mutopia' expects a dir of .ly files.",
    )
    parser.add_argument(
        "--reference-path",
        required=True,
        help="Path to reference JSONL (kind=test) or directory (kind=mutopia).",
    )
    parser.add_argument(
        "--embedder-checkpoint",
        required=True,
        help="Path or HF id of the LilyBERT checkpoint.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON file with FMD result and metadata.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length for LilyBERT input (default 512 = CodeBERT max).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Skip documents shorter than this (likely degenerate outputs).",
    )
    return parser.parse_args()


def _load_generations(path: Path, min_chars: int) -> list[str]:
    docs: list[str] = []
    for p in sorted(path.rglob("*.ly")):
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if len(txt) >= min_chars:
            docs.append(txt)
    return docs


def _load_reference(kind: str, path: Path, min_chars: int) -> list[str]:
    if kind == "test":
        docs: list[str] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Reassemble a minimal LilyPond document from the assignment 'input' + 'output'.
                input_part = rec.get("input", "")
                output_part = rec.get("output", "")
                doc = f"{input_part}\n{output_part}".strip()
                if len(doc) >= min_chars:
                    docs.append(doc)
        return docs
    # kind == "mutopia"
    return _load_generations(path, min_chars)


@torch.no_grad()
def _embed_corpus(
    docs: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Return (N, D) array of [CLS]-token embeddings."""
    embeddings: list[np.ndarray] = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        # [CLS] position, typical BERT convention.
        cls = out.last_hidden_state[:, 0, :]
        embeddings.append(cls.detach().float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _fmd(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between two Gaussian fits."""
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)

    diff = mu_x - mu_y

    # Matrix sqrt of product — follow the numerical-stability recipe from
    # Dowson & Landau / pytorch-fid.
    covmean, _ = linalg.sqrtm(sigma_x @ sigma_y, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_x.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_x + offset) @ (sigma_y + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_x) + np.trace(sigma_y) - 2 * np.trace(covmean))


def main() -> int:
    args = _parse_args()
    gen_dir = Path(args.generations_dir).expanduser().resolve()
    ref_path = Path(args.reference_path).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not gen_dir.exists():
        print(f"[fmd] generations dir not found: {gen_dir}", file=sys.stderr)
        return 2
    if not ref_path.exists():
        print(f"[fmd] reference path not found: {ref_path}", file=sys.stderr)
        return 2

    gens = _load_generations(gen_dir, args.min_chars)
    refs = _load_reference(args.reference_kind, ref_path, args.min_chars)
    if len(gens) < 2 or len(refs) < 2:
        print(f"[fmd] need >=2 docs in each set (got {len(gens)} generations, {len(refs)} reference)", file=sys.stderr)
        return 2

    print(f"[fmd] generations: {len(gens)}  reference({args.reference_kind}): {len(refs)}")
    print(f"[fmd] loading embedder: {args.embedder_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.embedder_checkpoint, use_fast=True)
    model = AutoModel.from_pretrained(args.embedder_checkpoint).to(args.device).eval()

    print("[fmd] embedding generations...")
    x = _embed_corpus(gens, tokenizer, model, args.device, args.batch_size, args.max_length)
    print("[fmd] embedding reference...")
    y = _embed_corpus(refs, tokenizer, model, args.device, args.batch_size, args.max_length)

    value = _fmd(x, y)
    print(f"[fmd] FMD = {value:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "fmd": value,
        "generations_dir": str(gen_dir),
        "reference_kind": args.reference_kind,
        "reference_path": str(ref_path),
        "n_generations": len(gens),
        "n_reference": len(refs),
        "embedder": args.embedder_checkpoint,
        "max_length": args.max_length,
    }, indent=2), encoding="utf-8")
    print(f"[fmd] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
