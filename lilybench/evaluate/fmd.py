#!/usr/bin/env python3
"""Compute Fréchet Music Distance (FMD) between generated and reference LilyPond.

FMD (Retkowski et al., 2024) adapts Fréchet Inception Distance to symbolic music:
given embeddings X (generations) and Y (reference), fit Gaussians N(mu_x, Sigma_x)
and N(mu_y, Sigma_y) and compute

    FMD = ||mu_x - mu_y||^2 + Tr(Sigma_x + Sigma_y - 2 * sqrtm(Sigma_x * Sigma_y))

This script uses LilyBERT (csc-unipd/lilybert) as the symbolic-music embedding
model, applied directly to LilyPond source text (no MIDI round-trip). Report
FMD against two reference sets per the LilyBench protocol:
  - in-domain  : held-out test split (test.jsonl 'output' fields)
  - out-of-domain : Mutopia LilyPond corpus (directory of .ly files)

Hydra entry point. Run with ``python -m lilybench.evaluate.fmd`` and override
config values on the command line, e.g.::

    python -m lilybench.evaluate.fmd \\
        generations_dir=data/inference/samples/phi4_zero \\
        reference_kind=test \\
        reference_path=data/splits_full/test.jsonl \\
        embedder_checkpoint=/path/to/lilybert
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy import linalg
from transformers import AutoModel, PreTrainedTokenizerFast

from lilybench.hf_cache import apply_hf_env


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
                doc = rec.get("full_text", "").strip()
                if len(doc) >= min_chars:
                    docs.append(doc)
        return docs
    if kind == "mutopia":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        root = path.parent
        entries = manifest.values() if isinstance(manifest, dict) else manifest
        docs = []
        for entry in entries:
            if isinstance(entry, dict):
                rel = entry.get("localPath") or entry.get("path") or entry.get("lyFile")
            else:
                rel = entry
            if not rel:
                continue
            p = (root / rel).resolve()
            if not p.exists():
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
            if len(txt) >= min_chars:
                docs.append(txt)
        return docs
    return _load_generations(path, min_chars)


@torch.no_grad()
def _embed_corpus(
    docs: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
    embed_layer: int | None,
) -> np.ndarray:
    want_hidden = embed_layer is not None
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
        out = model(**enc, output_hidden_states=want_hidden)
        if want_hidden:
            cls = out.hidden_states[embed_layer][:, 0, :]
        else:
            cls = out.last_hidden_state[:, 0, :]
        embeddings.append(cls.detach().float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _fmd(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)

    diff = mu_x - mu_y

    covmean, _ = linalg.sqrtm(sigma_x @ sigma_y, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_x.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_x + offset) @ (sigma_y + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_x) + np.trace(sigma_y) - 2 * np.trace(covmean))


@hydra.main(config_path="../../configs", config_name="evaluate/fmd", version_base=None)
def main(cfg: DictConfig) -> int:
    apply_hf_env(cfg.get("hf"))
    print(OmegaConf.to_yaml(cfg))

    gen_dir = Path(cfg.generations_dir).expanduser().resolve()
    ref_path = Path(cfg.reference_path).expanduser().resolve()
    out_path = Path(cfg.out).expanduser().resolve()

    if not gen_dir.exists():
        print(f"[fmd] generations dir not found: {gen_dir}", file=sys.stderr)
        return 2
    if not ref_path.exists():
        print(f"[fmd] reference path not found: {ref_path}", file=sys.stderr)
        return 2

    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    embed_layer = cfg.get("embed_layer")
    if embed_layer is not None:
        embed_layer = int(embed_layer)
    ref_cache = cfg.get("reference_embeddings_path")
    ref_cache_path = Path(ref_cache).expanduser().resolve() if ref_cache else None

    gens = _load_generations(gen_dir, cfg.min_chars)

    refs: list[str] | None = None
    y: np.ndarray | None = None
    n_reference: int | None = None

    if ref_cache_path is not None and ref_cache_path.exists():
        print(f"[fmd] loading cached reference embeddings: {ref_cache_path}")
        with np.load(ref_cache_path) as npz:
            y = npz["embeddings"]
            n_reference = int(npz["n"]) if "n" in npz else int(y.shape[0])
    else:
        refs = _load_reference(cfg.reference_kind, ref_path, cfg.min_chars)
        n_reference = len(refs)

    if len(gens) < 2 or (n_reference is not None and n_reference < 2):
        print(
            f"[fmd] need >=2 docs in each set (got {len(gens)} generations, {n_reference} reference)",
            file=sys.stderr,
        )
        return 2

    print(
        f"[fmd] generations: {len(gens)}  reference({cfg.reference_kind}): {n_reference}  "
        f"embed_layer={embed_layer if embed_layer is not None else 'final'}"
    )
    print(f"[fmd] loading embedder: {cfg.embedder_checkpoint}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.embedder_checkpoint)
    model = AutoModel.from_pretrained(cfg.embedder_checkpoint).to(device).eval()

    print("[fmd] embedding generations...")
    x = _embed_corpus(
        gens, tokenizer, model, device, cfg.batch_size, cfg.max_length, embed_layer
    )

    if y is None:
        assert refs is not None
        print("[fmd] embedding reference...")
        y = _embed_corpus(
            refs, tokenizer, model, device, cfg.batch_size, cfg.max_length, embed_layer
        )
        if ref_cache_path is not None:
            ref_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                ref_cache_path,
                embeddings=y,
                n=np.asarray(len(refs), dtype=np.int64),
                embed_layer=np.asarray(
                    -1 if embed_layer is None else embed_layer, dtype=np.int64
                ),
            )
            print(f"[fmd] cached reference embeddings to {ref_cache_path}")

    value = _fmd(x, y)
    print(f"[fmd] FMD = {value:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "fmd": value,
                "generations_dir": str(gen_dir),
                "reference_kind": cfg.reference_kind,
                "reference_path": str(ref_path),
                "reference_embeddings_path": str(ref_cache_path) if ref_cache_path else None,
                "n_generations": len(gens),
                "n_reference": int(n_reference),
                "embedder": cfg.embedder_checkpoint,
                "embed_layer": embed_layer,
                "max_length": cfg.max_length,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[fmd] wrote {out_path}")
    return 0


if __name__ == "__main__":
    main()
