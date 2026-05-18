"""Fréchet Music Distance with LilyBERT embeddings.

FMD (Retkowski et al. 2024) adapts FID to symbolic music. Given a set of
generations X and a reference set Y, embed both into a shared latent
space and compute

    FMD = ||mu_x - mu_y||² + Tr(Σ_x + Σ_y - 2·sqrt(Σ_x · Σ_y))

LilyBench uses the public **LilyBERT** encoder (Spanio et al. 2026) at
layer 6 directly on raw LilyPond text — no MIDI round-trip — so the
metric is compile-independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy import linalg


@torch.no_grad()
def lilybert_embed(
    docs: Sequence[str],
    *,
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
    max_length: int = 512,
    embed_layer: int | None = 6,
) -> np.ndarray:
    """Embed ``docs`` with the LilyBERT checkpoint at ``checkpoint``.

    ``embed_layer`` selects which hidden state to use (``6`` reproduces
    the paper). Pass ``None`` for the final hidden state.
    """
    from transformers import AutoModel, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(checkpoint))
    model = AutoModel.from_pretrained(str(checkpoint)).to(device).eval()
    want_hidden = embed_layer is not None
    chunks: list[np.ndarray] = []
    for i in range(0, len(docs), batch_size):
        batch = list(docs[i : i + batch_size])
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        out = model(**enc, output_hidden_states=want_hidden)
        cls = (
            out.hidden_states[embed_layer][:, 0, :]
            if want_hidden else out.last_hidden_state[:, 0, :]
        )
        chunks.append(cls.detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0))


def frechet_music_distance(
    x: np.ndarray, y: np.ndarray, *, eps: float = 1e-6
) -> float:
    """Compute FMD between two embedding matrices (rows = documents)."""
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError(f"need >=2 docs per set (got {x.shape[0]}, {y.shape[0]})")
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


def load_documents(paths: Iterable[str | Path], *, min_chars: int = 40) -> list[str]:
    """Read a list of ``.ly`` paths, dropping documents shorter than ``min_chars``."""
    docs: list[str] = []
    for p in paths:
        try:
            txt = Path(p).read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if len(txt) >= min_chars:
            docs.append(txt)
    return docs
