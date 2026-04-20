"""Smoke tests for the new FMD options (embed_layer, mutopia loader)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lilybench.evaluate.fmd import _embed_corpus, _load_reference


class _FakeTokenizerOutput(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __call__(self, batch, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        n = len(batch)
        return _FakeTokenizerOutput(
            input_ids=torch.zeros((n, 4), dtype=torch.long),
            attention_mask=torch.ones((n, 4), dtype=torch.long),
        )


class _FakeOutput:
    def __init__(self, last_hidden_state, hidden_states):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _FakeModel:
    def __init__(self, dim: int = 8, n_layers: int = 8):
        self.dim = dim
        self.n_layers = n_layers

    def __call__(self, *, input_ids, attention_mask, output_hidden_states=False):
        n, seq_len = input_ids.shape
        last = torch.full((n, seq_len, self.dim), 99.0)
        if output_hidden_states:
            hs = tuple(
                torch.full((n, seq_len, self.dim), float(layer))
                for layer in range(self.n_layers + 1)
            )
            return _FakeOutput(last_hidden_state=last, hidden_states=hs)
        return _FakeOutput(last_hidden_state=last, hidden_states=None)


def test_embed_corpus_with_layer_selects_right_hidden_state():
    tok = _FakeTokenizer()
    mdl = _FakeModel(dim=4, n_layers=8)
    docs = ["doc one body", "doc two body"]
    emb = _embed_corpus(docs, tok, mdl, device="cpu", batch_size=2, max_length=16, embed_layer=6)
    assert emb.shape == (2, 4)
    # Layer 6 embeddings are filled with the float 6.0 in the fake model
    assert np.allclose(emb, 6.0)


def test_embed_corpus_none_layer_falls_back_to_last_hidden_state():
    tok = _FakeTokenizer()
    mdl = _FakeModel(dim=4, n_layers=8)
    docs = ["doc one body"]
    emb = _embed_corpus(docs, tok, mdl, device="cpu", batch_size=2, max_length=16, embed_layer=None)
    assert emb.shape == (1, 4)
    assert np.allclose(emb, 99.0)


def test_load_reference_mutopia_resolves_relative_paths(tmp_path: Path):
    (tmp_path / "scores").mkdir()
    a = tmp_path / "scores" / "a.ly"
    b = tmp_path / "scores" / "b.ly"
    a.write_text('\\version "2.24.0"\n\\relative { c4 d e f g a b c }\n', encoding="utf-8")
    b.write_text('\\version "2.24.0"\n\\relative { g4 a b c d e f g }\n', encoding="utf-8")
    manifest = tmp_path / "dataset_mutopia.json"
    manifest.write_text(
        json.dumps(
            {
                "piece_a": {"path": "scores/a.ly"},
                "piece_b": {"path": "scores/b.ly"},
                "missing": {"path": "scores/does_not_exist.ly"},
            }
        ),
        encoding="utf-8",
    )
    docs = _load_reference("mutopia", manifest, min_chars=10)
    assert len(docs) == 2
    assert all("\\version" in d for d in docs)


def test_load_reference_mutopia_accepts_list_manifest(tmp_path: Path):
    (tmp_path / "scores").mkdir()
    p = tmp_path / "scores" / "x.ly"
    p.write_text('\\relative { c4 d e f g a b c }\n', encoding="utf-8")
    manifest = tmp_path / "dataset_mutopia.json"
    manifest.write_text(json.dumps([{"path": "scores/x.ly"}]), encoding="utf-8")
    docs = _load_reference("mutopia", manifest, min_chars=5)
    assert len(docs) == 1
