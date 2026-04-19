"""Hugging Face cache/auth env-var helper.

Translates a Hydra ``cfg.hf`` block into the ``HF_*`` environment variables
that ``transformers`` / ``huggingface_hub`` read at ``from_pretrained`` time.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def apply_hf_env(hf_cfg: Any) -> None:
    """Export HF env vars from a Hydra ``cfg.hf`` block. No-op on null fields.

    Uses ``os.environ.setdefault`` so a pre-existing shell export (e.g. from a
    SLURM template) wins over the Hydra default.
    """
    if hf_cfg is None:
        return

    home = _stringify(hf_cfg.get("home"))
    hub_cache = _stringify(hf_cfg.get("hub_cache"))
    transformers_cache = _stringify(hf_cfg.get("transformers_cache"))
    datasets_cache = _stringify(hf_cfg.get("datasets_cache"))
    token = _stringify(hf_cfg.get("token"))
    offline = bool(hf_cfg.get("offline") or False)

    if home is not None:
        home_path = str(Path(home).expanduser())
        os.environ.setdefault("HF_HOME", home_path)
        if hub_cache is None:
            hub_cache = str(Path(home_path) / "hub")
        if transformers_cache is None:
            transformers_cache = str(Path(home_path) / "transformers")
        if datasets_cache is None:
            datasets_cache = str(Path(home_path) / "datasets")

    if hub_cache is not None:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hub_cache).expanduser()))
    if transformers_cache is not None:
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(transformers_cache).expanduser()))
    if datasets_cache is not None:
        os.environ.setdefault("HF_DATASETS_CACHE", str(Path(datasets_cache).expanduser()))

    if token is not None:
        os.environ.setdefault("HF_TOKEN", token)

    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
