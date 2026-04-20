#!/usr/bin/env python3
"""Build a 1000-record metadata-conditioned prompt bank for inference sweeps.

Walks ``data/bmdataset/preprocessed/*.ly``, resolves each file's metadata via
``lilybench.preprocess.metadata_header.resolve_metadata`` (so the empirical
distribution matches the training prior exactly — each file is one draw), and
samples ``n`` records with replacement using a fixed seed. For each sampled
file emits one JSONL record with ``metadata`` (composer / period /
musical_form / ensemble / part) and a natural-language ``user_prompt``
rendered from that tuple.

The bank is reused byte-for-byte across all 15 inference runs in the sweep so
every model and regime sees identical inputs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lilybench.preprocess.metadata_header import load_metadata, resolve_metadata


def _render_user_prompt(meta_dict: dict) -> str:
    composer = meta_dict.get("composer") or "an unnamed 18th-century composer"
    period = meta_dict.get("period") or "Baroque"
    forms = meta_dict.get("musical_form") or []
    form_str = ", ".join(forms) if forms else "a short piece"
    ensemble = meta_dict.get("ensemble") or []
    ensemble_str = ", ".join(ensemble) if ensemble else "a small ensemble"
    part = meta_dict.get("part") or "full"
    part_clause = (
        "the full ensemble score"
        if part == "full"
        else f"the {part} part"
    )
    return (
        f"Write a LilyPond fragment in the style of {composer} ({period}). "
        f"Form: {form_str}. Ensemble: {ensemble_str}. Produce {part_clause}. "
        "Use Dutch (nederlands) note names and lowercase relative notation. "
        "Output only the LilyPond code; no prose, no markdown."
    )


def _meta_to_dict(meta) -> dict:
    return {
        "composer": meta.composer,
        "period": meta.period,
        "musical_form": list(meta.musical_form),
        "ensemble": list(meta.ensemble),
        "part": meta.part,
    }


def build_bank(
    preprocessed_dir: Path,
    metadata_path: Path,
    n: int,
    seed: int,
) -> list[dict]:
    metadata, key_index = load_metadata(metadata_path)
    files = sorted(preprocessed_dir.glob("*.ly"))
    if not files:
        raise SystemExit(f"no .ly files in {preprocessed_dir}")

    rng = random.Random(seed)
    sampled = [rng.choice(files) for _ in range(n)]

    records: list[dict] = []
    for idx, fp in enumerate(sampled):
        resolved = resolve_metadata(fp.stem, metadata, key_index=key_index)
        md = _meta_to_dict(resolved)
        records.append(
            {
                "id": f"bank_{idx:04d}",
                "source_file": fp.name,
                "metadata": md,
                "user_prompt": _render_user_prompt(md),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("data/bmdataset/preprocessed"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/bmdataset/metadata.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/prompt_bank/bank_1000.jsonl"),
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = build_bank(args.preprocessed_dir, args.metadata, args.n, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
