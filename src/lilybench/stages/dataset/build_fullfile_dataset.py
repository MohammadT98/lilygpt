#!/usr/bin/env python3

"""Build the full-file LilyPond training dataset from ``data/bmdataset/preprocessed``.

For each preprocessed ``.ly`` file this script:

1. Resolves metadata (composer, period, form, ensemble, part) from the filename.
2. Generates ``K`` augmented variants (original + shuffle + drop + inline).
3. Filters out variants that fail the fast syntactic gate (brace balance).
4. Chunks each surviving variant at top-level variable-assignment boundaries
   so every chunk fits in a character budget (~4× the model's token budget).
5. Prepends a ``%% === METADATA ===`` comment block to every chunk. Chunk #0
   carries the full prelude; chunks #1..N carry only ``\\version`` / ``\\language``.
6. Emits one JSONL record per chunk with ``label_mask_char_ranges`` covering the
   metadata + prelude/header region so training-time loss masking is trivial.

Design rationale lives in ``notelog.md`` at the repo root.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from .augmentations import (
    check_brace_balance,
    drop_unused_prelude,
    inline_variables,
    shuffle_prelude,
)
from .metadata_header import (
    load_metadata,
    render_metadata_block,
    resolve_metadata,
)
from .prelude import (
    PreludeInfo,
    find_version_language_header,
    identify_prelude,
    iter_toplevel_assignments,
)

DEFAULT_INPUT_DIR = Path("data/bmdataset/preprocessed")
DEFAULT_METADATA_PATH = Path("data/bmdataset/metadata.json")
DEFAULT_OUTPUT_DIR = Path("data/fullfile_dataset")
DEFAULT_MAX_CHARS = 8192  # ≈ 2048 tokens × 4 chars/token
DEFAULT_SEED = 42


@dataclass(frozen=True)
class VariantSpec:
    """One augmentation recipe applied on top of the original file."""

    name: str
    seed_salt: int


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("original", 0),
    VariantSpec("shuffle", 1),
    VariantSpec("drop", 2),
    VariantSpec("inline", 3),
)


def _apply_variant(
    text: str,
    info: PreludeInfo,
    variant: VariantSpec,
    rng: random.Random,
) -> str:
    """Apply the augmentation pipeline named by ``variant.name``.

    ``drop`` and ``inline`` stack on a shuffle so that the four variants cover
    orthogonal perturbations of the prelude; the original is passed through
    untouched.
    """
    if variant.name == "original":
        return text
    text_s = shuffle_prelude(text, info, rng)
    if variant.name == "shuffle":
        return text_s
    info_s = identify_prelude(text_s)
    if variant.name == "drop":
        return drop_unused_prelude(text_s, info_s, rng, p=0.3)
    if variant.name == "inline":
        return inline_variables(text_s, info_s, rng, p=0.5)
    raise ValueError(f"unknown variant name: {variant.name!r}")


def _body_segments(
    body: str, assignment_starts: list[int]
) -> list[tuple[int, int]]:
    """Split ``body`` into atomic segments at top-level assignment starts.

    The first segment covers any non-assignment prefix (blank lines, stray
    comments); subsequent segments each start at one assignment and end at the
    next (or at ``len(body)``).
    """
    if not assignment_starts:
        return [(0, len(body))]
    segs: list[tuple[int, int]] = []
    if assignment_starts[0] > 0:
        segs.append((0, assignment_starts[0]))
    for i, s in enumerate(assignment_starts):
        end = assignment_starts[i + 1] if i + 1 < len(assignment_starts) else len(body)
        segs.append((s, end))
    return segs


def _pack_chunks(
    segments: list[tuple[int, int]],
    chunk0_body_budget: int,
    chunki_body_budget: int,
) -> list[tuple[int, int]]:
    """Greedily pack segments into chunks, respecting per-position body budgets.

    The first chunk gets ``chunk0_body_budget`` chars for its body (it carries
    the full prelude on top); subsequent chunks get ``chunki_body_budget`` (they
    carry only the minimal ``\\version``/``\\language`` header). A single segment
    exceeding its budget is emitted as one oversize chunk (logged by the caller)
    rather than cut mid-assignment.
    """
    if not segments:
        return []
    chunks: list[tuple[int, int]] = []
    i = 0
    while i < len(segments):
        budget = chunk0_body_budget if not chunks else chunki_body_budget
        cur_s, cur_e = segments[i]
        j = i + 1
        while j < len(segments) and (segments[j][1] - cur_s) <= budget:
            cur_e = segments[j][1]
            j += 1
        chunks.append((cur_s, cur_e))
        i = j if j > i else i + 1
    return chunks


def _file_seed(stem: str) -> int:
    """Stable per-file 32-bit seed derived from the filename stem."""
    h = 0
    for ch in stem:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h or 1


def _build_records_for_file(
    path: Path,
    metadata: dict,
    key_index: list[str],
    max_chars: int,
    global_seed: int,
    variants: tuple[VariantSpec, ...] = VARIANTS,
) -> tuple[list[dict], dict]:
    """Return ``(records, stats)`` for a single preprocessed ``.ly`` file."""
    text = path.read_text(encoding="utf-8")
    stem = path.stem

    stats = {
        "source_invalid": 0,
        "variants_attempted": 0,
        "variants_dropped": 0,
        "chunks_emitted": 0,
        "oversize_chunks": 0,
    }

    if not check_brace_balance(text):
        stats["source_invalid"] = 1
        return [], stats

    meta = resolve_metadata(stem, metadata, key_index)
    info_orig = identify_prelude(text)
    file_seed = global_seed ^ _file_seed(stem)
    records: list[dict] = []

    for variant in variants:
        stats["variants_attempted"] += 1
        seed = file_seed ^ variant.seed_salt
        try:
            aug_text = _apply_variant(
                text, info_orig, variant, random.Random(seed)
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[build] {stem} / {variant.name}: augmentation error: {exc}",
                file=sys.stderr,
            )
            stats["variants_dropped"] += 1
            continue

        if not check_brace_balance(aug_text):
            stats["variants_dropped"] += 1
            continue

        aug_info = identify_prelude(aug_text)
        prelude = aug_text[: aug_info.prelude_end]
        body = aug_text[aug_info.prelude_end :]
        minimal_header = find_version_language_header(aug_text)

        asgn_starts = [
            s - aug_info.prelude_end
            for s, _ in iter_toplevel_assignments(aug_text, aug_info.prelude_end)
        ]
        meta_block = render_metadata_block(
            meta,
            rng=random.Random(seed ^ 0xABCD),
            p_field=0.15,
            p_block=0.10,
        )
        chunk0_budget = max(1, max_chars - len(meta_block) - len(prelude))
        chunki_budget = max(1, max_chars - len(meta_block) - len(minimal_header))
        chunks = _pack_chunks(
            _body_segments(body, asgn_starts), chunk0_budget, chunki_budget
        )

        for chunk_idx, (body_s, body_e) in enumerate(chunks):
            body_piece = body[body_s:body_e]
            header = prelude if chunk_idx == 0 else minimal_header
            full_text = meta_block + header + body_piece
            meta_end = len(meta_block)
            prelude_end = meta_end + len(header)
            if len(full_text) > max_chars:
                stats["oversize_chunks"] += 1
            records.append({
                "id": f"{stem}__{variant.name}__c{chunk_idx}",
                "source_file": stem,
                "variant": variant.name,
                "chunk_index": chunk_idx,
                "chunk_total": len(chunks),
                "seed": seed,
                "full_text": full_text,
                "metadata_char_range": [0, meta_end],
                "prelude_char_range": [meta_end, prelude_end],
                "label_mask_char_ranges": [[0, prelude_end]],
            })
            stats["chunks_emitted"] += 1

    return records, stats


def build_dataset(
    input_dir: Path,
    metadata_path: Path,
    output_path: Path,
    max_chars: int = DEFAULT_MAX_CHARS,
    global_seed: int = DEFAULT_SEED,
) -> dict:
    """Walk ``input_dir``, emit a JSONL at ``output_path``, return totals."""
    metadata, key_index = load_metadata(metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.ly"))
    print(f"[build] input:  {input_dir}  ({len(files)} files)")
    print(f"[build] output: {output_path}")

    totals = {
        "files": 0,
        "files_with_records": 0,
        "source_invalid": 0,
        "variants_attempted": 0,
        "variants_dropped": 0,
        "chunks_emitted": 0,
        "oversize_chunks": 0,
    }
    chunk_count_histogram: dict[int, int] = {}

    with output_path.open("w", encoding="utf-8") as f:
        for i, path in enumerate(files):
            records, stats = _build_records_for_file(
                path, metadata, key_index, max_chars, global_seed
            )
            totals["files"] += 1
            totals["source_invalid"] += stats["source_invalid"]
            totals["variants_attempted"] += stats["variants_attempted"]
            totals["variants_dropped"] += stats["variants_dropped"]
            totals["chunks_emitted"] += stats["chunks_emitted"]
            totals["oversize_chunks"] += stats["oversize_chunks"]
            if records:
                totals["files_with_records"] += 1
                per_variant_chunks: dict[str, int] = {}
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    per_variant_chunks[rec["variant"]] = rec["chunk_total"]
                for n in per_variant_chunks.values():
                    chunk_count_histogram[n] = chunk_count_histogram.get(n, 0) + 1
            if (i + 1) % 200 == 0:
                print(
                    f"[build] {i + 1}/{len(files)} files, "
                    f"{totals['chunks_emitted']} chunks, "
                    f"{totals['variants_dropped']} dropped",
                    flush=True,
                )

    print(f"[build] done: {totals}")
    if chunk_count_histogram:
        print("[build] chunk-count distribution (per variant):")
        for n in sorted(chunk_count_histogram):
            print(f"  {n} chunks: {chunk_count_histogram[n]} variants")
    return totals


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the full-file LilyPond training dataset.",
    )
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "all_examples.jsonl"),
    )
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    build_dataset(
        input_dir=Path(args.input_dir),
        metadata_path=Path(args.metadata),
        output_path=Path(args.output),
        max_chars=args.max_chars,
        global_seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
