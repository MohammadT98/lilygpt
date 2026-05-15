"""Build the music-understanding benchmark JSONL from a Mutopia manifest.

Public entry: ``build_bench(corpus, seed, task_sizes=None) -> list[dict]``.

The bench is produced offline (no LilyPond compilation needed) and is
byte-stable given a seed: tests rely on this. ``scripts/build_understanding_bench.py``
wraps this into a CLI; see that script for the canonical invocation.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lilybench.understanding import tasks
from lilybench.understanding.bar_utils import count_bars, split_bars
from lilybench.understanding.score_metadata import (
    extract_key,
    extract_meter,
    extract_note_length,
    mask_field,
)
from lilybench.understanding.title_parser import extract_title


@dataclass
class CorpusEntry:
    """One Mutopia piece with resolved metadata."""
    key_id: str
    source_file: str
    composer: str
    style: str
    text: str
    title: str | None
    key: str | None
    meter: str | None
    note_length: str | None


# ----------------------- corpus loading -----------------------

def _resolve_mutopia_path(entry: dict, root: Path) -> Path | None:
    """Mirror ``lilybench/evaluate/fmd.py::_load_reference`` path resolution."""
    cly = entry.get("convert_ly_path")
    if cly:
        p = Path(cly).expanduser().resolve()
        if p.exists():
            return p
    rel = entry.get("localPath") or entry.get("path") or entry.get("lyFile")
    if not rel:
        return None
    p = (root / rel).resolve()
    return p if p.exists() else None


def build_corpus(manifest_path: Path, root: Path) -> list[CorpusEntry]:
    """Read ``dataset_mutopia.json`` and load every existing piece.

    Missing files are silently skipped (same convention as the FMD loader).
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        items = list(manifest.items())
    else:
        items = [(str(i), e) for i, e in enumerate(manifest)]

    out: list[CorpusEntry] = []
    for key_id, entry in items:
        if not isinstance(entry, dict):
            continue
        composer = (entry.get("composer") or "").strip()
        style = (entry.get("style") or "").strip()
        if not composer or not style:
            continue
        resolved = _resolve_mutopia_path(entry, Path(root))
        if resolved is None:
            continue
        try:
            text = resolved.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        out.append(
            CorpusEntry(
                key_id=str(key_id),
                source_file=str(resolved),
                composer=composer,
                style=style,
                text=text,
                title=extract_title(text),
                key=extract_key(text),
                meter=extract_meter(text),
                note_length=extract_note_length(text),
            )
        )
    return out


# ----------------------- per-task builders -----------------------

_METADATA_FIELDS = ("key", "meter", "note_length")


def _pick_subset(
    candidates: list[CorpusEntry],
    n: int,
    rng: random.Random,
) -> list[CorpusEntry]:
    if not candidates:
        return []
    n = min(n, len(candidates))
    return rng.sample(candidates, n)


def _build_bar_count(
    corpus: list[CorpusEntry], n: int, rng: random.Random
) -> list[dict]:
    eligible = [c for c in corpus if count_bars(c.text) >= 1]
    spec = tasks.TASKS["bar_count"]
    out = []
    for i, entry in enumerate(_pick_subset(eligible, n, rng)):
        bars = count_bars(entry.text)
        prompt = tasks.format_structured_prompt(
            input_content=entry.text,
            task_instruction=spec.task_instruction,
            structured_output_template=spec.structured_output_template,
        )
        out.append({
            "task": spec.name,
            "id": f"{spec.name}_{i:04d}",
            "source_file": entry.source_file,
            "input_content": entry.text,
            "task_instruction": spec.task_instruction,
            "structured_output_template": spec.structured_output_template,
            "gold": str(bars),
            "template_kind": spec.template_kind,
            "prompt": prompt,
        })
    return out


def _build_metadata_qa(
    corpus: list[CorpusEntry], n: int, rng: random.Random
) -> list[dict]:
    spec = tasks.TASKS["metadata_qa"]
    field_values: dict[str, list[str]] = {f: [] for f in _METADATA_FIELDS}
    for e in corpus:
        for f in _METADATA_FIELDS:
            v = getattr(e, f)
            if v:
                field_values[f].append(v)
    # Dedupe pools.
    field_values = {f: list(dict.fromkeys(v)) for f, v in field_values.items()}

    out: list[dict] = []
    eligible = [c for c in corpus if any(getattr(c, f) for f in _METADATA_FIELDS)]
    picked = _pick_subset(eligible, n, rng)
    for i, entry in enumerate(picked):
        # Cycle through fields to keep balance.
        field = _METADATA_FIELDS[i % len(_METADATA_FIELDS)]
        gold = getattr(entry, field)
        if not gold:
            # Fall back to whichever field is populated.
            for alt in _METADATA_FIELDS:
                if getattr(entry, alt):
                    field, gold = alt, getattr(entry, alt)
                    break
            else:
                continue
        pool = field_values[field]
        if len([x for x in pool if x != gold]) < 3:
            continue
        distractors = tasks.sample_distractors(pool, gold, rng, k=3)
        options = distractors + [gold]
        rng.shuffle(options)
        gold_index = options.index(gold)
        task_instruction = (
            f"{spec.task_instruction} Field: {field}."
        )
        prompt = tasks.format_mc_prompt(
            input_content=entry.text,
            task_instruction=task_instruction,
            options=options,
        )
        out.append({
            "task": spec.name,
            "id": f"{spec.name}_{i:04d}",
            "source_file": entry.source_file,
            "input_content": entry.text,
            "task_instruction": task_instruction,
            "question_field": field,
            "options": options,
            "gold": gold,
            "gold_index": gold_index,
            "template_kind": spec.template_kind,
            "prompt": prompt,
        })
    return out


def _build_bar_sequencing(
    corpus: list[CorpusEntry], n: int, rng: random.Random
) -> list[dict]:
    spec = tasks.TASKS["bar_sequencing"]
    eligible = [c for c in corpus if count_bars(c.text) >= 4]
    out: list[dict] = []
    for i, entry in enumerate(_pick_subset(eligible, n, rng)):
        bars = split_bars(entry.text)[:4]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [bars[j] for j in order]
        # The displayed labels 0..3 map to the shuffled position; gold is the
        # original-index permutation that recovers ``bars`` from ``shuffled``.
        gold = "".join(str(order.index(j)) for j in range(4))
        input_content = "\n".join(f"{idx}. {seg}" for idx, seg in enumerate(shuffled))
        prompt = tasks.format_structured_prompt(
            input_content=input_content,
            task_instruction=spec.task_instruction,
            structured_output_template=spec.structured_output_template,
        )
        out.append({
            "task": spec.name,
            "id": f"{spec.name}_{i:04d}",
            "source_file": entry.source_file,
            "input_content": input_content,
            "task_instruction": spec.task_instruction,
            "structured_output_template": spec.structured_output_template,
            "gold": gold,
            "shuffled_indices": order,
            "template_kind": spec.template_kind,
            "prompt": prompt,
        })
    return out


def _build_next_bar(
    corpus: list[CorpusEntry], n: int, rng: random.Random
) -> list[dict]:
    spec = tasks.TASKS["next_bar_prediction"]
    eligible = [c for c in corpus if count_bars(c.text) >= 5 + 3]
    out: list[dict] = []
    for i, entry in enumerate(_pick_subset(eligible, n, rng)):
        bars = split_bars(entry.text)
        context_n = 4
        context = bars[:context_n]
        gold_bar = bars[context_n]
        # Pool of distractors = bars from later in the same score.
        pool = bars[context_n + 1:]
        if len(pool) < 3:
            continue
        distractors = rng.sample(pool, 3)
        options = distractors + [gold_bar]
        rng.shuffle(options)
        gold_index = options.index(gold_bar)
        input_content = "\n".join(context)
        prompt = tasks.format_mc_prompt(
            input_content=input_content,
            task_instruction=spec.task_instruction,
            options=options,
        )
        out.append({
            "task": spec.name,
            "id": f"{spec.name}_{i:04d}",
            "source_file": entry.source_file,
            "input_content": input_content,
            "task_instruction": spec.task_instruction,
            "options": options,
            "gold": str(gold_index),
            "gold_index": gold_index,
            "template_kind": spec.template_kind,
            "prompt": prompt,
        })
    return out


def _build_metadata_prediction(
    corpus: list[CorpusEntry], n: int, rng: random.Random
) -> list[dict]:
    spec = tasks.TASKS["metadata_prediction"]
    field_values: dict[str, list[str]] = {f: [] for f in _METADATA_FIELDS}
    for e in corpus:
        for f in _METADATA_FIELDS:
            v = getattr(e, f)
            if v:
                field_values[f].append(v)
    field_values = {f: list(dict.fromkeys(v)) for f, v in field_values.items()}

    eligible = [c for c in corpus if any(getattr(c, f) for f in _METADATA_FIELDS)]
    picked = _pick_subset(eligible, n, rng)
    out: list[dict] = []
    for i, entry in enumerate(picked):
        field = _METADATA_FIELDS[i % len(_METADATA_FIELDS)]
        gold = getattr(entry, field)
        if not gold:
            for alt in _METADATA_FIELDS:
                if getattr(entry, alt):
                    field, gold = alt, getattr(entry, alt)
                    break
            else:
                continue
        pool = field_values[field]
        if len([x for x in pool if x != gold]) < 3:
            continue
        distractors = tasks.sample_distractors(pool, gold, rng, k=3)
        options = distractors + [gold]
        rng.shuffle(options)
        gold_index = options.index(gold)
        masked = mask_field(entry.text, field)
        task_instruction = f"{spec.task_instruction} Masked field: {field}."
        prompt = tasks.format_mc_prompt(
            input_content=masked,
            task_instruction=task_instruction,
            options=options,
        )
        out.append({
            "task": spec.name,
            "id": f"{spec.name}_{i:04d}",
            "source_file": entry.source_file,
            "input_content": masked,
            "task_instruction": task_instruction,
            "question_field": field,
            "options": options,
            "gold": gold,
            "gold_index": gold_index,
            "template_kind": spec.template_kind,
            "prompt": prompt,
        })
    return out


def _strip_title(ly_text: str, title: str) -> str:
    """Remove the line carrying the title from the score body."""
    # Replace the title value with an empty string inside the header.
    return ly_text.replace(f'title = "{title}"', 'title = ""')


def _build_captioning(
    corpus: list[CorpusEntry], n: int, rng: random.Random
) -> list[dict]:
    spec = tasks.TASKS["music_captioning"]
    titled = [c for c in corpus if c.title]
    title_pool = list(dict.fromkeys(c.title for c in titled if c.title))
    out: list[dict] = []
    picked = _pick_subset(titled, n, rng)
    for i, entry in enumerate(picked):
        gold = entry.title
        if not gold or len([t for t in title_pool if t != gold]) < 3:
            continue
        distractors = tasks.sample_distractors(title_pool, gold, rng, k=3)
        options = distractors + [gold]
        rng.shuffle(options)
        gold_index = options.index(gold)
        stripped_text = _strip_title(entry.text, gold)
        prompt = tasks.format_mc_prompt(
            input_content=stripped_text,
            task_instruction=spec.task_instruction,
            options=options,
        )
        out.append({
            "task": spec.name,
            "id": f"{spec.name}_{i:04d}",
            "source_file": entry.source_file,
            "input_content": stripped_text,
            "task_instruction": spec.task_instruction,
            "options": options,
            "gold": gold,
            "gold_index": gold_index,
            "template_kind": spec.template_kind,
            "prompt": prompt,
        })
    return out


def _strip_composer(ly_text: str, composer: str) -> str:
    """Remove the composer's name from the header (best-effort)."""
    # Strip any ``composer = "..."`` field value in a \header block.
    import re
    return re.sub(
        r'(composer\s*=\s*)"[^"]*"', r'\1""', ly_text
    )


def _build_composer(
    corpus: list[CorpusEntry], n: int, rng: random.Random
) -> list[dict]:
    spec = tasks.TASKS["composer_recognition"]
    pool = list(dict.fromkeys(c.composer for c in corpus if c.composer))
    out: list[dict] = []
    if len(pool) < 4:
        return out
    picked = _pick_subset(corpus, n, rng)
    for i, entry in enumerate(picked):
        gold = entry.composer
        if not gold:
            continue
        try:
            distractors = tasks.sample_distractors(pool, gold, rng, k=3)
        except ValueError:
            continue
        options = distractors + [gold]
        rng.shuffle(options)
        gold_index = options.index(gold)
        stripped = _strip_composer(entry.text, gold)
        prompt = tasks.format_mc_prompt(
            input_content=stripped,
            task_instruction=spec.task_instruction,
            options=options,
        )
        out.append({
            "task": spec.name,
            "id": f"{spec.name}_{i:04d}",
            "source_file": entry.source_file,
            "input_content": stripped,
            "task_instruction": spec.task_instruction,
            "options": options,
            "gold": gold,
            "gold_index": gold_index,
            "template_kind": spec.template_kind,
            "prompt": prompt,
        })
    return out


def _build_genre(
    corpus: list[CorpusEntry], n: int, rng: random.Random
) -> list[dict]:
    spec = tasks.TASKS["genre_recognition"]
    pool = list(dict.fromkeys(c.style for c in corpus if c.style))
    out: list[dict] = []
    if len(pool) < 4:
        return out
    picked = _pick_subset(corpus, n, rng)
    for i, entry in enumerate(picked):
        gold = entry.style
        if not gold:
            continue
        try:
            distractors = tasks.sample_distractors(pool, gold, rng, k=3)
        except ValueError:
            continue
        options = distractors + [gold]
        rng.shuffle(options)
        gold_index = options.index(gold)
        prompt = tasks.format_mc_prompt(
            input_content=entry.text,
            task_instruction=spec.task_instruction,
            options=options,
        )
        out.append({
            "task": spec.name,
            "id": f"{spec.name}_{i:04d}",
            "source_file": entry.source_file,
            "input_content": entry.text,
            "task_instruction": spec.task_instruction,
            "options": options,
            "gold": gold,
            "gold_index": gold_index,
            "template_kind": spec.template_kind,
            "prompt": prompt,
        })
    return out


_BUILDERS = {
    "bar_count": _build_bar_count,
    "metadata_qa": _build_metadata_qa,
    "bar_sequencing": _build_bar_sequencing,
    "next_bar_prediction": _build_next_bar,
    "metadata_prediction": _build_metadata_prediction,
    "music_captioning": _build_captioning,
    "composer_recognition": _build_composer,
    "genre_recognition": _build_genre,
}


def build_bench(
    corpus: list[CorpusEntry],
    seed: int,
    task_sizes: dict[str, int] | None = None,
) -> list[dict]:
    """Generate one JSONL record per (task, item).

    ``task_sizes`` overrides the per-task target counts from ``tasks.TASKS``
    (handy for tests with a tiny corpus). Records are emitted in task-name
    order to keep the JSONL diffable.
    """
    records: list[dict] = []
    for name in sorted(_BUILDERS.keys()):
        if task_sizes and name not in task_sizes:
            continue
        n = (task_sizes or {}).get(name, tasks.TASKS[name].n)
        # Each task gets its own seeded RNG derived from the master seed so
        # that subset-runs are stable.
        rng = random.Random(seed ^ hash(name) & 0xFFFFFFFF)
        records.extend(_BUILDERS[name](corpus, n, rng))
    return records


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    """Write the bench records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================================================================
# EMOPIA emotion-recognition task (separate corpus + bench)
# =========================================================================

_EMOTION_QUADRANTS = ("Q1", "Q2", "Q3", "Q4")


@dataclass
class EmotionEntry:
    """One EMOPIA clip after midi2ly conversion and 16-bar truncation."""
    clip_id: str
    song_id: str
    label: str           # one of {"Q1","Q2","Q3","Q4"}
    source_file: str
    text: str            # truncated LilyPond (≤ max_bars bars)


def _truncate_to_bars(ly_text: str, max_bars: int) -> str:
    """Return a LilyPond snippet containing the first ``max_bars`` bars.

    Falls back to the original text when ``split_bars`` finds none — better
    to keep a possibly-too-long input than to drop the entry silently.
    """
    bars = split_bars(ly_text)
    if not bars:
        return ly_text
    if len(bars) <= max_bars:
        # Reconstruct so the trailing partial (after the last ``|``) is dropped.
        body = " | ".join(bars) + " |\n"
    else:
        body = " | ".join(bars[:max_bars]) + " |\n"
    return body


def load_emotion_corpus(
    manifest_csv: Path,
    ly_root: Path,
    *,
    max_bars: int = 16,
) -> list[EmotionEntry]:
    """Read the EMOPIA manifest CSV and load each clip's truncated LilyPond.

    Manifest schema (produced by ``scripts/prepare_emopia.py``):
        clip_id, song_id, label, ly_path[, n_bars_full, n_bars_truncated]

    ``ly_path`` is resolved relative to ``ly_root`` when not absolute.
    Missing or empty files are silently skipped (midi2ly may have failed).
    """
    manifest_csv = Path(manifest_csv)
    ly_root = Path(ly_root)
    out: list[EmotionEntry] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = (row.get("label") or "").strip()
            if label not in _EMOTION_QUADRANTS:
                continue
            ly_path = Path(row.get("ly_path") or "")
            if not ly_path.is_absolute():
                ly_path = (ly_root / ly_path).resolve()
            if not ly_path.exists():
                continue
            try:
                raw = ly_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            text = _truncate_to_bars(raw, max_bars)
            if not text.strip():
                continue
            out.append(
                EmotionEntry(
                    clip_id=(row.get("clip_id") or "").strip(),
                    song_id=(row.get("song_id") or "").strip(),
                    label=label,
                    source_file=str(ly_path),
                    text=text,
                )
            )
    return out


def build_emotion_bench(
    corpus: list[EmotionEntry],
    *,
    seed: int,
    n: int = 120,
) -> list[dict]:
    """Sample a balanced bench: floor(n/4) records per quadrant.

    Options are always the full {Q1, Q2, Q3, Q4} set shuffled per record;
    the gold index is wherever the true quadrant lands in the shuffle.
    """
    spec = tasks.TASKS["emotion_recognition"]
    per_q = n // len(_EMOTION_QUADRANTS)
    rng = random.Random(seed ^ (hash("emotion_recognition") & 0xFFFFFFFF))

    by_q: dict[str, list[EmotionEntry]] = defaultdict(list)
    for e in corpus:
        by_q[e.label].append(e)

    out: list[dict] = []
    idx = 0
    for q in _EMOTION_QUADRANTS:
        bucket = by_q.get(q, [])
        if not bucket:
            continue
        take = min(per_q, len(bucket))
        picked = rng.sample(bucket, take)
        for entry in picked:
            options = list(_EMOTION_QUADRANTS)
            rng.shuffle(options)
            gold_index = options.index(entry.label)
            prompt = tasks.format_mc_prompt(
                input_content=entry.text,
                task_instruction=spec.task_instruction,
                options=options,
            )
            out.append({
                "task": spec.name,
                "id": f"{spec.name}_{idx:04d}",
                "clip_id": entry.clip_id,
                "song_id": entry.song_id,
                "source_file": entry.source_file,
                "input_content": entry.text,
                "task_instruction": spec.task_instruction,
                "options": options,
                "gold": entry.label,
                "gold_index": gold_index,
                "template_kind": spec.template_kind,
                "prompt": prompt,
            })
            idx += 1
    return out
