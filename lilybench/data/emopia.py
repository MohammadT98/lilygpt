"""Loader for the EMOPIA emotion-recognition corpus.

EMOPIA distributes pop-piano MIDI clips labelled with the four Russell
valence-arousal quadrants. The clips are converted to LilyPond using
``midi2ly`` (see :mod:`lilybench.understanding.midi_to_lily`) and bundled
on Zenodo as a CSV manifest plus a directory of ``.ly`` files. The first
``max_bars`` bars of each clip are kept so the prompt fits the context
budget of all four registered backbones.
"""

from __future__ import annotations

import csv
from pathlib import Path

from lilybench.data.types import CorpusEntry
from lilybench.understanding.bar_utils import split_bars


_EMOTION_QUADRANTS = ("Q1", "Q2", "Q3", "Q4")


def _truncate_to_bars(ly_text: str, max_bars: int) -> str:
    bars = split_bars(ly_text)
    if not bars:
        return ly_text
    keep = bars if len(bars) <= max_bars else bars[:max_bars]
    return " | ".join(keep) + " |\n"


def load_emopia(
    manifest_csv: str | Path,
    ly_root: str | Path,
    *,
    max_bars: int = 16,
) -> list[CorpusEntry]:
    """Read the EMOPIA manifest CSV and truncate each clip to ``max_bars``.

    Manifest schema (produced by ``scripts/prepare_emopia.py``):
    ``clip_id``, ``song_id``, ``label``, ``ly_path`` plus optional extras.
    """
    manifest_csv = Path(manifest_csv)
    ly_root = Path(ly_root)
    out: list[CorpusEntry] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("label") or "").strip()
            if label not in _EMOTION_QUADRANTS:
                continue
            ly = Path(row.get("ly_path") or "")
            if not ly.is_absolute():
                ly = (ly_root / ly).resolve()
            if not ly.exists():
                continue
            try:
                raw = ly.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            text = _truncate_to_bars(raw, max_bars)
            if not text.strip():
                continue
            out.append(
                CorpusEntry(
                    source_id=(row.get("clip_id") or ly.stem).strip(),
                    source_file=str(ly),
                    text=text,
                    extras={
                        "clip_id": (row.get("clip_id") or "").strip(),
                        "song_id": (row.get("song_id") or "").strip(),
                        "label": label,
                    },
                )
            )
    return out
