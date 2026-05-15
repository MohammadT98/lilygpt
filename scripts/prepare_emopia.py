#!/usr/bin/env python3
"""Download EMOPIA, convert each MIDI to LilyPond, and emit a manifest CSV.

One-shot prep for the music-understanding emotion-recognition task. The
output manifest has columns ``clip_id, song_id, label, ly_path,
n_bars_full, n_bars_truncated`` and is consumed by
``scripts/build_emotion_bench.py``.

Default invocation (on the cluster, submitted via ``slurm/misc/prepare_emopia.slurm``)::

    python scripts/prepare_emopia.py \\
      --zip-url https://zenodo.org/records/5257995/files/EMOPIA_2.2.zip \\
      --workdir /nfsd/voce/machine_learning/datasets/emopia/_workdir \\
      --out-root /nfsd/voce/machine_learning/datasets/emopia \\
      --midi2ly /home/spanio/lilypond-2.24.4/bin/midi2ly \\
      --jobs 16
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from lilybench.understanding.bar_utils import count_bars
from lilybench.understanding.midi_to_lily import convert_midi_to_lily


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip-url", type=str,
                   default="https://zenodo.org/records/5257995/files/EMOPIA_2.2.zip")
    p.add_argument("--workdir", type=Path, required=True,
                   help="Scratch dir for zip + extracted tree.")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Destination for the lilypond/ tree + manifest CSV.")
    p.add_argument("--midi2ly", type=str, default="midi2ly",
                   help="Path to the midi2ly binary.")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--timeout-s", type=int, default=30,
                   help="Per-clip midi2ly timeout.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of MIDI files processed (for smoke runs).")
    return p.parse_args()


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[prep-emopia] zip already present at {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prep-emopia] downloading {url}  →  {dest}")
    subprocess.run(
        ["wget", "--quiet", "--show-progress", "-O", str(dest), url],
        check=True,
    )


def _unzip(zip_path: Path, dest: Path) -> Path:
    """Unzip into ``dest`` (idempotent). Return path to the EMOPIA root inside."""
    dest.mkdir(parents=True, exist_ok=True)
    sentinel = dest / "EMOPIA_2.2"
    if sentinel.exists() and any(sentinel.iterdir()):
        print(f"[prep-emopia] EMOPIA tree already extracted at {sentinel}")
        return sentinel
    print(f"[prep-emopia] unzipping {zip_path}  →  {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    if not sentinel.exists():
        # Some EMOPIA distributions zip without the top-level dir; fall back.
        for child in dest.iterdir():
            if child.is_dir() and (child / "midis").exists():
                return child
        raise FileNotFoundError(f"could not locate EMOPIA root under {dest}")
    return sentinel


def _read_label_csv(label_csv: Path) -> dict[str, str]:
    """Return ``{clip_stem: 'Qn'}`` from EMOPIA's ``label.csv``."""
    out: dict[str, str] = {}
    with label_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stem = (row.get("clip_name") or row.get("filename") or row.get("clip")
                    or "").strip()
            if stem.endswith(".mid"):
                stem = stem[:-4]
            label = (row.get("4Q") or row.get("label") or row.get("emotion")
                     or "").strip()
            if not stem or not label:
                continue
            if not label.startswith("Q"):
                label = f"Q{label}"
            out[stem] = label
    return out


def _read_song_map(song_csv: Path) -> dict[str, str]:
    """Return ``{clip_stem: song_id}`` from EMOPIA's ``metadata_by_song.csv``."""
    out: dict[str, str] = {}
    if not song_csv.exists():
        return out
    with song_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            song_id = (row.get("song_id") or row.get("YouTube_id")
                       or row.get("youtube_id") or "").strip()
            clips_field = (row.get("clip_name") or row.get("clips")
                           or row.get("clip_list") or "").strip()
            if not song_id or not clips_field:
                continue
            for clip in clips_field.replace(";", ",").split(","):
                clip = clip.strip().rstrip(".mid")
                if clip:
                    out[clip] = song_id
    return out


def _convert_one(args: tuple) -> tuple[str, bool, int, int]:
    """Worker: convert one MIDI → .ly, return survey counts."""
    midi_path_str, out_path_str, midi2ly, timeout_s, max_bars = args
    midi_path = Path(midi_path_str)
    out_path = Path(out_path_str)
    result = convert_midi_to_lily(
        midi_path=midi_path,
        out_path=out_path,
        midi2ly_bin=midi2ly,
        timeout_s=timeout_s,
    )
    if result is None:
        return midi_path.stem, False, 0, 0
    try:
        text = out_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return midi_path.stem, False, 0, 0
    full_bars = count_bars(text)
    trunc_bars = min(full_bars, max_bars)
    return midi_path.stem, True, full_bars, trunc_bars


def main() -> int:
    args = _parse_args()
    workdir = args.workdir.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    ly_dir = out_root / "lilypond"
    ly_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    zip_path = workdir / "EMOPIA_2.2.zip"
    _download(args.zip_url, zip_path)
    emopia_root = _unzip(zip_path, workdir)

    label_csv = emopia_root / "label.csv"
    song_csv = emopia_root / "metadata_by_song.csv"
    if not label_csv.exists():
        print(f"[prep-emopia] label.csv not found at {label_csv}", file=sys.stderr)
        return 2
    label_map = _read_label_csv(label_csv)
    song_map = _read_song_map(song_csv)
    print(f"[prep-emopia] labels: {len(label_map)} clips  songs: {len(song_map)}")

    midi_dir = emopia_root / "midis"
    midi_paths = sorted(midi_dir.glob("*.mid")) + sorted(midi_dir.glob("*.MID"))
    if args.limit:
        midi_paths = midi_paths[: args.limit]
    print(f"[prep-emopia] {len(midi_paths)} MIDI files to convert "
          f"(jobs={args.jobs}, timeout_s={args.timeout_s})")

    payloads = [
        (str(p), str(ly_dir / f"{p.stem}.ly"), args.midi2ly, args.timeout_s, 16)
        for p in midi_paths
    ]

    successes: list[tuple[str, int, int]] = []
    failures = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for fut in as_completed(ex.submit(_convert_one, p) for p in payloads):
            stem, ok, full_bars, trunc_bars = fut.result()
            if ok:
                successes.append((stem, full_bars, trunc_bars))
            else:
                failures += 1
            total = len(successes) + failures
            if total % 100 == 0:
                print(f"[prep-emopia] processed {total}/{len(midi_paths)} "
                      f"(ok={len(successes)} fail={failures})")

    # Write manifest.
    manifest_path = out_root / "emopia_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["clip_id", "song_id", "label", "ly_path",
                        "n_bars_full", "n_bars_truncated"],
        )
        w.writeheader()
        rows_by_quad: Counter = Counter()
        for stem, full_bars, trunc_bars in sorted(successes):
            label = label_map.get(stem, "")
            if not label:
                continue
            song_id = song_map.get(stem, stem.rsplit("_", 1)[0])
            w.writerow({
                "clip_id": stem,
                "song_id": song_id,
                "label": label,
                "ly_path": str((ly_dir / f"{stem}.ly").relative_to(out_root)),
                "n_bars_full": full_bars,
                "n_bars_truncated": trunc_bars,
            })
            rows_by_quad[label] += 1

    print(
        f"[prep-emopia] wrote {manifest_path}  "
        f"(successes={len(successes)} failures={failures})"
    )
    for q in ("Q1", "Q2", "Q3", "Q4"):
        print(f"        {q}: {rows_by_quad.get(q, 0)} clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
