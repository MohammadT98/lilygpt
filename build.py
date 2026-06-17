#!/usr/bin/env python3
"""Render the demo's static assets from ``demo-src/`` — no cluster needed.

For every generated sample: LilyPond ``.ly`` -> cropped SVG (sheet music) and
the paper's MIDI -> MP3 (FluidSynth + ffmpeg). Then emit the three JSON files
the page fetches: ``data/samples.json``, ``data/understanding.json``,
``data/results.json``. Idempotent; skips assets that already exist.

    python3 build.py            # render missing assets + (re)write data/
    python3 build.py --force    # re-render everything

Requires ``lilypond``, ``fluidsynth``, ``ffmpeg`` on PATH and a GM soundfont.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("demo-src")
DATA = Path("data")
SCORES = Path("assets/scores")
AUDIO = Path("assets/audio")
SOUNDFONTS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
]
FORCE = "--force" in sys.argv


def need(tool: str) -> str:
    p = shutil.which(tool)
    if not p:
        sys.exit(f"build.py: required tool '{tool}' not found on PATH")
    return p


def soundfont() -> str:
    for sf in SOUNDFONTS:
        if Path(sf).exists():
            return sf
    sys.exit("build.py: no GM soundfont found; install fluid-soundfont-gm")


def render_svg(ly: Path, out_stem: Path) -> None:
    """LilyPond -> cropped SVG at ``out_stem.svg`` (one tight-cropped page)."""
    final = out_stem.with_suffix(".svg")
    if final.exists() and not FORCE:
        return
    tmp = out_stem.parent / f".{out_stem.name}"
    subprocess.run(
        [need("lilypond"), "-dno-point-and-click", "-dcrop", "-fsvg",
         "-o", str(tmp), str(ly)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    cropped = tmp.with_name(tmp.name + ".cropped.svg")
    src = cropped if cropped.exists() else tmp.with_suffix(".svg")
    if not src.exists():
        raise RuntimeError(f"no SVG produced for {ly}")
    src.replace(final)
    for junk in out_stem.parent.glob(f".{out_stem.name}*"):
        junk.unlink()


def render_mp3(midi: Path, out_stem: Path, sf: str) -> None:
    """MIDI -> MP3 at ``out_stem.mp3`` via FluidSynth + ffmpeg."""
    final = out_stem.with_suffix(".mp3")
    if final.exists() and not FORCE:
        return
    wav = out_stem.with_suffix(".wav")
    subprocess.run(
        [need("fluidsynth"), "-ni", "-g", "0.8", "-r", "44100", "-F", str(wav), sf, str(midi)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    subprocess.run(
        [need("ffmpeg"), "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "128k", str(final)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    wav.unlink(missing_ok=True)


def build_generation(sf: str) -> list:
    prompts = json.loads((SRC / "generation/prompts.json").read_text())
    SCORES.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(parents=True, exist_ok=True)
    for p in prompts:
        for m in p["models"]:
            stem = f"{p['idx']:04d}_{m['id']}"
            ly = SRC / "generation/ly" / m["ly"]
            m["ly_code"] = ly.read_text(encoding="utf-8", errors="ignore")
            render_svg(ly, SCORES / stem)
            m["score"] = f"assets/scores/{stem}.svg"
            assert (SCORES / f"{stem}.svg").exists(), f"missing svg {stem}"
            if m.get("midi"):
                midi = SRC / "generation/midi" / m["midi"]
                render_mp3(midi, AUDIO / stem, sf)
                m["audio"] = f"assets/audio/{stem}.mp3"
                assert (AUDIO / f"{stem}.mp3").exists(), f"missing mp3 {stem}"
            else:
                m["audio"] = None
            m.pop("ly", None)
            m.pop("midi", None)
    return prompts


def main() -> None:
    sf = soundfont()
    DATA.mkdir(exist_ok=True)
    samples = build_generation(sf)
    (DATA / "samples.json").write_text(json.dumps(samples, indent=2))
    shutil.copyfile(SRC / "understanding/understanding.json", DATA / "understanding.json")
    shutil.copyfile(SRC / "results.json", DATA / "results.json")
    n_panels = sum(len(p["models"]) for p in samples)
    n_audio = sum(1 for p in samples for m in p["models"] if m["audio"])
    print(f"build.py: {len(samples)} prompts, {n_panels} scores, {n_audio} audio tracks")
    print("  data/ -> samples.json, understanding.json, results.json")


if __name__ == "__main__":
    main()
