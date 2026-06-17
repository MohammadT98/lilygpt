# LilyBench — demo site (gh-pages)

Companion website for the paper **"Can LLMs understand LilyPond? A benchmark for
symbolic music generation and understanding"** (Ital-IA 2026).
Lives on the `gh-pages` branch, served at
**https://cscpadova.github.io/lilybench/**. Kept separate from the Python
package on `master`.

It renders, from the real paper experiments:

- **Generation** — the same metadata-conditioned prompt sent to the four
  backbones (Phi-4, Qwen2.5-Coder-14B, DeepSeek-Coder-V2-Lite, Codestral-22B),
  with each zero-shot LilyPond output engraved (SVG) and sonified (MP3).
- **Understanding** — one real question per task with every model's greedy
  answer, plus the two result tables from the paper.

## Layout

```
index.html, style.css, script.js   static site (vanilla ES6, no build step)
.nojekyll                           serve files as-is
data/*.json                         what the page fetches (built artifact)
assets/scores/*.svg                 engraved sheet music
assets/audio/*.mp3                  sonified MIDI
demo-src/                           reproducible inputs (curated .ly + .mid + json)
build.py                            demo-src/ -> assets/ + data/   (reproducible)
curate.py                           cluster pull -> demo-src/      (needs the cluster)
```

## Rebuild

Static files are committed, so GitHub Pages needs no build. To regenerate the
rendered assets from the committed sources (needs `lilypond`, `fluidsynth`,
`ffmpeg`, a GM soundfont):

```bash
python3 build.py           # demo-src/ -> assets/ + data/
python3 -m http.server     # preview at http://localhost:8000
```

`curate.py` re-selects which samples are featured and is the only step that
needs the cluster artifacts (`/nfsd/voce/.../lilybench`); end users never run it.
