#!/usr/bin/env python3
"""Translate italian pitch names to nederlands in ``data/bmdataset/preprocessed/``.

The bmdataset ``preprocessed/`` corpus declares ``\\language "nederlands"`` in
most files but actually contains italian pitch names (``re``, ``la``, ``fa``,
``mi``, ``si``, ``do``, ``sol``) — a discrepancy inherited from the upstream
preprocessing. See ``notelog.md`` §8.1 for the finding.

This script fixes the corpus in place. For each file it force-rewrites the
``\\language`` directive to ``italiano`` (injecting one if missing), runs
python-ly's ``ly.pitch.translate`` to convert all italian notes to nederlands,
and writes the result back. python-ly's italiano pitch reader only matches
italian note names (``do``, ``re``, …) — single-letter nederlands notes
(``a``, ``b``, ``c``, ``cis``, ``bes``, …) pass through unchanged — so the
translate call is safe on mixed-notation files and idempotent on already-
nederlands files.

Run once after installing ``python-ly``. Re-running is a no-op (cheap).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import ly.document
import ly.pitch.translate

LANGUAGE_RE = re.compile(r'\\language\s+"(\w+)"')


def translate_text(text: str) -> str:
    """Return ``text`` with italian pitch names rewritten to nederlands."""
    m = LANGUAGE_RE.search(text)
    if m:
        forced = text[: m.start()] + '\\language "italiano"' + text[m.end() :]
    else:
        forced = '\\language "italiano"\n' + text
    doc = ly.document.Document(forced, mode="lilypond")
    ly.pitch.translate.translate(ly.document.Cursor(doc), "nederlands")
    return doc.plaintext()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        default="data/bmdataset/preprocessed",
        help="Directory holding the .ly files to translate (modified in place).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report per-file byte deltas without writing.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    root = Path(args.input_dir)
    files = sorted(root.glob("*.ly"))
    if not files:
        print(f"[translate] no .ly files under {root}", file=sys.stderr)
        return 2

    print(f"[translate] {len(files)} files under {root}")
    n_changed = 0
    n_errors = 0
    for i, path in enumerate(files):
        text = path.read_text(encoding="utf-8")
        try:
            out = translate_text(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[translate] ERROR {path.name}: {exc}", file=sys.stderr)
            n_errors += 1
            continue
        if out != text:
            n_changed += 1
            if not args.dry_run:
                path.write_text(out, encoding="utf-8")
        if (i + 1) % 200 == 0:
            print(f"[translate] {i + 1}/{len(files)} changed={n_changed} errors={n_errors}", flush=True)

    verb = "would change" if args.dry_run else "changed"
    print(f"[translate] done: {verb}={n_changed} errors={n_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
