"""Rescue truncated LoRA-generated LilyPond samples.

LoRA models trained on full-file train.jsonl records produce 200+ bar pieces
that exceed the inference token budget and end mid-line, leaving unbalanced
braces and uncompilable output (0% compile rate observed). This script does
a best-effort surgical fix:

  1. Lex the file tracking brace depth, ignoring strings (`"..."`),
     line comments (`% ...`) and block comments (`%{ ... %}`).
  2. Find the **last bar separator `|`** that sits outside strings/comments
     and inside at least one open brace level (i.e. inside music, not the
     header). Truncate after it.
  3. Append the closing punctuation needed to balance: `>>` for each open
     simultaneous block, `}` for each open curly. If we're still inside an
     unclosed `\\score { ... }`, also add `\\layout {} \\midi {}` before
     the final `}` so the score actually compiles to MIDI.

Not a parser — just a heuristic. Won't fix Scheme corruptions or content
inside an unterminated string. Empirically recovers most truncated outputs
that have valid LilyPond up to the cut.

Usage:
    python scripts/rescue_lora_outputs.py \\
        --input-dir /nfsd/.../inference/phi4_lora/samples \\
        --output-dir /nfsd/.../inference/phi4_lora_rescued/samples
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_VERSION_RE = re.compile(r'\\version\s+"([^"]+)"')
_LANGUAGE_RE = re.compile(r'\\language\s+"([^"]+)"')


def _scan(text: str) -> dict:
    """Walk text once, tracking lexical state and brace depth.

    Returns a dict with:
      - depth_curly: int (final unclosed `{` count)
      - depth_angle: int (final unclosed `<<` count)
      - last_bar_pos: int — position right after the last `|` outside
        strings/comments and inside at least one open brace, or -1 if none.
      - score_open: bool — whether we are inside an unclosed `\\score { ... }`
        block at the end.
      - has_layout: bool — whether `\\layout` keyword appeared inside the
        currently-open score block.
      - has_midi: bool — whether `\\midi` keyword appeared inside the
        currently-open score block.
    """
    n = len(text)
    i = 0
    depth_curly = 0
    depth_angle = 0
    in_string = False
    in_line_comment = False
    in_block_comment = False
    last_bar_pos = -1

    # Stack of brace-depth-at-which-each-score-was-opened
    score_open_at: list[int] = []
    has_layout = False
    has_midi = False

    while i < n:
        c = text[i]
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if c == "%" and i + 1 < n and text[i + 1] == "}":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_string:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == "%":
            if i + 1 < n and text[i + 1] == "{":
                in_block_comment = True
                i += 2
                continue
            in_line_comment = True
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "<" and i + 1 < n and text[i + 1] == "<":
            depth_angle += 1
            i += 2
            continue
        if c == ">" and i + 1 < n and text[i + 1] == ">":
            depth_angle -= 1
            i += 2
            continue
        if c == "|" and depth_curly > 0:
            last_bar_pos = i + 1
        if c == "{":
            depth_curly += 1
            i += 1
            continue
        if c == "}":
            # Close any score that was opened at this depth.
            if score_open_at and score_open_at[-1] == depth_curly:
                score_open_at.pop()
            depth_curly -= 1
            i += 1
            continue
        # Detect \score keyword
        if c == "\\":
            for kw, marker in (("score", "score"), ("layout", "layout"), ("midi", "midi")):
                klen = len(kw) + 1  # plus the backslash
                if (
                    i + klen <= n
                    and text[i + 1 : i + 1 + len(kw)] == kw
                    and (i + 1 + len(kw) >= n or not text[i + 1 + len(kw)].isalpha())
                ):
                    if marker == "score":
                        # find next `{`
                        j = i + 1 + len(kw)
                        while j < n and text[j] in " \t\r\n":
                            j += 1
                        if j < n and text[j] == "{":
                            # The `{` will be processed in the next iteration.
                            # Record that depth+1 will be a score-opening level.
                            score_open_at.append(depth_curly + 1)
                    elif marker == "layout" and score_open_at:
                        has_layout = True
                    elif marker == "midi" and score_open_at:
                        has_midi = True
                    break
        i += 1

    return {
        "depth_curly": depth_curly,
        "depth_angle": depth_angle,
        "last_bar_pos": last_bar_pos,
        "score_open": bool(score_open_at),
        "has_layout": has_layout,
        "has_midi": has_midi,
    }


def rescue_text(text: str) -> tuple[str, str]:
    """Return (rescued_text, status). status in {'unchanged', 'rescued', 'unrecoverable'}."""
    state = _scan(text)
    if state["depth_curly"] == 0 and state["depth_angle"] == 0:
        return text, "unchanged"

    # Cut at last bar separator inside music; if none, cut at the end.
    cut = state["last_bar_pos"] if state["last_bar_pos"] > 0 else len(text)
    head = text[:cut].rstrip()

    # Re-scan the cut head to get brace counts at the cut point.
    state2 = _scan(head)
    n_curly = state2["depth_curly"]
    n_angle = state2["depth_angle"]
    if n_curly < 0 or n_angle < 0:
        return text, "unrecoverable"

    suffix_parts: list[str] = []
    # Close angle brackets first (they sit inside curly).
    for _ in range(n_angle):
        suffix_parts.append(">>")
    # If inside a score block and missing \layout/\midi, inject them
    # before the score's closing brace so MIDI rendering succeeds.
    if state2["score_open"] and n_curly >= 1:
        if not state2["has_layout"]:
            suffix_parts.append("\\layout {}")
        if not state2["has_midi"]:
            suffix_parts.append("\\midi {}")
        # Close the score brace.
        suffix_parts.append("}")
        # Close any further outer curly braces.
        for _ in range(n_curly - 1):
            suffix_parts.append("}")
    else:
        for _ in range(n_curly):
            suffix_parts.append("}")

    rescued = head + "\n" + "\n".join(suffix_parts) + "\n"
    return rescued, "rescued"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.ly"))
    if not files:
        print(f"[rescue] no .ly files in {args.input_dir}")
        return 2

    counts = {"unchanged": 0, "rescued": 0, "unrecoverable": 0}
    for src in files:
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            counts["unrecoverable"] += 1
            continue
        rescued, status = rescue_text(text)
        counts[status] = counts.get(status, 0) + 1
        out = args.output_dir / src.name
        out.write_text(rescued, encoding="utf-8")

    print(
        f"[rescue] {len(files)} files: "
        f"unchanged={counts['unchanged']}, "
        f"rescued={counts['rescued']}, "
        f"unrecoverable={counts['unrecoverable']}"
    )
    print(f"[rescue] wrote outputs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
