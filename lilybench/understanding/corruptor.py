"""Synthetic error injection for the error_detection task.

Five categories matching arXiv-2509.23350v1 §"Error Detection":

  1. invalid_metadata        — corrupt the ``\\key`` directive
  2. invalid_content         — garbage tokens inside a bar
  3. invalid_bar_duration    — total durations don't sum to the time signature
  4. melodic_leap            — single-step jump >10 scale degrees
  5. accidental_outside_key  — pitch not in the declared key signature

Each injector returns a ``Corruption`` with the modified text, the 1-indexed
bar number(s) carrying the injected error, and the category. Returns ``None``
when the injection isn't applicable to the given input — the bench builder
re-rolls a different injector / piece in that case.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from lilybench.understanding.bar_utils import count_bars


ERROR_CATEGORIES = (
    "invalid_metadata",
    "invalid_content",
    "invalid_bar_duration",
    "melodic_leap",
    "accidental_outside_key",
)


@dataclass(frozen=True)
class Corruption:
    text: str
    error_bars: tuple[int, ...]   # 1-indexed bar numbers
    category: str


# ----------- position helpers -----------

_HEADER_BLOCK_RE = re.compile(r"\\header\s*\{[^{}]*\}", re.DOTALL)
_PAPER_BLOCK_RE = re.compile(r"\\paper\s*\{[^{}]*\}", re.DOTALL)
_BAR_DIRECTIVE_RE = re.compile(r'\\bar\s*"[^"]*"')
_QUOTED_STRING_RE = re.compile(r'"[^"\n]*"')


def _mask_keep_positions(text: str) -> str:
    """Replace preamble blocks / quoted strings with spaces so character
    positions are preserved but their ``|`` are no longer counted."""
    def _pad(m: re.Match) -> str:
        return " " * len(m.group(0))
    text = _HEADER_BLOCK_RE.sub(_pad, text)
    text = _PAPER_BLOCK_RE.sub(_pad, text)
    text = _BAR_DIRECTIVE_RE.sub(_pad, text)
    text = _QUOTED_STRING_RE.sub(_pad, text)
    return text


def _bar_separator_positions(text: str) -> list[int]:
    """Character positions of every valid bar separator in ``text``."""
    masked = _mask_keep_positions(text)
    return [i for i, c in enumerate(masked) if c == "|"]


def bar_at_position(text: str, pos: int) -> int:
    """1-indexed bar number that contains character position ``pos``."""
    bars_before = count_bars(text[:pos])
    return bars_before + 1


# ----------- category 1: invalid_metadata -----------

_KEY_FULL_RE = re.compile(
    r"\\key\s+(?:[a-g]|do|re|mi|fa|sol|la|si)(?:isis|eses|is|es|dd|d|bb|b)?\s+\\[A-Za-z]+"
)


def inject_invalid_metadata(text: str, rng: random.Random) -> Corruption | None:
    """Replace the first ``\\key`` directive with a nonsense one."""
    if not _KEY_FULL_RE.search(text):
        return None
    new_text = _KEY_FULL_RE.sub(r"\\key xx \\nonsense", text, count=1)
    return Corruption(text=new_text, error_bars=(1,), category="invalid_metadata")


# ----------- category 2: invalid_content -----------

_GARBAGE_TOKENS = (" @#$% ", " ??? ", " %%%bogus%%% ", " <<garbled>> ")


def inject_invalid_content(text: str, rng: random.Random) -> Corruption | None:
    """Insert non-LilyPond garbage tokens into a randomly chosen bar."""
    positions = _bar_separator_positions(text)
    if len(positions) < 2:
        return None
    # Pick a bar index in [1, N] (1-indexed). For bar k, insert just before
    # the k-th separator (so the garbage lives inside that bar).
    n = len(positions)
    bar_idx = rng.randrange(1, n + 1)
    insert_at = positions[bar_idx - 1]
    garbage = rng.choice(_GARBAGE_TOKENS)
    new_text = text[:insert_at] + garbage + text[insert_at:]
    return Corruption(text=new_text, error_bars=(bar_idx,), category="invalid_content")


# ----------- category 3: invalid_bar_duration -----------

_TIME_RE = re.compile(r"\\time\s+(\d+)\s*/\s*(\d+)")
# A "fragment" we can insert that obviously doesn't fit any sane meter:
# eight quarter notes (= 2 whole notes' worth) in a single bar.
_DURATION_GARBAGE = " c4 d4 e4 f4 g4 a4 b4 c'4 "


def inject_invalid_bar_duration(text: str, rng: random.Random) -> Corruption | None:
    """Insert a bar with the wrong total duration.

    Strategy: pick a bar, insert eight quarter notes inside it. Combined with
    the original content, the bar's total duration overflows any standard
    meter (4/4, 6/8, 3/4 ...). We rely on the bar boundary `|` staying in
    place, so the model still sees the bar but with too much music in it.
    """
    if not _TIME_RE.search(text):
        return None  # can't reason about duration without a meter
    positions = _bar_separator_positions(text)
    if len(positions) < 2:
        return None
    n = len(positions)
    bar_idx = rng.randrange(1, n + 1)
    insert_at = positions[bar_idx - 1]
    new_text = text[:insert_at] + _DURATION_GARBAGE + text[insert_at:]
    return Corruption(
        text=new_text, error_bars=(bar_idx,), category="invalid_bar_duration"
    )


# ----------- category 4: melodic_leap -----------

# A single jump from the very low to the very high register (10+ scale degrees).
_LEAP_TOKENS = (" c,,,4 c''''4 ", " c''''4 c,,,4 ", " c,,,8 c''''8 ")


def inject_melodic_leap(text: str, rng: random.Random) -> Corruption | None:
    """Insert two consecutive notes with a >10-degree leap into a bar."""
    positions = _bar_separator_positions(text)
    if len(positions) < 2:
        return None
    n = len(positions)
    bar_idx = rng.randrange(1, n + 1)
    insert_at = positions[bar_idx - 1]
    leap = rng.choice(_LEAP_TOKENS)
    new_text = text[:insert_at] + leap + text[insert_at:]
    return Corruption(text=new_text, error_bars=(bar_idx,), category="melodic_leap")


# ----------- category 5: accidental_outside_key -----------

# Notes that are sharps in nederlands LilyPond syntax. We pick one outside the
# declared key signature. For simplicity we focus on the two most common keys:
# C major (no sharps/flats) and G major (one sharp: F#). For others we fall back
# to a chromatic note that's almost always out-of-key.
_KEY_SIGS = {
    "c \\major": {"in": set(), "out": ["fis", "cis", "gis", "dis", "ais"]},
    "a \\minor": {"in": set(), "out": ["fis", "cis", "gis", "dis", "ais"]},
    "g \\major": {"in": {"fis"}, "out": ["cis", "gis", "dis", "ais"]},
    "f \\major": {"in": {"bes"}, "out": ["fis", "cis", "gis", "dis"]},
    "d \\major": {"in": {"fis", "cis"}, "out": ["gis", "dis", "ais", "bes"]},
}


def _normalize_key(text: str) -> str | None:
    m = re.search(
        r"\\key\s+([a-g](?:isis|eses|is|es|dd|d|bb|b)?)\s+\\(\w+)", text
    )
    if not m:
        return None
    return f"{m.group(1)} \\{m.group(2)}"


def inject_accidental_outside_key(text: str, rng: random.Random) -> Corruption | None:
    """Insert an accidental note not in the declared key signature."""
    key = _normalize_key(text)
    if key is None or key not in _KEY_SIGS:
        # Fall back to a near-universally-out-of-key chromatic: B-double-sharp.
        candidates = ["bisis", "feses"]
    else:
        candidates = _KEY_SIGS[key]["out"]
    if not candidates:
        return None
    positions = _bar_separator_positions(text)
    if len(positions) < 2:
        return None
    n = len(positions)
    bar_idx = rng.randrange(1, n + 1)
    insert_at = positions[bar_idx - 1]
    pitch = rng.choice(candidates)
    new_text = text[:insert_at] + f" {pitch}4 " + text[insert_at:]
    return Corruption(
        text=new_text, error_bars=(bar_idx,), category="accidental_outside_key"
    )


# ----------- public dispatch -----------

_INJECTORS = {
    "invalid_metadata": inject_invalid_metadata,
    "invalid_content": inject_invalid_content,
    "invalid_bar_duration": inject_invalid_bar_duration,
    "melodic_leap": inject_melodic_leap,
    "accidental_outside_key": inject_accidental_outside_key,
}


def inject(text: str, category: str, *, rng: random.Random) -> Corruption | None:
    """Dispatch to the injector for ``category``."""
    if category not in _INJECTORS:
        raise KeyError(
            f"unknown category {category!r}; expected one of {ERROR_CATEGORIES}"
        )
    return _INJECTORS[category](text, rng)
