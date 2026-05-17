"""LilyPond compile rate: invoke the ``lilypond`` binary and check returncode.

The paper reports the fraction of generated ``.ly`` files that compile to
MIDI without errors. We shell out to the user's installed LilyPond binary
(``$LILYPOND_BIN`` env var or ``lilypond`` on ``PATH``) and record the
compile time + first stderr line for failed runs so debugging is easy.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lilybench.utils import find_lilypond


@dataclass(frozen=True)
class CompileResult:
    path: Path
    ok: bool
    seconds: float
    midi_path: Path | None
    error: str | None = None


def _wrap_with_version(text: str, *, version: str = "2.24.4") -> str:
    if "\\version" in text:
        return text
    return f'\\version "{version}"\n' + text


def compile_to_midi(
    ly_path: str | Path,
    *,
    midi_dir: str | Path | None = None,
    lilypond_bin: str | Path | None = None,
    timeout_s: int = 15,
) -> CompileResult:
    """Compile a single ``.ly`` file. Returns a :class:`CompileResult`.

    ``midi_dir`` controls where the produced MIDI is moved. When ``None``,
    no MIDI is retained — useful when only the compile rate is needed.
    """
    ly_path = Path(ly_path)
    binary = Path(lilypond_bin) if lilypond_bin else find_lilypond()
    if binary is None or not binary.exists():
        return CompileResult(ly_path, False, 0.0, None, "lilypond binary not found")

    text = ly_path.read_text(encoding="utf-8", errors="ignore")
    src = _wrap_with_version(text)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src_path = tmp / "src.ly"
        src_path.write_text(src, encoding="utf-8")
        out_base = tmp / "out"
        cmd = [
            str(binary),
            "-dno-point-and-click",
            "-o", str(out_base),
            str(src_path),
        ]
        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return CompileResult(ly_path, False, timeout_s, None, "lilypond timeout")
        elapsed = time.perf_counter() - start

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "lilypond failed").splitlines()
            return CompileResult(ly_path, False, elapsed, None, err[0][:300] if err else "lilypond failed")

        midi_candidates = list(tmp.glob("out*.midi")) + list(tmp.glob("out*.mid"))
        if not midi_candidates:
            return CompileResult(ly_path, False, elapsed, None, "no midi produced")

        midi_out: Path | None = None
        if midi_dir is not None:
            midi_dir = Path(midi_dir)
            midi_dir.mkdir(parents=True, exist_ok=True)
            midi_out = (midi_dir / f"{ly_path.stem}.mid")
            midi_candidates[0].replace(midi_out)
        return CompileResult(ly_path, True, elapsed, midi_out, None)


def compile_rate(
    ly_paths: Iterable[str | Path],
    *,
    midi_dir: str | Path | None = None,
    lilypond_bin: str | Path | None = None,
    timeout_s: int = 15,
) -> tuple[float, list[CompileResult]]:
    """Return ``(rate, results)`` over an iterable of ``.ly`` paths."""
    results = [
        compile_to_midi(p, midi_dir=midi_dir, lilypond_bin=lilypond_bin, timeout_s=timeout_s)
        for p in ly_paths
    ]
    if not results:
        return 0.0, results
    rate = sum(1 for r in results if r.ok) / len(results)
    return rate, results
