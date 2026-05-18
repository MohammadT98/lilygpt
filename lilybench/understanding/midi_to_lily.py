"""Subprocess wrapper around LilyPond's ``midi2ly`` tool.

LilyPond ships ``midi2ly`` alongside the main binary. We use it to convert
EMOPIA pop-piano MIDI clips into compilable LilyPond text for the emotion-
recognition benchmark task.

The wrapper is intentionally thin — no retry, no quality scoring. The caller
is responsible for surveying which conversions failed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def convert_midi_to_lily(
    *,
    midi_path: Path,
    out_path: Path,
    midi2ly_bin: str | Path = "midi2ly",
    timeout_s: int = 30,
) -> Path | None:
    """Run ``midi2ly`` on ``midi_path`` and write LilyPond to ``out_path``.

    Returns ``out_path`` on success and ``None`` on any failure:
        * source file missing,
        * non-zero exit code,
        * timeout,
        * subprocess succeeded but produced no output file.
    """
    midi_path = Path(midi_path)
    out_path = Path(out_path)
    if not midi_path.exists():
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(midi2ly_bin),
        "--quiet",
        f"--output={out_path}",
        str(midi_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    if not out_path.exists() or out_path.stat().st_size == 0:
        return None
    return out_path
