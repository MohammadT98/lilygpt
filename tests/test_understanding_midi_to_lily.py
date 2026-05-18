"""Tests for the midi2ly subprocess wrapper.

Two layers:
  * pure-Python tests using ``monkeypatch`` on ``subprocess.run`` — always run.
  * happy-path integration test that actually invokes ``midi2ly`` — skipped
    when neither ``mido`` (to build the synthetic MIDI) nor the ``midi2ly``
    binary is available on ``$PATH``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from lilybench.understanding.midi_to_lily import convert_midi_to_lily


def test_returns_none_when_source_missing(tmp_path: Path):
    result = convert_midi_to_lily(
        midi_path=tmp_path / "does-not-exist.mid",
        out_path=tmp_path / "out.ly",
    )
    assert result is None


def test_returns_none_on_nonzero_exit(tmp_path: Path, monkeypatch):
    src = tmp_path / "fake.mid"
    src.write_bytes(b"not-a-midi")
    out = tmp_path / "out.ly"

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout=b"", stderr=b"boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = convert_midi_to_lily(
        midi_path=src, out_path=out, midi2ly_bin="midi2ly"
    )
    assert result is None


def test_returns_none_on_timeout(tmp_path: Path, monkeypatch):
    src = tmp_path / "fake.mid"
    src.write_bytes(b"not-a-midi")
    out = tmp_path / "out.ly"

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = convert_midi_to_lily(midi_path=src, out_path=out, timeout_s=1)
    assert result is None


def test_returns_none_when_output_not_written(tmp_path: Path, monkeypatch):
    """Subprocess succeeds but no file was created — treat as failure."""
    src = tmp_path / "fake.mid"
    src.write_bytes(b"not-a-midi")
    out = tmp_path / "out.ly"

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = convert_midi_to_lily(midi_path=src, out_path=out)
    assert result is None


def test_returns_path_when_output_present(tmp_path: Path, monkeypatch):
    src = tmp_path / "fake.mid"
    src.write_bytes(b"not-a-midi")
    out = tmp_path / "out.ly"

    def fake_run(cmd, **kwargs):
        out.write_text('\\version "2.24.0"\n{ c4 d4 e4 f4 | }\n', encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = convert_midi_to_lily(midi_path=src, out_path=out)
    assert result == out
    assert out.exists()
    assert "\\version" in out.read_text(encoding="utf-8")


def test_passes_correct_args_to_midi2ly(tmp_path: Path, monkeypatch):
    src = tmp_path / "fake.mid"
    src.write_bytes(b"not-a-midi")
    out = tmp_path / "sub" / "out.ly"
    captured: list = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ok", encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    convert_midi_to_lily(
        midi_path=src,
        out_path=out,
        midi2ly_bin="/opt/lilypond/bin/midi2ly",
    )
    assert captured
    cmd = captured[0]
    assert cmd[0] == "/opt/lilypond/bin/midi2ly"
    assert str(src) in cmd
    # Output path should be communicated to midi2ly somehow (either via -o or
    # by reading captured stdout). Accept either form for forward-compat.
    joined = " ".join(str(c) for c in cmd)
    assert str(out) in joined or "--output" in joined or "-o" in joined


# --- happy-path integration ----------------------------------------------

_MIDI2LY = shutil.which("midi2ly")


@pytest.mark.skipif(_MIDI2LY is None, reason="midi2ly binary not on $PATH")
def test_happy_path_with_real_midi2ly(tmp_path: Path):
    mido = pytest.importorskip("mido")
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    # Four bars of quarter notes (C major scale-ish).
    for note in (60, 62, 64, 65, 67, 69, 71, 72):
        track.append(mido.Message("note_on", note=note, velocity=80, time=0))
        track.append(mido.Message("note_off", note=note, velocity=0, time=480))
    midi_path = tmp_path / "scale.mid"
    mid.save(midi_path)

    out_path = tmp_path / "scale.ly"
    result = convert_midi_to_lily(midi_path=midi_path, out_path=out_path)

    assert result == out_path
    text = out_path.read_text(encoding="utf-8")
    assert "\\version" in text
    assert "|" in text  # at least one bar separator
