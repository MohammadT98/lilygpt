from openai import OpenAI
from IPython.display import display, Markdown
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime

# ---- Optional deps (graceful if missing) ----
try:
    import muspy
except Exception:
    muspy = None

try:
    from music21 import converter, interval, analysis, note
except Exception:
    converter = interval = analysis = note = None


# ---------------- Helper functions ----------------

def find_lilypond():
    """Try to find LilyPond executable."""
    lilypond = shutil.which("lilypond")
    if lilypond:
        try:
            result = subprocess.run([lilypond, "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return lilypond
        except:
            pass

    env_path = os.environ.get("LILYPOND_BIN")
    if env_path and Path(env_path).exists():
        try:
            result = subprocess.run([env_path, "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return env_path
        except:
            pass

    default_path = r"C:\lilypond-2.24.4-mingw-x86_64\lilypond-2.24.4\bin\lilypond.exe"
    if Path(default_path).exists():
        try:
            result = subprocess.run([default_path, "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return default_path
        except:
            pass

    return None


def extract_lilypond_code(text: str) -> str:
    """Extract LilyPond code from text, removing markdown code blocks if present."""
    text = re.sub(r'```(?:lilypond)?\s*\n?(.*?)```', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'^(?:Here|Here\'s|Here is|Output|LilyPond code:|The following).*?:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


def _ensure_version_header(code: str) -> str:
    code = code.strip()
    return code if '\\version' in code else '\\version "2.24.4"\n' + code


def validate_lilypond(code: str, lilypond_path: str):
    """Validate if LilyPond code compiles. Returns (is_valid, error_message)."""
    if not code.strip():
        return False, "Empty code"
    code = _ensure_version_header(code)

    with tempfile.TemporaryDirectory() as tdir:
        tmp_file = Path(tdir) / "test.ly"
        tmp_file.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                [lilypond_path, "-dno-print-pages", "-dbackend=null",
                 "-o", str(Path(tdir) / "test"), str(tmp_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=12,
                encoding="utf-8"
            )
            if result.returncode == 0:
                return True, ""
            else:
                error = (result.stderr or result.stdout or "Unknown error").strip()
                error = re.sub(r'[A-Za-z]:[\\/][^\s:]+[\\/]([^:]+):', r'\1:', error)  # Windows paths
                error = re.sub(r'/[^\s:]+/([^:]+):', r'\1:', error)  # Unix paths
                error = re.sub(r'[A-Za-z]:[\\/][^\s:]+[\\/]', '', error)
                error = re.sub(r'/[^\s:]+/', '', error)
                error_lines = [l for l in error.split('\n') if 'error' in l.lower()]
                if error_lines:
                    return False, error_lines[0][:200]
                lines = [l.strip() for l in error.split('\n') if l.strip()]
                return False, lines[0][:200] if lines else "Compilation failed"
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)[:150]


def render_lilypond_to_midi(code: str, lilypond_path: str) -> str | None:
    """
    Render LilyPond code to MIDI in a temp dir and return the MIDI path (or None if fails).
    We disable PDF by using -dbackend=null and request only MIDI via --formats=midi.
    """
    code = _ensure_version_header(code)
    tdir = tempfile.TemporaryDirectory()
    tpath = Path(tdir.name)
    ly_file = tpath / "score.ly"
    ly_file.write_text(code, encoding="utf-8")

    try:
        # Note: --formats=midi ensures a .midi or .mid is produced without PDF.
        result = subprocess.run(
            [lilypond_path, "-dno-print-pages", "-dbackend=null", "--formats=midi",
             "-o", str(tpath / "score"), str(ly_file)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, encoding="utf-8"
        )
        if result.returncode != 0:
            return None

        midi_path = tpath / "score.midi"
        if not midi_path.exists():
            midi_path = tpath / "score.mid"
        if midi_path.exists():
            return str(midi_path)
        return None
    except Exception:
        return None


def music21_metrics(midi_path: str) -> dict | None:
    """Compute theory-style metrics with music21 (key, diatonicity, step/leap)."""
    if converter is None:
        return None

    try:
        s = converter.parse(midi_path)
        k = s.analyze('key')  # detected key
        scale_pcs = {p.pitchClass for p in k.getPitches()}

        total = 0
        in_key = 0
        notes_seq = []
        for n in s.recurse().notes:
            if isinstance(n, note.Note):
                total += 1
                if n.pitch.pitchClass in scale_pcs:
                    in_key += 1
                notes_seq.append(n)

        diatonic_ratio = (in_key / total) if total else 1.0

        # Interval profile
        ivals = []
        last = None
        for n in notes_seq:
            if last is not None:
                iv = interval.Interval(noteStart=last, noteEnd=n)
                ivals.append(abs(iv.semitones))
            last = n

        step_ratio = (sum(1 for x in ivals if x in (1, 2)) / len(ivals)) if ivals else 1.0
        leap_sizes = [x for x in ivals if x > 2]
        avg_leap = (sum(leap_sizes) / len(leap_sizes)) if leap_sizes else 0.0

        # Ambitus (pitch range) from music21 (fallback if muspy missing)
        try:
            amb = s.analyze('ambitus')
            amb_semitones = amb.semitones
        except Exception:
            amb_semitones = None

        return {
            "m21_detected_key": f"{k.tonic.name} {k.mode}",
            "m21_diatonic_note_ratio": round(diatonic_ratio, 4),
            "m21_stepwise_ratio": round(step_ratio, 4),
            "m21_avg_leap_semitones": round(avg_leap, 3),
            "m21_pitch_range_semitones": amb_semitones
        }
    except Exception:
        return None


def muspy_metrics(midi_path: str) -> dict | None:
    """Compute quantitative metrics with muspy."""
    if muspy is None:
        return None
    try:
        m = muspy.read_midi(midi_path)
        if not m.tracks:
            return None
        # Choose densest track as melody if multiple
        if len(m.tracks) > 1:
            m.tracks.sort(key=lambda t: sum(n.duration for n in t.notes), reverse=True)
            m.tracks = [m.tracks[0]]

        metrics = {
            "mp_pitch_range": muspy.pitch_range(m),
            "mp_note_density": round(muspy.note_density(m), 4),
            "mp_pitch_entropy": round(muspy.pitch_entropy(m), 4),
            "mp_repetition": round(muspy.repetition(m), 4),
            "mp_n_pitches": muspy.n_pitches_used(m),
            "mp_avg_ioi_beats": round(muspy.average_inter_onset_interval(m), 4),
        }
        return metrics
    except Exception:
        return None


# ---------------- Main runner ----------------

def run_zero_shot_test(
    prompt_file: str | Path = "../configs/prompts/zero_shot.txt",
    model: str = "openai/gpt-oss-20b:fireworks-ai",
    num_runs: int = 10,
    max_tokens: int = 10000,
    temperature: float = 0.7
):
    prompt_file = Path(prompt_file)

    # LilyPond
    lilypond_path = find_lilypond()
    if lilypond_path:
        display(Markdown(f"✅ **LilyPond found:** `{lilypond_path}`"))
    else:
        display(Markdown("⚠️ **LilyPond not found** — validation and MIDI metrics will be skipped."))

    # Optional libs info
    if muspy is None:
        display(Markdown("⚠️ **muspy not installed** → `pip install muspy` to enable quantitative metrics."))
    if converter is None:
        display(Markdown("⚠️ **music21 not installed** → `pip install music21` to enable theory metrics."))

    # Initialize client
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        display(Markdown("**❌ Error:** `HF_TOKEN` environment variable not set!"))
        return

    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)

    # Load prompt
    if not prompt_file.exists():
        display(Markdown(f"**❌ Error:** Prompt file not found: `{prompt_file}`"))
        return

    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    display(Markdown(f"### 📝 Input Prompt\n```\n{prompt_text}\n```"))
    display(Markdown("---"))

    results = []
    total_start_time = datetime.now()

    # Aggregate metric accumulators
    agg_counts = {
        "content": 0,
        "valid": 0,
        "midi": 0,
        "muspy": 0,
        "music21": 0
    }
    agg_sums = {}  # metric_name -> float sum for averaging

    def _accumulate(metrics: dict):
        nonlocal agg_sums
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                agg_sums[k] = agg_sums.get(k, 0.0) + float(v)

    for run_num in range(1, num_runs + 1):
        display(Markdown(f"## 🔄 Run {run_num}/{num_runs}"))

        start_time = datetime.now()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = (datetime.now() - start_time).total_seconds()

        choice = resp.choices[0]
        text = choice.message.content if choice.message and choice.message.content else ""
        lilypond_code = extract_lilypond_code(text) if text else ""

        is_valid, validation_error = None, None
        if lilypond_code and lilypond_path:
            is_valid, validation_error = validate_lilypond(lilypond_code, lilypond_path)

        # Try MIDI render (only if we have lilypond and some code)
        midi_path = None
        if lilypond_code and lilypond_path and (is_valid is True):
            midi_path = render_lilypond_to_midi(lilypond_code, lilypond_path)

        # Metrics
        m21 = None
        mp = None
        if midi_path:
            agg_counts["midi"] += 1
            if converter is not None:
                m21 = music21_metrics(midi_path)
                if m21:
                    agg_counts["music21"] += 1
                    _accumulate(m21)
            if muspy is not None:
                mp = muspy_metrics(midi_path)
                if mp:
                    agg_counts["muspy"] += 1
                    _accumulate(mp)

        # Store result
        results.append({
            "run": run_num,
            "elapsed": elapsed,
            "tokens": resp.usage.total_tokens if resp.usage else 0,
            "has_content": bool(text),
            "finish_reason": choice.finish_reason,
            "is_valid_lilypond": is_valid,
            "validation_error": validation_error,
            "midi_rendered": bool(midi_path),
            "music21": m21,
            "muspy": mp,
        })

        # Display output block
        if not text:
            display(Markdown(
                f"**⚠️ No content returned.**  \n"
                f"- Finish reason: `{choice.finish_reason}`  \n"
                f"- Time: {elapsed:.2f}s"
            ))
        else:
            md = f"### 🎵 Model Output\n```lilypond\n{lilypond_code}\n```\n"
            md += f"⏱️ Time: {elapsed:.2f}s | Tokens: {resp.usage.total_tokens if resp.usage else 0}"
            if is_valid is not None:
                if is_valid:
                    md += " | ✅ **Valid LilyPond**"
                else:
                    md += f" | ❌ **Invalid**: {validation_error}"
            elif lilypond_code and not lilypond_path:
                md += " | ⚠️ *Not validated (LilyPond not found)*"
            if midi_path:
                md += " | 🎼 MIDI rendered"
            display(Markdown(md))

            # Metrics pretty print
            if midi_path and (m21 or mp):
                lines = []
                if m21:
                    lines.append(
                        f"**music21** → key: `{m21['m21_detected_key']}`, "
                        f"diatonic: **{m21['m21_diatonic_note_ratio']}**, "
                        f"stepwise: **{m21['m21_stepwise_ratio']}**, "
                        f"avg leap: **{m21['m21_avg_leap_semitones']}**"
                        + (f", range: **{m21['m21_pitch_range_semitones']}** st" if m21.get('m21_pitch_range_semitones') is not None else "")
                    )
                if mp:
                    lines.append(
                        f"**muspy** → range: **{mp['mp_pitch_range']}** st, "
                        f"density: **{mp['mp_note_density']}** notes/beat, "
                        f"entropy: **{mp['mp_pitch_entropy']}**, "
                        f"repetition: **{mp['mp_repetition']}**, "
                        f"pitches: **{mp['mp_n_pitches']}**, "
                        f"avg IOI: **{mp['mp_avg_ioi_beats']}** beats"
                    )
                display(Markdown("#### 📐 Metrics\n" + "<br>".join(lines)))
            elif lilypond_path and is_valid and not midi_path:
                display(Markdown("⚠️ MIDI render failed — metrics skipped."))

        display(Markdown("---"))

        # Aggregates
        if text:
            agg_counts["content"] += 1
        if is_valid:
            agg_counts["valid"] += 1

    # Summary
    total_elapsed = (datetime.now() - total_start_time).total_seconds()
    successful_runs = agg_counts["content"]
    valid_lilypond_runs = agg_counts["valid"]
    avg_time = sum(r["elapsed"] for r in results) / len(results)
    total_tokens = sum(r["tokens"] for r in results)

    summary_md = [
        f"## 📊 Summary ({num_runs} runs)",
        f"- **Responses with content:** {successful_runs}/{num_runs} ({successful_runs/num_runs*100:.1f}%)",
        f"- **Valid LilyPond code:** {valid_lilypond_runs}/{num_runs} ({valid_lilypond_runs/num_runs*100:.1f}%)" if find_lilypond() else "",
        f"- **MIDI rendered:** {agg_counts['midi']}/{num_runs}",
        f"- **music21 metrics computed:** {agg_counts['music21']}/{num_runs}",
        f"- **muspy metrics computed:** {agg_counts['muspy']}/{num_runs}",
        f"- **Total time:** {total_elapsed:.2f}s",
        f"- **Average time per run:** {avg_time:.2f}s",
        f"- **Total tokens:** {total_tokens}",
        f"- **Average tokens per run:** {total_tokens // num_runs if num_runs else 0}",
    ]

    # Averages over numeric metrics
    if agg_sums:
        summary_md.append("\n### 🔎 Averages (only over runs where metric was available)")
        keys_sorted = sorted(agg_sums.keys())
        avg_lines = []
        # Determine denominators per family
        denom_map = {
            "m21_": max(1, agg_counts["music21"]),
            "mp_": max(1, agg_counts["muspy"])
        }
        for k in keys_sorted:
            denom = denom_map["m21_"] if k.startswith("m21_") else denom_map["mp_"]
            avg_lines.append(f"- **{k}**: {round(agg_sums[k]/denom, 4)}")
        summary_md.extend(avg_lines)

    display(Markdown("\n".join(line for line in summary_md if line != "")))
    return results

# Example:
# results = run_zero_shot_test()
