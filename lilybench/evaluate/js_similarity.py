"""JS Divergence Similarity vs. a reference distribution.

Ported verbatim from mxlGPT (mxlGPT/src/evaluation/generation/muspy_eval.py),
adapted to use our `muspy_`-prefixed metric keys and to load the reference
aggregate from a cache file (mirrors fmd.py's reference_embeddings_path pattern).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from lilybench.evaluate.muspy_metrics import compute_muspy_metrics

log = logging.getLogger(__name__)


JS_METRICS: tuple[str, ...] = (
    "muspy_polyphony_rate",
    "muspy_groove_consistency",
    "muspy_scale_consistency",
)

_MIDI_SUFFIXES = {".mid", ".midi"}


def aggregate(per_file: dict[str, dict[str, float | None]]) -> dict[str, dict]:
    """Compute mean, std (unbiased), and n for each metric across files.

    Files that returned None for a metric are excluded from that metric's
    statistics. Returns ``{metric: {"mean": float|None, "std": float|None, "n": int}}``.
    """
    if not per_file:
        return {}

    all_metrics = list(next(iter(per_file.values())).keys())
    agg: dict[str, dict] = {}

    for metric in all_metrics:
        values = [v[metric] for v in per_file.values() if v.get(metric) is not None]
        n = len(values)
        if n == 0:
            agg[metric] = {"mean": None, "std": None, "n": 0}
        elif n == 1:
            agg[metric] = {"mean": float(values[0]), "std": None, "n": 1}
        else:
            mean = sum(values) / n
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            agg[metric] = {"mean": float(mean), "std": math.sqrt(variance), "n": n}

    return agg


def _js_divergence_gaussian(
    mu1: float, sigma1: float, mu2: float, sigma2: float, n_points: int = 2000
) -> float:
    """Numerically compute JS divergence between N(mu1,sigma1²) and N(mu2,sigma2²)."""
    lo = min(mu1 - 5 * sigma1, mu2 - 5 * sigma2)
    hi = max(mu1 + 5 * sigma1, mu2 + 5 * sigma2)
    x = np.linspace(lo, hi, n_points)
    dx = x[1] - x[0]

    p = scipy_stats.norm.pdf(x, mu1, sigma1)
    q = scipy_stats.norm.pdf(x, mu2, sigma2)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])) * dx)

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compute_js_similarity(
    model_agg: dict[str, dict],
    ref_agg: dict[str, dict],
    metrics: tuple[str, ...] = JS_METRICS,
) -> float | None:
    """Return JS Divergence Similarity = 100 * exp(-2 * mean_JS) across the metrics.

    Returns None if any metric lacks sufficient statistics in either distribution.
    """
    js_values: list[float] = []
    for metric in metrics:
        m_stats = model_agg.get(metric, {})
        r_stats = ref_agg.get(metric, {})

        mu1, s1 = m_stats.get("mean"), m_stats.get("std")
        mu2, s2 = r_stats.get("mean"), r_stats.get("std")

        if any(v is None for v in (mu1, s1, mu2, s2)):
            log.warning("Skipping JS for '%s': missing mean/std in one distribution.", metric)
            return None
        if s1 <= 0 or s2 <= 0:
            log.warning("Skipping JS for '%s': zero std.", metric)
            return None

        js_values.append(_js_divergence_gaussian(mu1, s1, mu2, s2))

    mean_js = sum(js_values) / len(js_values)
    return 100.0 * math.exp(-2.0 * mean_js)


def load_reference_aggregate(
    reference_midi_dir: Path | str | None,
    reference_aggregate_path: Path | str | None,
) -> dict[str, dict] | None:
    """Load-or-compute reference aggregate stats for JS similarity.

    Resolution order (mirrors fmd.py's reference_embeddings_path pattern):
      1. If ``reference_aggregate_path`` exists, load it and return.
      2. Else if ``reference_midi_dir`` is set, walk it for MIDIs, compute
         per-file muspy metrics, aggregate, optionally cache, and return.
      3. Else return None.
    """
    cache_path = Path(reference_aggregate_path).expanduser().resolve() if reference_aggregate_path else None
    midi_dir = Path(reference_midi_dir).expanduser().resolve() if reference_midi_dir else None

    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if midi_dir is None:
        return None

    if not midi_dir.exists():
        log.warning("Reference MIDI dir does not exist: %s", midi_dir)
        return None

    per_file: dict[str, dict[str, float | None]] = {}
    for path in sorted(midi_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _MIDI_SUFFIXES:
            continue
        per_file[str(path)] = compute_muspy_metrics(path)

    if not per_file:
        log.warning("No MIDI files found under %s", midi_dir)
        return None

    agg = aggregate(per_file)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    return agg
