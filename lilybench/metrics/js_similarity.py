"""JS-similarity over the three MusPy descriptors.

The paper reports

    JS-similarity = 100 · exp(-2 · mean(JS divergence))

where the JS divergence is computed numerically between two univariate
Gaussians fit to the model and reference distributions of each
descriptor. The metric depends only on the per-file descriptor values:
two callers can compute the reference aggregate once (cached on disk)
and reuse it across model evaluations.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import stats as scipy_stats

from lilybench.metrics.muspy_descriptors import DESCRIPTOR_KEYS, compute_muspy_descriptors


log = logging.getLogger(__name__)

JS_METRICS: tuple[str, ...] = DESCRIPTOR_KEYS


def aggregate_descriptor_stats(
    per_file: Mapping[str, Mapping[str, float | None]],
) -> dict[str, dict]:
    """Compute ``{metric: {mean, std, n}}`` over the descriptor values."""
    if not per_file:
        return {}
    out: dict[str, dict] = {}
    keys = list(next(iter(per_file.values())).keys())
    for metric in keys:
        values = [
            v[metric]
            for v in per_file.values()
            if v.get(metric) is not None and isinstance(v[metric], (int, float)) and not isinstance(v[metric], bool)
        ]
        n = len(values)
        if n == 0:
            out[metric] = {"mean": None, "std": None, "n": 0}
        elif n == 1:
            out[metric] = {"mean": float(values[0]), "std": None, "n": 1}
        else:
            mean = sum(values) / n
            var = sum((x - mean) ** 2 for x in values) / (n - 1)
            out[metric] = {"mean": float(mean), "std": math.sqrt(var), "n": n}
    return out


def _js_divergence_gaussian(
    mu1: float, sigma1: float, mu2: float, sigma2: float, n_points: int = 2000
) -> float:
    lo = min(mu1 - 5 * sigma1, mu2 - 5 * sigma2)
    hi = max(mu1 + 5 * sigma1, mu2 + 5 * sigma2)
    x = np.linspace(lo, hi, n_points)
    dx = x[1] - x[0]
    p = scipy_stats.norm.pdf(x, mu1, sigma1)
    q = scipy_stats.norm.pdf(x, mu2, sigma2)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])) * dx)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def js_descriptor_similarity(
    model_agg: Mapping[str, Mapping[str, float | None]],
    ref_agg: Mapping[str, Mapping[str, float | None]],
    metrics: tuple[str, ...] = JS_METRICS,
) -> float | None:
    """Return ``100 * exp(-2 * mean_JS)`` over the descriptor distributions."""
    js_vals: list[float] = []
    for metric in metrics:
        m_stats = model_agg.get(metric, {})
        r_stats = ref_agg.get(metric, {})
        mu1, s1 = m_stats.get("mean"), m_stats.get("std")
        mu2, s2 = r_stats.get("mean"), r_stats.get("std")
        if any(v is None for v in (mu1, s1, mu2, s2)):
            log.warning("JS skipped for %r: missing mean/std", metric)
            return None
        if s1 <= 0 or s2 <= 0:
            log.warning("JS skipped for %r: zero std", metric)
            return None
        js_vals.append(_js_divergence_gaussian(mu1, s1, mu2, s2))
    return 100.0 * math.exp(-2.0 * (sum(js_vals) / len(js_vals)))


def load_or_build_reference(
    cache_path: str | Path | None = None,
    midi_dir: str | Path | None = None,
) -> dict[str, dict] | None:
    """Load the cached reference aggregate or build it from MIDIs under ``midi_dir``.

    When ``cache_path`` exists it is loaded directly; otherwise descriptors
    are computed over every ``.mid`` / ``.midi`` file under ``midi_dir``
    and (optionally) written to ``cache_path`` for reuse.
    """
    cache_path = Path(cache_path).expanduser().resolve() if cache_path else None
    midi_dir = Path(midi_dir).expanduser().resolve() if midi_dir else None
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if midi_dir is None or not midi_dir.exists():
        return None
    per_file: dict[str, dict] = {}
    for p in sorted(midi_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".mid", ".midi"}:
            per_file[str(p)] = compute_muspy_descriptors(p)
    if not per_file:
        return None
    agg = aggregate_descriptor_stats(per_file)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return agg
