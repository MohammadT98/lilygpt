"""LilyBench — an evaluation framework for LLMs on LilyPond.

LilyBench is the companion code for the Ital-IA 2026 paper "Can LLMs
understand LilyPond? A benchmark for symbolic music generation and
understanding". It pairs a generation benchmark (zero-shot / few-shot
LilyPond synthesis evaluated with compile rate, JS-similarity over MusPy
descriptors, and a LilyBERT-based Fréchet Music Distance) with a ten-task
understanding suite adapted from ABC-Eval.

Top-level subpackages:

* :mod:`lilybench.models` — model registry and loading helpers.
* :mod:`lilybench.data` — corpus loaders (BMdataset, Mutopia, EMOPIA).
* :mod:`lilybench.generation` — prompt-bank construction, regimes, runner.
* :mod:`lilybench.understanding` — extensible task registry and runner.
* :mod:`lilybench.metrics` — compile rate, JS similarity, FMD, and the
  per-task scorers for understanding.

Both ``benchmarks`` are modular: new generation regimes plug in via
:class:`lilybench.generation.regimes.Regime`, new understanding tasks via
:func:`lilybench.understanding.register_task`.
"""

__version__ = "0.2.0"
