"""Evaluation entry points.

Three Hydra-driven entry points, one per metric family:

* :mod:`lilybench.evaluate.loss` \u2014 held-out cross-entropy on a JSONL split.
* :mod:`lilybench.evaluate.text_midi` \u2014 per-sample LilyPond text + MIDI checks.
* :mod:`lilybench.evaluate.fmd` \u2014 Fr\u00e9chet Music Distance against a reference.

Plus :mod:`lilybench.evaluate.extract_detokenized`, an argparse helper that
extracts ``Detokenized Output`` blocks from SLURM inference logs.
"""
