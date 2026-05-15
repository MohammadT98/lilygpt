"""Music-understanding benchmark suite (adapted from arXiv-2509.23350v1).

Eight LilyPond-input tasks on the Mutopia corpus: bar count, metadata QA,
bar sequencing, next-bar prediction, metadata prediction, music captioning,
composer recognition, genre recognition.

Submodules are imported lazily to keep `lilybench.evaluate` and `lilybench.data`
free of unrelated dependencies.
"""
