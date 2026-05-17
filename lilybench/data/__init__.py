"""Corpus loaders used by LilyBench.

Each loader reads a manifest or directory layout and returns a uniform
``CorpusEntry`` so the bench builders and metric runners stay backbone-
agnostic. Datasets are released alongside the paper on Zenodo; see the
README for download and layout instructions.
"""

from lilybench.data.types import CorpusEntry
from lilybench.data.bmdataset import load_bmdataset
from lilybench.data.mutopia import load_mutopia
from lilybench.data.emopia import load_emopia
from lilybench.data.splits import split_by_work

__all__ = [
    "CorpusEntry",
    "load_bmdataset",
    "load_mutopia",
    "load_emopia",
    "split_by_work",
]
