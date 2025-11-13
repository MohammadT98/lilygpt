# D:\uniPadova\Thesis\lilynorm\src\lilynorm\utils\options.py
from dataclasses import dataclass

@dataclass
class NormOptions:
    keep_engraving: bool = False
    strip_scheme_blocks: bool = True
    strip_comments: bool = True
    normalize_whitespace: bool = True
    expand_relative: bool = False
