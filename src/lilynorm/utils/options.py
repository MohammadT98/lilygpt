from dataclasses import dataclass

@dataclass
class NormOptions:
    keep_engraving: bool = False #False
    strip_scheme_blocks: bool = True #True
    strip_comments: bool = True #True
    normalize_whitespace: bool = True #True