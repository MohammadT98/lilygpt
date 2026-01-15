from dataclasses import dataclass

@dataclass
class NormOptions:
    # Preparse options
    strip_scheme_blocks: bool = True
    strip_comments: bool = True
    normalize_whitespace: bool = True

    # Expansion options (LilyPond normalize/expand)
    expand_music_functions: bool = True
    resolve_transpose: bool = True
    expand_repeat_unfold: bool = True
    normalize_tuplets: bool = True
    preserve_linebreaks: bool = True
    canonicalize_chord_brackets: bool = True

    # Engraving/cleanup options
    keep_engraving: bool = False
