from __future__ import annotations

import re
from pathlib import Path

from lilynorm.utils.options import NormOptions
from lilynorm.stages import normalization


def normalize_file(path: Path, opts: NormOptions, stats: dict[str, int] | None = None) -> list[str]:
    from lilynorm.stages.normalization.postprocessing import apply_postprocessing_fixes, remove_empty_variable_assignments
    from lilynorm.stages.normalization import forma_inline, structure_prepend

    # Stage 0: Resolve includes and split on forma blocks
    stage0_pieces = normalization.file_resolver.run(path, exclude_variabili=False, split_forma=True)

    normalized_pieces = []

    for piece in stage0_pieces:
        # Stage 1: Preparse (remove comments, clean whitespace)
        stage1 = normalization.preparse.run(piece, opts)
        if stats is not None:
            stats["line_removed"] += max(0, len(piece.splitlines()) - len(stage1.splitlines()))
            stats["block_removed"] += max(0, piece.count("{") - stage1.count("{"))

        # Stage 2: Expand syntax (transpositions, repeats, tuplets)
        stage2 = normalization.expand.run(stage1, opts)
        if stats is not None:
            def count_occurrences(text: str) -> dict[str, int]:
                return {
                    "vars": len(re.findall(r"(?m)^[A-Za-z_][\\w-]*\\s*=\\s*\\{", text)),
                    "transpose": text.count("\\transpose"),
                    "repeat": text.count("\\repeat"),
                    "tuplets": text.count("\\tuplet") + text.count("\\times"),
                }

            before = count_occurrences(stage1)
            after = count_occurrences(stage2)
            stats["vars_removed"] += max(0, before["vars"] - after["vars"])
            stats["transpose_removed"] += max(0, before["transpose"] - after["transpose"])
            stats["repeat_removed"] += max(0, before["repeat"] - after["repeat"])
            stats["tuplets_removed"] += max(0, before["tuplets"] - after["tuplets"])

        # Stage 3: Prepend global structure to music variables
        # Skip if \forma is referenced; Stage 4 will inline the full structure.
        if re.search(r"\\forma\\b", stage2):
            stage3 = stage2
        else:
            stage3 = structure_prepend.run(stage2, opts)

        # Stage 4: Inline forma into voices (remove parallel structure lane)
        stage4 = forma_inline.run(stage3, opts)

        # Stage 5: Strip engraving directives
        stage5 = normalization.engrave_strip.run(stage4, opts)

        if stats is not None:
            def count_engraving(text: str) -> dict[str, int]:
                return {
                    "overrides": text.count("\\override"),
                    "markups": text.count("\\markup"),
                    "marks": text.count("\\mark"),
                    "dynamics": sum(text.count(tok) for tok in ["\\pp", "\\p", "\\mp", "\\mf", "\\f", "\\ff", "\\fp", "\\sfz"]),
                    "hairpins": text.count("\\<") + text.count("\\>"),
                    "quotes": text.count("\\quote"),
                }

            before = count_engraving(stage2)
            after = count_engraving(stage3)
            stats["overrides_removed"] += max(0, before["overrides"] - after["overrides"])
            stats["markups_removed"] += max(0, before["markups"] - after["markups"])
            stats["marks_removed"] += max(0, before["marks"] - after["marks"])
            stats["dynamics_removed"] += max(0, before["dynamics"] - after["dynamics"])
            stats["hairpins_removed"] += max(0, before["hairpins"] - after["hairpins"])
            stats["quotes_removed"] += max(0, before["quotes"] - after["quotes"])

        # Stage 6: Postprocessing fixes for malformed patterns
        stage6 = apply_postprocessing_fixes(stage5)

        # Stage 7: Remove empty variable assignments
        stage7, _ = remove_empty_variable_assignments(stage6)

        normalized_pieces.append(stage7)

    return normalized_pieces
