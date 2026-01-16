from __future__ import annotations

"""Normalization pipeline orchestration."""

import re
from pathlib import Path

from lilynorm.stages import normalization
from lilynorm.utils.options import NormOptions


def _split_inline_assignments(text: str) -> str:
    """Insert blank lines between consecutive inline assignments."""
    text = re.sub(r"}\s+(?=[A-Za-z_][\w-]*\s*=)", "}\n\n", text)
    text = re.sub(r"\)\s+(?=[A-Za-z_][\w-]*\s*=)", ")\n\n", text)
    return text


def _count_syntax_features(text: str) -> dict[str, int]:
    return {
        "vars": len(re.findall(r"(?m)^[A-Za-z_][\\w-]*\\s*=\\s*\\{", text)),
        "transpose": text.count("\\transpose"),
        "repeat": text.count("\\repeat"),
        "tuplets": text.count("\\tuplet") + text.count("\\times"),
    }


def _count_engraving_features(text: str) -> dict[str, int]:
    return {
        "overrides": text.count("\\override"),
        "markups": text.count("\\markup"),
        "marks": text.count("\\mark"),
        "dynamics": sum(
            text.count(token)
            for token in ("\\pp", "\\p", "\\mp", "\\mf", "\\f", "\\ff", "\\fp", "\\sfz")
        ),
        "hairpins": text.count("\\<") + text.count("\\>"),
        "quotes": text.count("\\quote"),
    }


def normalize_file(
    path: Path,
    opts: NormOptions,
    stats: dict[str, int] | None = None,
) -> list[str]:
    """Normalize a LilyPond source into cleaned training-ready blocks."""
    from lilynorm.stages.normalization import forma
    from lilynorm.stages.normalization.postprocessing import (
        apply_postprocessing_fixes,
        remove_empty_variable_assignments,
    )

    stage0_pieces = normalization.file_resolver.run(
        path,
        exclude_variabili=False,
        split_forma=True,
    )

    normalized_pieces: list[str] = []

    for piece in stage0_pieces:
        stage1 = normalization.preprocess.run(piece, opts)
        if stats is not None:
            stats["line_removed"] += max(
                0, len(piece.splitlines()) - len(stage1.splitlines())
            )
            stats["block_removed"] += max(0, piece.count("{") - stage1.count("{"))

        stage2 = normalization.normalize_syntax.run(stage1, opts)
        stage2 = _split_inline_assignments(stage2)
        if stats is not None:
            before = _count_syntax_features(stage1)
            after = _count_syntax_features(stage2)
            stats["vars_removed"] += max(0, before["vars"] - after["vars"])
            stats["transpose_removed"] += max(
                0, before["transpose"] - after["transpose"]
            )
            stats["repeat_removed"] += max(0, before["repeat"] - after["repeat"])
            stats["tuplets_removed"] += max(0, before["tuplets"] - after["tuplets"])

        stage3 = forma.prepend_structure(stage2, opts)
        stage4 = forma.inline_forma(stage3, opts)

        stage5 = normalization.engrave_strip.run(stage4, opts)
        if stats is not None:
            before = _count_engraving_features(stage2)
            after = _count_engraving_features(stage5)
            stats["overrides_removed"] += max(
                0, before["overrides"] - after["overrides"]
            )
            stats["markups_removed"] += max(
                0, before["markups"] - after["markups"]
            )
            stats["marks_removed"] += max(0, before["marks"] - after["marks"])
            stats["dynamics_removed"] += max(
                0, before["dynamics"] - after["dynamics"]
            )
            stats["hairpins_removed"] += max(
                0, before["hairpins"] - after["hairpins"]
            )
            stats["quotes_removed"] += max(0, before["quotes"] - after["quotes"])

        stage6 = apply_postprocessing_fixes(stage5)
        stage7, empty_removed = remove_empty_variable_assignments(stage6)
        if stats is not None:
            stats["vars_removed"] += empty_removed

        normalized_pieces.append(stage7)

    return normalized_pieces
