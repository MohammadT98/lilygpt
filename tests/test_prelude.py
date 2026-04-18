"""Unit tests for prelude boundary detection and assignment extraction."""

from __future__ import annotations

from lilybench.stages.dataset.prelude import (
    find_version_language_header,
    identify_prelude,
    iter_toplevel_assignments,
)


def test_identify_prelude_splits_at_variabili_marker(preprocessed_ly: str) -> None:
    info = identify_prelude(preprocessed_ly)

    prelude = preprocessed_ly[: info.prelude_end]
    body = preprocessed_ly[info.prelude_end :]

    assert "% === END INCLUDE: variabili.ly ===" in prelude
    assert prelude.endswith("\n")
    assert body.lstrip().startswith("violinoI = {")


def test_identify_prelude_without_marker_falls_back_to_music_body() -> None:
    text = (
        '\\version "2.24.0"\n'
        '\\language "nederlands"\n'
        "tune = { c'4 d' e' f' }\n"
    )

    info = identify_prelude(text)

    assert text[: info.prelude_end].endswith("\"nederlands\"\n")
    assert text[info.prelude_end :].startswith("tune = {")


def test_iter_toplevel_assignments_finds_body_variables(preprocessed_ly: str) -> None:
    info = identify_prelude(preprocessed_ly)
    spans = list(iter_toplevel_assignments(preprocessed_ly, info.prelude_end))

    names = [preprocessed_ly[s:e].split("=", 1)[0].strip() for s, e in spans]
    assert names == ["violinoI", "violinoII"]
    for s, e in spans:
        assert preprocessed_ly[s:e].rstrip().endswith("}")


def test_prelude_vars_are_extracted_for_short_declarations(preprocessed_ly: str) -> None:
    info = identify_prelude(preprocessed_ly)
    names = {v.name for v in info.vars}

    assert {"su", "giu", "tr"}.issubset(names)
    for var in info.vars:
        if var.name in {"su", "giu", "tr"}:
            assert var.is_short
            assert not var.has_pitch_tokens


def test_find_version_language_header_returns_both_lines(preprocessed_ly: str) -> None:
    header = find_version_language_header(preprocessed_ly)

    assert '\\version "2.24.0"' in header
    assert '\\language "nederlands"' in header
    assert header.endswith("\n")


def test_find_version_language_header_handles_missing_lines() -> None:
    assert find_version_language_header("c'4 d' e' f'") == ""
    only_version = find_version_language_header('\\version "2.24.0"\nfoo = { c4 }\n')
    assert only_version == '\\version "2.24.0"\n'
