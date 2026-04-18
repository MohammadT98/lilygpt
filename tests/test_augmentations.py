"""Unit tests for anti-memorization augmentations and the brace-balance gate."""

from __future__ import annotations

import random

import pytest

from lilybench.stages.dataset.augmentations import (
    check_brace_balance,
    drop_unused_prelude,
    inline_variables,
    shuffle_prelude,
)
from lilybench.stages.dataset.prelude import identify_prelude


class TestCheckBraceBalance:
    def test_balanced_simple_block(self) -> None:
        assert check_brace_balance("foo = { c4 d4 e4 }")

    def test_unbalanced_missing_close(self) -> None:
        assert not check_brace_balance("foo = { c4 d4 e4 ")

    def test_unbalanced_extra_close(self) -> None:
        assert not check_brace_balance("foo = { c4 } }")

    def test_braces_inside_strings_are_ignored(self) -> None:
        assert check_brace_balance('title = "a { b" foo = { c4 }')

    def test_braces_inside_line_comments_are_ignored(self) -> None:
        assert check_brace_balance("% stray { brace\nfoo = { c4 }\n")

    def test_braces_inside_block_comments_are_ignored(self) -> None:
        assert check_brace_balance("%{ stray } braces %}\nfoo = { c4 }\n")

    def test_unterminated_block_comment_fails(self) -> None:
        assert not check_brace_balance("%{ opened but never closed\nfoo = { c4 }\n")

    def test_scheme_parens_are_not_checked(self) -> None:
        assert check_brace_balance("foo = #(lambda (x) '(a b c)) bar = { c4 }")


def test_shuffle_prelude_is_deterministic_given_same_seed(preprocessed_ly: str) -> None:
    info = identify_prelude(preprocessed_ly)

    first = shuffle_prelude(preprocessed_ly, info, random.Random(7))
    second = shuffle_prelude(preprocessed_ly, info, random.Random(7))

    assert first == second


def test_shuffle_prelude_preserves_body_and_brace_balance(preprocessed_ly: str) -> None:
    info = identify_prelude(preprocessed_ly)
    out = shuffle_prelude(preprocessed_ly, info, random.Random(3))

    info_out = identify_prelude(out)
    assert out[info_out.prelude_end :] == preprocessed_ly[info.prelude_end :]
    assert check_brace_balance(out)
    assert {v.name for v in info_out.vars} == {v.name for v in info.vars}


def test_shuffle_prelude_no_op_when_fewer_than_two_shuffleable() -> None:
    text = (
        '\\version "2.24.0"\n'
        '\\language "nederlands"\n'
        "% === END INCLUDE: variabili.ly ===\n"
        "tune = { c'4 d' }\n"
    )
    info = identify_prelude(text)
    assert shuffle_prelude(text, info, random.Random(0)) == text


def test_drop_unused_prelude_removes_only_unreferenced(preprocessed_ly: str) -> None:
    body_refs_su = preprocessed_ly.replace(
        "violinoI = {\n  \\key g \\major\n  c'4 d' e' f' |\n",
        "violinoI = {\n  \\key g \\major\n  c'4 d'^\\su e' f' |\n",
        1,
    )
    info = identify_prelude(body_refs_su)

    dropped = drop_unused_prelude(body_refs_su, info, random.Random(0), p=1.0)

    assert "su = \\markup" in dropped
    assert "giu = \\markup" not in dropped
    assert "tr = \\markup" not in dropped


def test_drop_unused_prelude_p_zero_is_identity(preprocessed_ly: str) -> None:
    info = identify_prelude(preprocessed_ly)
    assert drop_unused_prelude(preprocessed_ly, info, random.Random(0), p=0.0) == preprocessed_ly


def test_inline_variables_replaces_call_sites_and_deletes_decl() -> None:
    text = (
        '\\version "2.24.0"\n'
        '\\language "nederlands"\n'
        "% === BEGIN INCLUDE: variabili.ly ===\n"
        "tr = \\markup { \\italic tr }\n"
        "% === END INCLUDE: variabili.ly ===\n"
        "violino = { c'4^\\tr d' e' f' }\n"
    )
    info = identify_prelude(text)

    out = inline_variables(text, info, random.Random(0), p=1.0)

    assert "tr = \\markup" not in out
    assert "c'4^\\markup { \\italic tr } d'" in out
    assert check_brace_balance(out)


@pytest.mark.parametrize("seed", [0, 1, 42, 1234])
def test_augmentations_preserve_brace_balance(preprocessed_ly: str, seed: int) -> None:
    info = identify_prelude(preprocessed_ly)
    shuffled = shuffle_prelude(preprocessed_ly, info, random.Random(seed))
    assert check_brace_balance(shuffled)

    dropped = drop_unused_prelude(shuffled, identify_prelude(shuffled), random.Random(seed), p=0.5)
    assert check_brace_balance(dropped)
