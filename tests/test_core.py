"""Tests for the corpus types and IBM Model 1 training and alignment."""

from __future__ import annotations

import itertools

from statistical_translation import IBMModel1, SentencePair, tokenize


def test_tokenize_and_null() -> None:
    pair = SentencePair.from_text("the house", "la maison")
    assert pair.source == ("the", "house")
    assert pair.source_with_null()[0] == "<null>"
    assert tokenize("The, Cat!") == ["the", "cat"]


def _toy_corpus() -> list[SentencePair]:
    # A tiny but consistent bilingual corpus (English source, pseudo-target).
    return [
        SentencePair.from_text("the house", "la maison"),
        SentencePair.from_text("the book", "le livre"),
        SentencePair.from_text("a house", "une maison"),
        SentencePair.from_text("a book", "un livre"),
        SentencePair.from_text("the red house", "la maison rouge"),
        SentencePair.from_text("the red book", "le livre rouge"),
    ]


def test_em_increases_log_likelihood() -> None:
    model = IBMModel1()
    curve = model.train(_toy_corpus(), iterations=15)
    assert curve[-1] > curve[0]
    # Likelihood is non-decreasing under EM (Model 1 is convex).
    for earlier, later in itertools.pairwise(curve):
        assert later >= earlier - 1e-9


def test_learns_word_translations() -> None:
    model = IBMModel1()
    model.train(_toy_corpus(), iterations=20)
    assert model.translate_word("house", k=1)[0][0] == "maison"
    assert model.translate_word("book", k=1)[0][0] == "livre"
    assert model.translate_word("red", k=1)[0][0] == "rouge"


def test_alignment_points_to_correct_source() -> None:
    model = IBMModel1()
    model.train(_toy_corpus(), iterations=20)
    pair = SentencePair.from_text("the red house", "la maison rouge")
    alignment = dict(model.align(pair).pairs)
    source = pair.source_with_null()
    # "maison" (target index 1) should align to "house"; "rouge" to "red".
    assert source[alignment[1]] == "house"
    assert source[alignment[2]] == "red"
