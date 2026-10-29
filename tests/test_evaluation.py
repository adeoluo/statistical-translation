"""Tests for IBM Model 2, the gold corpus, and alignment-error-rate evaluation."""

from __future__ import annotations

from statistical_translation import (
    IBMModel1,
    IBMModel2,
    SentencePair,
    alignment_error_rate,
    gold_pairs,
    train_pairs,
)
from statistical_translation.cli import main


def test_model2_learns_alignment() -> None:
    model = IBMModel2()
    model.train(train_pairs(), iterations=15)
    pair = SentencePair.from_text("the red house", "la maison rouge")
    alignment = dict(model.align(pair).pairs)
    source = pair.source_with_null()
    assert source[alignment[1]] == "house"  # maison -> house
    assert source[alignment[2]] == "red"  # rouge -> red


def test_alignment_error_rate_is_reasonable() -> None:
    gold = gold_pairs()
    model1 = IBMModel1()
    model1.train(train_pairs(), iterations=15)
    report = alignment_error_rate(model1, gold)
    assert 0.0 <= report.aer <= 1.0
    assert report.aer < 0.5  # clearly better than chance
    assert report.correct > 0


def test_model2_at_least_matches_model1() -> None:
    gold = gold_pairs()
    model1 = IBMModel1()
    model1.train(train_pairs(), iterations=15)
    model2 = IBMModel2()
    model2.train(train_pairs(), iterations=15)
    report1 = alignment_error_rate(model1, gold)
    report2 = alignment_error_rate(model2, gold)
    assert report2.aer <= report1.aer + 1e-9


def test_cli_evaluate(capsys) -> None:
    main(["--iterations", "12", "evaluate"])
    out = capsys.readouterr().out
    assert "alignment error rate" in out
    assert "Model 1" in out and "Model 2" in out


def test_cli_align(capsys) -> None:
    main(["--iterations", "12", "align", "the red house", "la maison rouge"])
    out = capsys.readouterr().out
    assert "->" in out


def test_cli_translate(capsys) -> None:
    main(["--iterations", "12", "translate", "house"])
    out = capsys.readouterr().out
    assert "maison" in out
