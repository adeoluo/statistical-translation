"""Alignment Error Rate evaluation for word alignment models.

Alignment Error Rate (AER) is the standard metric for word alignment. With a set
of predicted links A and gold sure links S (and optionally possible links P), AER
is ``1 - (|A and S| + |A and P|) / (|A| + |S|)``. Lower is better. Here only sure
links are annotated, so possible equals sure. NULL alignments (source position 0)
are not scored, matching the standard protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from statistical_translation.data import GoldPair
from statistical_translation.ibm1 import Alignment as Alignment1
from statistical_translation.ibm2 import Alignment as Alignment2

AlignmentLike = Alignment1 | Alignment2


def _predicted_links(alignment: AlignmentLike) -> set[tuple[int, int]]:
    # Convert (target, source_with_null) pairs to (target, source_1_based), dropping NULL.
    return {(target, source) for target, source in alignment.pairs if source > 0}


@dataclass(frozen=True)
class AERReport:
    """Alignment Error Rate over a set of gold pairs."""

    aer: float
    predicted: int
    sure: int
    correct: int

    def format(self) -> str:
        return (
            f"alignment error rate: {self.aer:.3f} "
            f"(correct {self.correct}, predicted {self.predicted}, sure {self.sure})"
        )


def alignment_error_rate(aligner: object, gold: Sequence[GoldPair]) -> AERReport:
    """Compute AER for an aligner exposing ``align(pair) -> Alignment``."""
    total_predicted = 0
    total_sure = 0
    total_correct = 0
    for gold_pair in gold:
        alignment = aligner.align(gold_pair.pair)  # type: ignore[attr-defined]
        predicted = _predicted_links(alignment)
        sure = set(gold_pair.sure)
        total_predicted += len(predicted)
        total_sure += len(sure)
        total_correct += len(predicted & sure)
    denominator = total_predicted + total_sure
    aer = 1.0 - (2 * total_correct) / denominator if denominator else 0.0
    return AERReport(aer=aer, predicted=total_predicted, sure=total_sure, correct=total_correct)
