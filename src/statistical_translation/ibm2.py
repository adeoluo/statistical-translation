"""IBM Model 2 word alignment, trained with expectation-maximization.

Model 2 extends Model 1 by adding an explicit alignment probability that depends
on positions: ``q(i | j, l, m)`` is the probability that target position ``j``
(of length ``m``) aligns to source position ``i`` (of length ``l``). This lets
the model prefer monotonic or locally reordered alignments instead of treating
all alignments as equally likely.

Following standard practice, the translation table is initialized from a trained
Model 1 (whose convex likelihood avoids poor local optima), and then Model 2's EM
updates both the translation table ``t`` and the alignment table ``q``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from statistical_translation.corpus import SentencePair
from statistical_translation.ibm1 import IBMModel1


@dataclass(frozen=True)
class Alignment:
    """An alignment of target positions to source positions (0 = NULL)."""

    pairs: tuple[tuple[int, int], ...]


class IBMModel2:
    """IBM Model 2, initialized from Model 1 and trained by EM."""

    def __init__(self) -> None:
        self._t: dict[tuple[str, str], float] = {}
        self._q: dict[tuple[int, int, int, int], float] = {}

    def train(
        self, pairs: Sequence[SentencePair], *, model1_iterations: int = 10, iterations: int = 10
    ) -> None:
        """Initialize ``t`` from Model 1, then run Model 2 EM over ``t`` and ``q``."""
        model1 = IBMModel1()
        model1.train(pairs, iterations=model1_iterations)
        self._t = dict(model1._t)

        # Initialize q uniformly over source positions for each (j, l, m).
        self._q = {}
        for pair in pairs:
            source_len = len(pair.source_with_null())
            target_len = len(pair.target)
            for j in range(target_len):
                for i in range(source_len):
                    self._q[(i, j, source_len, target_len)] = 1.0 / source_len

        for _ in range(iterations):
            t_counts: dict[tuple[str, str], float] = {}
            t_totals: dict[str, float] = {}
            q_counts: dict[tuple[int, int, int, int], float] = {}
            q_totals: dict[tuple[int, int, int], float] = {}
            for pair in pairs:
                source = pair.source_with_null()
                src_len, tgt_len = len(source), len(pair.target)
                for j, target_word in enumerate(pair.target):
                    denominator = sum(
                        self._q[(i, j, src_len, tgt_len)]
                        * self._t.get((target_word, source[i]), 0.0)
                        for i in range(src_len)
                    )
                    if denominator <= 0.0:
                        continue
                    for i, source_word in enumerate(source):
                        delta = (
                            self._q[(i, j, src_len, tgt_len)]
                            * self._t.get((target_word, source_word), 0.0)
                            / denominator
                        )
                        t_counts[(target_word, source_word)] = (
                            t_counts.get((target_word, source_word), 0.0) + delta
                        )
                        t_totals[source_word] = t_totals.get(source_word, 0.0) + delta
                        q_counts[(i, j, src_len, tgt_len)] = (
                            q_counts.get((i, j, src_len, tgt_len), 0.0) + delta
                        )
                        q_totals[(j, src_len, tgt_len)] = (
                            q_totals.get((j, src_len, tgt_len), 0.0) + delta
                        )
            for (target_word, source_word), value in t_counts.items():
                self._t[(target_word, source_word)] = value / t_totals[source_word]
            for (i, j, src_len, tgt_len), value in q_counts.items():
                self._q[(i, j, src_len, tgt_len)] = value / q_totals[(j, src_len, tgt_len)]

    def align(self, pair: SentencePair) -> Alignment:
        """Return the most probable alignment using both ``t`` and ``q``."""
        source = pair.source_with_null()
        src_len, tgt_len = len(source), len(pair.target)
        aligned: list[tuple[int, int]] = []
        for j, target_word in enumerate(pair.target):
            best_i = 0
            best_score = -1.0
            for i, source_word in enumerate(source):
                score = self._q.get((i, j, src_len, tgt_len), 1.0 / src_len) * self._t.get(
                    (target_word, source_word), 0.0
                )
                if score > best_score:
                    best_score = score
                    best_i = i
            aligned.append((j, best_i))
        return Alignment(pairs=tuple(aligned))
