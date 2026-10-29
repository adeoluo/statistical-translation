"""IBM Model 1 word alignment, trained with expectation-maximization.

IBM Model 1 is the foundation of statistical machine translation. It models the
probability of a target sentence given a source sentence as a product of word
translation probabilities ``t(target_word | source_word)``, with every alignment
equally likely a priori. Those probabilities are learned with EM:

* **E step** - under the current ``t``, compute the posterior probability that
  each target word aligns to each source word (the expected alignment counts).
* **M step** - re-estimate ``t`` by normalizing the accumulated counts.

Model 1's likelihood is convex, so EM converges to the global optimum regardless
of initialization. Everything here is pure Python.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from statistical_translation.corpus import SentencePair, vocabulary


@dataclass(frozen=True)
class Alignment:
    """An alignment: for each target position, the aligned source position.

    Source positions are indexed with the NULL token at position 0, so a value of
    0 means the target word aligns to NULL (no source word).
    """

    pairs: tuple[tuple[int, int], ...]  # (target_index, source_index_with_null)


class IBMModel1:
    """IBM Model 1 translation probabilities trained by expectation-maximization."""

    def __init__(self) -> None:
        self._t: dict[tuple[str, str], float] = {}

    def probability(self, target_word: str, source_word: str) -> float:
        """Return ``t(target_word | source_word)``."""
        return self._t.get((target_word, source_word), 0.0)

    def train(self, pairs: Sequence[SentencePair], *, iterations: int = 15) -> list[float]:
        """Train by EM and return the mean log-likelihood per iteration.

        Returns an empty list and leaves the model empty when there is nothing to
        train on (no pairs, or no target tokens), rather than dividing by zero.
        """
        target_vocab = vocabulary([pair.target for pair in pairs])
        self._t = {}
        if not target_vocab:
            return []
        uniform = 1.0 / len(target_vocab)
        for pair in pairs:
            for target_word in pair.target:
                for source_word in pair.source_with_null():
                    self._t[(target_word, source_word)] = uniform

        log_likelihoods: list[float] = []
        for _ in range(iterations):
            counts: dict[tuple[str, str], float] = {}
            source_totals: dict[str, float] = {}
            total_log_likelihood = 0.0
            for pair in pairs:
                source_tokens = pair.source_with_null()
                for target_word in pair.target:
                    denominator = sum(
                        self._t[(target_word, source_word)] for source_word in source_tokens
                    )
                    total_log_likelihood += math.log(max(denominator, 1e-12))
                    for source_word in source_tokens:
                        delta = self._t[(target_word, source_word)] / denominator
                        counts[(target_word, source_word)] = (
                            counts.get((target_word, source_word), 0.0) + delta
                        )
                        source_totals[source_word] = source_totals.get(source_word, 0.0) + delta
            for (target_word, source_word), value in counts.items():
                self._t[(target_word, source_word)] = value / source_totals[source_word]
            pairs_seen = sum(len(pair.target) for pair in pairs)
            log_likelihoods.append(total_log_likelihood / max(pairs_seen, 1))
        return log_likelihoods

    def align(self, pair: SentencePair) -> Alignment:
        """Return the most probable alignment of target words to source words."""
        source_tokens = pair.source_with_null()
        aligned: list[tuple[int, int]] = []
        for target_index, target_word in enumerate(pair.target):
            best_source = 0
            best_score = -1.0
            for source_index, source_word in enumerate(source_tokens):
                score = self._t.get((target_word, source_word), 0.0)
                if score > best_score:
                    best_score = score
                    best_source = source_index
            aligned.append((target_index, best_source))
        return Alignment(pairs=tuple(aligned))

    def translate_word(self, source_word: str, *, k: int = 1) -> list[tuple[str, float]]:
        """Return the ``k`` most probable target words for a source word."""
        if k <= 0:
            return []
        candidates = [
            (target_word, probability)
            for (target_word, current_source), probability in self._t.items()
            if current_source == source_word
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:k]
