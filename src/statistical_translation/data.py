"""A bundled parallel corpus with gold word alignments for evaluation.

The corpus is a small English-to-pseudo-French set whose structure (determiners,
nouns, adjectives, and adjective reordering) exercises both lexical translation
and position-dependent alignment. Each evaluation pair carries a gold alignment
as sure links; alignment points use 1-based source indices into the source
sentence (NULL is position 0 and is not scored).
"""

from __future__ import annotations

from dataclasses import dataclass

from statistical_translation.corpus import SentencePair

TRAIN: tuple[tuple[str, str], ...] = (
    ("the house", "la maison"),
    ("the book", "le livre"),
    ("a house", "une maison"),
    ("a book", "un livre"),
    ("the red house", "la maison rouge"),
    ("the red book", "le livre rouge"),
    ("the big house", "la grande maison"),
    ("a red book", "un livre rouge"),
    ("the small book", "le petit livre"),
    ("a big house", "une grande maison"),
)


def train_pairs() -> list[SentencePair]:
    """Return the training sentence pairs."""
    return [SentencePair.from_text(source, target) for source, target in TRAIN]


@dataclass(frozen=True)
class GoldPair:
    """A sentence pair with a gold alignment (1-based source indices)."""

    pair: SentencePair
    sure: frozenset[tuple[int, int]]  # (target_index, source_index_1_based)


def gold_pairs() -> list[GoldPair]:
    """Return held-out pairs with gold sure alignments."""
    specs = (
        ("the red house", "la maison rouge", {(0, 1), (1, 3), (2, 2)}),
        ("a big book", "un grand livre", {(0, 1), (1, 2), (2, 3)}),
        ("the small house", "la petite maison", {(0, 1), (1, 3), (2, 2)}),
    )
    pairs: list[GoldPair] = []
    for source, target, sure in specs:
        pairs.append(GoldPair(pair=SentencePair.from_text(source, target), sure=frozenset(sure)))
    return pairs
