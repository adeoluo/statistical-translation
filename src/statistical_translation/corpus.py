"""Types for a parallel corpus of sentence pairs.

A sentence pair holds a source sentence and its target translation as token
lists. Following the standard IBM-model convention, the target is generated from
the source, and a special NULL token is prepended to the source so that target
words with no source counterpart can align to it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

NULL = "<null>"
_TOKEN = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class SentencePair:
    """A source sentence and its target translation, as token lists."""

    source: tuple[str, ...]
    target: tuple[str, ...]

    def source_with_null(self) -> tuple[str, ...]:
        """The source tokens with a leading NULL token."""
        return (NULL, *self.source)

    @classmethod
    def from_text(cls, source: str, target: str) -> SentencePair:
        return cls(source=tuple(tokenize(source)), target=tuple(tokenize(target)))


def vocabulary(sentences: Sequence[Sequence[str]]) -> set[str]:
    """Return the set of tokens appearing across the sentences."""
    return {token for sentence in sentences for token in sentence}
