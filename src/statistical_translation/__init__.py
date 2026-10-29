"""Statistical Translation - IBM Model 1 and 2 word alignment from scratch."""

from __future__ import annotations

from statistical_translation.corpus import NULL, SentencePair, tokenize, vocabulary
from statistical_translation.data import GoldPair, gold_pairs, train_pairs
from statistical_translation.evaluation import AERReport, alignment_error_rate
from statistical_translation.ibm1 import Alignment, IBMModel1
from statistical_translation.ibm2 import IBMModel2

__version__ = "0.1.0"

__all__ = [
    "NULL",
    "AERReport",
    "Alignment",
    "GoldPair",
    "IBMModel1",
    "IBMModel2",
    "SentencePair",
    "__version__",
    "alignment_error_rate",
    "gold_pairs",
    "tokenize",
    "train_pairs",
    "vocabulary",
]
