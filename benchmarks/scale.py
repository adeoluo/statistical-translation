"""A deterministic scale benchmark for IBM Model 2 alignment throughput.

Trains IBM Model 2 once on a small full-coverage synthetic lexicon, then runs
word alignment over progressively larger synthetic parallel corpora generated
here with a seeded ``random.Random``. Each generated pair carries a known gold
alignment (determiner, noun, and a reordered adjective), so alignment error rate
is an exact quality metric. For each workload the benchmark reports wall-clock
seconds, alignment throughput in sentence pairs per second, peak memory in MiB
measured with the standard-library ``tracemalloc`` module, and AER.

Timing is measured without tracemalloc (which otherwise slows every allocation);
peak memory is traced separately around the corpus plus a single alignment,
which is faithful because pairs are aligned one at a time. The corpus is
synthetic. This pure-Python from-scratch implementation is not competitive on
wall-clock time with compiled alignment toolkits; the goal is to show how EM
alignment scales with the number of pairs. Deterministic and offline; run with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import random
import time
import tracemalloc
from dataclasses import dataclass

from statistical_translation.corpus import SentencePair
from statistical_translation.data import GoldPair
from statistical_translation.evaluation import alignment_error_rate
from statistical_translation.ibm2 import IBMModel2

WORKLOAD_SIZES: tuple[int, ...] = (10000, 20000, 40000, 80000)
_BYTES_PER_MIB = 1024 * 1024

_DETERMINERS: tuple[tuple[str, str], ...] = (("the", "la"), ("a", "une"))
_ADJECTIVES: tuple[tuple[str, str], ...] = (
    ("red", "rouge"),
    ("big", "grand"),
    ("small", "petit"),
    ("green", "vert"),
    ("blue", "bleu"),
)
_NOUNS: tuple[tuple[str, str], ...] = (
    ("house", "maison"),
    ("book", "livre"),
    ("car", "voiture"),
    ("tree", "arbre"),
    ("dog", "chien"),
    ("cat", "chat"),
)


def _make_pair(rng: random.Random) -> GoldPair:
    """Return one synthetic pair with its gold alignment (1-based source links)."""

    det_en, det_fr = rng.choice(_DETERMINERS)
    noun_en, noun_fr = rng.choice(_NOUNS)
    if rng.random() < 0.5:
        source, target = f"{det_en} {noun_en}", f"{det_fr} {noun_fr}"
        sure = {(0, 1), (1, 2)}
    else:
        adj_en, adj_fr = rng.choice(_ADJECTIVES)
        source = f"{det_en} {adj_en} {noun_en}"
        target = f"{det_fr} {noun_fr} {adj_fr}"  # French reorders the adjective after the noun
        sure = {(0, 1), (1, 3), (2, 2)}
    return GoldPair(pair=SentencePair.from_text(source, target), sure=frozenset(sure))


def _generate(count: int, *, seed: int = 0) -> list[GoldPair]:
    rng = random.Random(seed)
    return [_make_pair(rng) for _ in range(count)]


def _training_corpus() -> list[SentencePair]:
    """Every determiner, noun, and adjective combination, for full lexical coverage."""

    pairs: list[SentencePair] = []
    for det_en, det_fr in _DETERMINERS:
        for noun_en, noun_fr in _NOUNS:
            pairs.append(SentencePair.from_text(f"{det_en} {noun_en}", f"{det_fr} {noun_fr}"))
            for adj_en, adj_fr in _ADJECTIVES:
                pairs.append(
                    SentencePair.from_text(
                        f"{det_en} {adj_en} {noun_en}", f"{det_fr} {noun_fr} {adj_fr}"
                    )
                )
    return pairs


@dataclass(frozen=True)
class ScaleRow:
    """One measured workload: size, timing, throughput, memory, and quality."""

    size: int
    seconds: float
    throughput: float
    peak_mib: float
    aer: float


def _train() -> IBMModel2:
    model = IBMModel2()
    model.train(_training_corpus(), iterations=20)
    return model


def _peak_mib(size: int) -> float:
    """Peak memory to hold ``size`` pairs and run a single alignment."""

    tracemalloc.start()
    tracemalloc.reset_peak()
    corpus = _generate(size, seed=0)
    model = _train()
    model.align(corpus[0].pair)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / _BYTES_PER_MIB


def run_scale(sizes: tuple[int, ...] = WORKLOAD_SIZES) -> list[ScaleRow]:
    """Train once, then measure alignment scaling across corpus sizes."""

    model = _train()
    rows: list[ScaleRow] = []
    for size in sizes:
        corpus = _generate(size, seed=0)
        start = time.perf_counter()
        report = alignment_error_rate(model, corpus)
        elapsed = time.perf_counter() - start
        rows.append(
            ScaleRow(
                size=size,
                seconds=elapsed,
                throughput=size / elapsed if elapsed else 0.0,
                peak_mib=_peak_mib(size),
                aer=report.aer,
            )
        )
    return rows


def main() -> None:
    print("Synthetic parallel corpus with gold links; from-scratch IBM Model 2.")
    print("Pure-Python alignment; not benchmarked against compiled alignment toolkits.")
    print(f"{'pairs':>8}  {'seconds':>9}  {'pairs/sec':>11}  {'peak MiB':>9}  {'AER':>7}")
    for row in run_scale():
        print(
            f"{row.size:>8d}  {row.seconds:>9.3f}  {row.throughput:>11.1f}  "
            f"{row.peak_mib:>9.2f}  {row.aer:>7.3f}"
        )


if __name__ == "__main__":
    main()
