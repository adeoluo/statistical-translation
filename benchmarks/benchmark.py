"""A deterministic micro-benchmark for training and alignment throughput.

Measures EM training time for Model 1 and Model 2, the held-out alignment error
rate, and alignment throughput. Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from statistical_translation.data import gold_pairs, train_pairs
from statistical_translation.evaluation import alignment_error_rate
from statistical_translation.ibm1 import IBMModel1
from statistical_translation.ibm2 import IBMModel2


def main() -> None:
    pairs = train_pairs()
    gold = gold_pairs()

    start = time.perf_counter()
    model1 = IBMModel1()
    model1.train(pairs, iterations=15)
    m1_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    model2 = IBMModel2()
    model2.train(pairs, iterations=15)
    m2_ms = (time.perf_counter() - start) * 1000

    report1 = alignment_error_rate(model1, gold)
    report2 = alignment_error_rate(model2, gold)

    sentences = [gp.pair for gp in gold] * 300
    start = time.perf_counter()
    for pair in sentences:
        model2.align(pair)
    align_s = time.perf_counter() - start

    print(f"Train pairs: {len(pairs)}")
    print(f"Model 1 training (15 iters): {m1_ms:.0f} ms | AER {report1.aer:.3f}")
    print(f"Model 2 training (15 iters): {m2_ms:.0f} ms | AER {report2.aer:.3f}")
    print(f"Alignment throughput: {len(sentences) / align_s:.0f} sentences/sec")


if __name__ == "__main__":
    main()
