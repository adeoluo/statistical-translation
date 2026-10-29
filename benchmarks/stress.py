"""A production-style stress benchmark for the alignment service.

Trains once, then drives the align handler under sustained, concurrent load and
reports throughput with p50, p95, and p99 latency. Deterministic and offline;
run with:

    uv run python benchmarks/stress.py
"""

from __future__ import annotations

import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from statistical_translation.service import TranslationService, handle

_PAIRS = (
    ("the red house", "la maison rouge"),
    ("a big book", "un grand livre"),
    ("the small house", "la petite maison"),
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    service = TranslationService.trained(iterations=15)
    rng = random.Random(0)
    requests = [{"source": s, "target": t} for s, t in (rng.choice(_PAIRS) for _ in range(3000))]

    def one(body: dict[str, object]) -> float:
        start = time.perf_counter()
        handle("POST", "/align", body, service)
        return (time.perf_counter() - start) * 1000

    print(f"Requests: {len(requests)}")
    for workers in (1, 4, 8):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            latencies = list(pool.map(one, requests))
        wall = time.perf_counter() - start
        print(
            f"workers={workers:<2}  "
            f"throughput={len(requests) / wall:>7.0f} req/s  "
            f"p50={statistics.median(latencies):>6.3f} ms  "
            f"p95={_percentile(latencies, 0.95):>6.3f} ms  "
            f"p99={_percentile(latencies, 0.99):>6.3f} ms"
        )


if __name__ == "__main__":
    main()
