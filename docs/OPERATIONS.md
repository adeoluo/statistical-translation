# Operations Guide - Statistical Translation

The operational contract of the alignment service.

## Running the service

```bash
uv run python -m statistical_translation serve --host 0.0.0.0 --port 8000
```

A standard-library HTTP server with **zero runtime dependencies**. Both models are trained
once at startup and then serve stateless requests.

## Endpoints

| Method | Path | Body | Success |
|---|---|---|---|
| GET | `/health` | none | status and corpus size (liveness) |
| GET | `/ready` | none | readiness once the models are trained |
| GET | `/metrics` | none | per-route request, error, and latency metrics (Prometheus text) |
| POST | `/align` | `{"source", "target"}` | word alignment (Model 2) |
| POST | `/translate` | `{"word", "k"?}` | most probable translations (Model 1) |
| GET | `/evaluate` | none | alignment error rate for both models |

Errors return JSON `{"error": ...}` with a 4xx status; empty, overlong, or out-of-range inputs
are rejected.

## Service level objectives

Measured by `benchmarks/stress.py` on a developer laptop:

| Objective | Target | Observed (single worker) |
|---|---|---|
| Align latency p50 | < 1 ms | about 0.02 ms |
| Align latency p99 | < 1 ms | about 0.05 ms |
| Throughput | > 50000 alignments/second/core | about 400000/second |
| Startup training | < 1 s | a few milliseconds |

## Operational constraints

- **Training is a startup cost; alignment is cheap.** EM runs once in milliseconds on the
  bundled corpus; each alignment is a table lookup over the sentence positions.
- **The align path is CPU bound and holds the global interpreter lock.** Throughput is flat
  across threads; scale with processes.
- **Requests are bounded**: sentences are length-capped and `k` is bounded.
- **No persistence, no authentication.** Front the service with a gateway.

## Failure modes and responses

| Symptom | Likely cause | Response |
|---|---|---|
| Poor alignment | Out-of-domain or unknown words | Retrain on in-domain bitext |
| `400` on request | Empty, overlong, or bad parameters | Inspect the `error` field |
| Rising p99 under load | Concurrency above one worker per core | Add replicas; cap concurrency |

## Capacity planning

A single core serves hundreds of thousands of alignments per second on short sentences; scale
with process-per-core replicas.
