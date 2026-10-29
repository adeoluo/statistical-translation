# ADR-0003: Initialize Model 2 from a trained Model 1

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

Model 2 adds a position-dependent alignment probability, which makes its likelihood
non-convex. EM on a non-convex objective is sensitive to initialization and can land in poor
local optima.

## Decision

Train Model 1 first (whose convex likelihood has a single global optimum) and use its learned
translation table to initialize Model 2, then run Model 2's EM over both the translation and
alignment tables. This is the standard IBM-model training pipeline.

## Alternatives considered

- **Random or uniform initialization of Model 2.** Simpler but prone to poor local optima and
  worse alignments.
- **Model 2 only.** Loses the convex warm start; typically converges to a worse solution.

## Consequences

- Model 2 starts from a good translation table and refines alignment positions.
- The pipeline runs two EM phases, which is more compute but far more reliable.
- The position model helps most on longer, reordered sentences; on short ones it matches
  Model 1.
