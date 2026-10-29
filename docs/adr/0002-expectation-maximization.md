# ADR-0002: Learn alignment by expectation-maximization

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

Word alignments between a source and target sentence are not observed; only the sentence pairs
are. The model must discover which words correspond from co-occurrence alone.

## Decision

Treat alignments as latent variables and learn the translation probabilities with
expectation-maximization: the E step computes expected alignment counts under the current
parameters, and the M step re-estimates the parameters by normalizing those counts.

## Alternatives considered

- **Supervised alignment.** Would need hand-aligned data, which is expensive and defeats the
  purpose of learning from raw bitext.
- **Heuristic co-occurrence (Dice, pointwise mutual information).** Cheap but ignores
  competition between candidate alignments; EM models it through the posterior.

## Consequences

- The model learns from unaligned parallel text, the setting statistical translation targets.
- Model 1's likelihood is convex, so EM reaches the global optimum.
- EM is iterative; the number of iterations trades compute for convergence.
