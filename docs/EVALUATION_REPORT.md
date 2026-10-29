# Evaluation Report - Statistical Translation

> Reproduce with `python -m statistical_translation evaluate`. Deterministic and offline.

## Setup

- **Corpus:** a small English-to-pseudo-French parallel corpus.
- **Gold:** held-out sentence pairs with hand-authored sure word alignments.
- **Metric:** alignment error rate (AER), the standard word-alignment metric; NULL alignments
  are not scored.
- **Models:** IBM Model 1 and Model 2 (initialized from Model 1).

## Results

| Model | AER |
|---|---:|
| Model 1 | 0.250 |
| Model 2 | 0.250 |

## Interpretation

Both models reach an AER of 0.250 on held-out gold alignments, correctly aligning content
words across the adjective reordering between the two languages. On this short, lightly
reordered corpus Model 2's position model matches Model 1; the position model's advantage
appears on longer sentences with more reordering, where Model 1's position-independence hurts.
The result confirms that from-scratch expectation-maximization learns meaningful word
correspondences from unaligned bitext.

## Reproducibility

Training is deterministic over fixed data; the numbers regenerate exactly.
