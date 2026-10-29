# System Card - Statistical Translation

## Overview

- **What it is:** IBM Model 1 and Model 2 word alignment models trained from scratch by
  expectation-maximization, with alignment and word-translation queries.
- **What it is not:** a full translation system; it aligns words, not phrases, and has no
  language model or decoder.
- **Version:** 0.1.0.

## Intended use

- **Primary use cases:** learning how statistical word alignment works; producing word
  alignments from small parallel corpora.
- **Primary users:** engineers and students studying machine translation.

## Out-of-scope / prohibited use

- Not a production translator.
- Pure-Python training limits corpus size.

## Performance

See the README: alignment error rate of 0.250 for both models on held-out gold alignments.

## Risks and limitations

- Word-level only; no phrase or reordering model beyond Model 2's positions.
- Small, synthetic corpus.

## Safety mitigations

- Deterministic training; offline; bounded request sizes in the service.

## Environmental / cost footprint

Negligible; training runs in a few milliseconds on the bundled corpus.
