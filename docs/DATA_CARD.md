# Data Card - Statistical Translation

The project bundles one small, synthetic parallel corpus with gold alignments. There is no
external data and nothing is downloaded.

## bundled corpus (`src/statistical_translation/data.py`)

- **Provenance:** authored for this project. English source sentences paired with pseudo-French
  targets, covering determiners, nouns, adjectives, and adjective reordering.
- **Gold alignments:** a held-out set of pairs with hand-authored sure word alignments for
  alignment-error-rate evaluation.
- **Licensing:** authored by this repository; MIT.
- **Known gaps:** tiny, clean, and synthetic; real bitext is noisier and larger.

## Ethical considerations

The corpus is synthetic and contains no personal, sensitive, or copyrighted content.
