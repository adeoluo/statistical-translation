# Scaling and Design Tradeoffs - Statistical Translation

How the reference design grows, and the tradeoffs taken.

## Reference architecture (today)

A single process trains Model 1 and Model 2 over an in-memory corpus and serves alignment and
translation over a standard-library HTTP service. Deterministic over fixed data.

## Scaling path

| Stage | Bottleneck | Change |
|---|---|---|
| One process | Align CPU, global interpreter lock | One replica per core |
| Large corpus | Pure-Python EM | Train offline; load tables; prune rare pairs |
| Long sentences | Quadratic alignment table | Beam over alignment positions |
| Memory | Dense translation table | Prune low-probability entries; hashing |
| Many replicas | Repeated startup training | Train once; ship tables as an artifact |

## Substantial tradeoffs

- **Expectation-maximization vs heuristic co-occurrence.** EM models competition between
  candidate alignments and is exact, at a higher cost than heuristics like Dice, which
  ignore it.
- **Model 2 vs Model 1.** Model 2's position model helps on reordered sentences but is
  non-convex and must be warm-started from Model 1; Model 1 is convex and simpler but
  position-independent.
- **Word alignment vs phrase-based translation.** Word alignment is the foundation; a full
  translator adds phrase extraction, a language model, and a decoder, which this project leaves
  as a boundary.
- **Dense vs pruned tables.** Dense tables are exact but grow with the vocabulary product;
  pruning low-probability entries trades a little accuracy for memory.

## What would change under real load

Train offline and ship the tables, serve from process-per-core replicas, prune rare entries,
and add beam alignment for long sentences. The EM training and the alignment inference are
unchanged; only the data plumbing changes.
