# End-to-End Case Study - Discovering Word Alignments from Bitext

How a model learns which words translate to which, with no dictionary and no supervision.

## Scenario

A team has parallel sentences but no word-level dictionary. They want to know, for each target
word, which source word it corresponds to - the first step of statistical translation.

## Aligning

```bash
curl -s -X POST localhost:8000/align -d '{"source": "the red house", "target": "la maison rouge"}'
```

```json
{"source": ["the", "red", "house"], "target": ["la", "maison", "rouge"],
 "alignment": [
   {"target": "la", "source": "the"},
   {"target": "maison", "source": "house"},
   {"target": "rouge", "source": "red"}]}
```

The model correctly aligns `maison` to `house` and `rouge` to `red` despite the adjective
appearing on opposite sides of the noun in the two languages. It learned this from
co-occurrence across the corpus, via expectation-maximization, with no supervision.

## Measuring it

```bash
curl localhost:8000/evaluate
# {"model1_aer": 0.25, "model2_aer": 0.25}
```

Alignment error rate on held-out gold alignments is 0.25, clearly better than chance under the
strict standard metric.

## What an operator sees

- **A fast, cheap aligner.** Training is milliseconds; alignment is hundreds of thousands of
  sentences per second.
- **Explicit word links.** The response returns each target word and its aligned source word.
- **An honest metric.** AER is reported on held-out gold alignments, not training data.

## Takeaway

Expectation-maximization turns raw parallel text into word alignments, the discovery that made
statistical machine translation possible, implemented here from the E and M steps up.
