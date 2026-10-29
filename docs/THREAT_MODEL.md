# Threat Model - Statistical Translation

## Assets

- Integrity of the learned tables and the alignment and translation outputs.

## Trust boundaries

- **Source and target sentences are untrusted input to the service.** They are tokenized and
  aligned, never executed, and length-bounded.

## Threats and mitigations

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Unbounded computation | Very long sentences | Resource use | Inputs length-bounded |
| Unbounded results | Huge `k` | Large responses | `k` bounded to 25 |
| Irreproducible training | Nondeterministic EM | Flaky tables | Deterministic EM over fixed data |
| Supply-chain compromise | Malicious dependency | Code execution | Zero runtime dependencies; Dependabot and CodeQL in continuous integration |

## Residual risk

Alignment quality depends on the training bitext; out-of-domain input aligns poorly. This is a
capability limit of the method, not a security flaw.
