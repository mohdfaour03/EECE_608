# Ideas and Hypotheses

These are not project results. Each item must be tested or reviewed before use.

## I-001 - Existing HPO winners may be salvageable

**Status:** HYPOTHESIS

If HPO touched only the old evaluation/non-member pool, the utility winners may
still be usable provided the passive audit receives a newly sampled disjoint
non-member pool that never influenced selection. This avoids recomputing HPO but
changes the audit split contract. Risks: limited unused data, distribution
shift, and comparability with planned cells.

## I-002 - Treat accountant choice as typed provenance

**Status:** PROPOSAL

Instead of a generic `epsilon_upper_theory`, record a typed structure containing
`epsilon_pld`, `epsilon_rdp`, `target_epsilon`, `target_accountant`, and
`tightness_denominator`. This should make accountant mismatches structurally
harder to introduce.

## I-003 - Separate censored estimates from numeric zero in the schema

**Status:** PROPOSAL

Represent an unavailable/censored lower bound as `null` plus a censoring reason,
never as `0.0`. Reserve numeric zero for a valid estimator output that is truly
zero under the declared procedure.

## I-004 - Use a checkpoint manifest

**Status:** PROPOSAL

Pair every long-run checkpoint with a small manifest containing file ID, SHA-256,
study name, finished-trial count, code/config digest, upload time, and SQLite
integrity result. The manifest would simplify recovery and expose mismatched or
stale uploads immediately.

## I-005 - Replace algebraic gates with falsifiable gates

**Status:** PROPOSAL

A useful gate must fail on plausible bad data. Candidate decomposition gates:
epsilon agreement, estimator validity, uncensored support, complete seed set,
accountant identity, artifact integrity, and bounded tightness where required.
