# Gap Explanation Framework

## Purpose

This framework is meant to explain, in a disciplined way, the gap between:

- literature-reported privacy tightness
- theoretical upper bounds used in this project
- empirical lower bounds observed through auditing

It is designed to be reusable. The goal is not only to report a gap, but to explain *which part of the pipeline is responsible for it*.

## Core Principle

Do not compare two privacy numbers directly until they are aligned on the following dimensions:

1. bound semantics
2. threat model
3. data/model regime
4. auditor family and access level
5. estimator and confidence convention
6. implementation validity

If any of these differ, the comparison is only partially meaningful.

## The Three Quantities To Track

### 1. Literature benchmark

This is not a single number. It is a result from a different setting with its own:

- model family
- dataset
- attack class
- adversary access
- sample budget
- estimator/reporting convention

Literature values should therefore be treated as contextual reference points, not interchangeable targets.

### 2. Project upper bound

This project currently uses at least three upper-bound notions that must stay separated:

- Opacus RDP accountant bound
- exact Google `dp_accounting` PLD bound, when actually available
- analytical GDP-style fallback currently exposed through the "PLD" path when exact PLD is unavailable

These are all upper-bound-like quantities, but they do not have identical semantics.

### 3. Project empirical lower bound

This is the leakage the project can *demonstrate* from an actual auditor and estimator.

It depends on:

- auditor strength
- sample/query budget
- threshold or score-selection logic
- confidence reporting mode
- implementation correctness

## Diagnostic Decomposition

This decomposition is diagnostic, not a strict algebraic identity.

When the project observes a large gap, it should attribute that gap across five buckets:

### A. Accounting gap

The portion caused by the upper bound being loose or by different accountant semantics.

Typical signal:

- RDP upper bound is much larger than a tighter upper bound in the same training regime

### B. Attack gap

The portion caused by the current auditor family not being strong enough to recover more leakage.

Typical signal:

- `epsilon_lower` barely rises as the attack family is strengthened or as sigma changes

### C. Statistical-evidence gap

The portion caused by finite-sample limits, sparse tails, unstable FPR estimates, or confidence intervals forcing conservative collapse.

Typical signal:

- point estimates are positive or pathological, but conservative lower bounds are near zero

### D. Regime mismatch gap

The portion caused by comparing unlike settings across papers or across project tracks.

Typical signal:

- literature result uses stronger access, different data, different model, or different attacker assumptions

### E. Implementation gap

The portion caused by bugs, score-direction mistakes, mis-seeded experiments, or mislabeled result semantics.

Typical signal:

- suspiciously identical runs
- lower bounds above upper bounds for obviously fragile reasons
- results changing dramatically after correctness fixes

## Decision Axes

Every major claim in the project should be evaluated on these axes.

| Axis | Question | Current project reading |
|---|---|---|
| Bound semantics | Are we comparing like-for-like upper bounds? | Partially. RDP is clear; "PLD" still needs semantic relabeling when fallback is used. |
| Threat model | Is the auditor comparable to the theorem or literature claim? | Often no. Passive auditing is much weaker than worst-case theorem settings. |
| Regime alignment | Are data/model/training settings aligned with comparison targets? | Limited. Current strongest evidence is still mostly MNIST + small MLP + low-compute settings. |
| Attack maturity | Is the auditor family stable and trusted? | Raw LiRA looks plausible; simple heuristics are weak; RMIA is exploratory and not yet trusted. |
| Statistical support | Can conservative lower bounds survive finite-sample noise? | Not consistently. This is a major current bottleneck. |
| Implementation validity | Has the pipeline been debugged enough to trust conclusions? | Medium. Several important bugs were fixed, but some result families remain unresolved. |

## Trust Tiers

Every result should carry one of these labels.

### Tier 1. Trusted

Use for claims where:

- implementation is validated
- semantics are clear
- estimator is appropriate
- conservative reporting is stable

### Tier 2. Provisional

Use for claims where:

- the result direction seems plausible
- but one major uncertainty remains

Typical example:

- a result depends on a tighter upper bound that may actually be an analytical fallback rather than exact PLD

### Tier 3. Exploratory

Use for claims where:

- the signal is interesting
- but the result may still be driven by setup or estimator mismatch

Typical example:

- RMIA outputs that are far stronger than all prior baselines and exceed all upper bounds

### Tier 4. Invalidated

Use for claims that should no longer be interpreted as evidence because later debugging showed they were artifacts.

Typical example:

- pre-fix canary claims that depended on broken seed propagation, except as historical debugging evidence

## Current Project Diagnosis Under This Framework

### What looks well-supported

- The project has a functioning privacy-gap measurement pipeline.
- The passive gap is real in the current MNIST/MLP setting.
- Simple heuristic passive scores are weak.
- Raw LiRA is stronger than those heuristics.
- The accounting choice matters strongly in low-sigma regimes.
- Conservative reporting changes the interpretation materially and must remain standard.

### What looks most likely right, but not fully settled

- A large share of the gap at low sigma is accounting-driven.
- A large share of the remaining passive gap is due to attack-family limits plus finite-sample limits.
- The passive auditor family may already be near saturation in the current small-model regime.

### What should not be treated as settled yet

- Any headline claim based on RMIA
- Any conclusion that assumes all reported "PLD" values are exact PLD rather than fallback GDP-style approximations
- Any interpretation of high-sigma lower-bound-above-upper-bound events as a true privacy-contract violation

## Literature Comparison Rule

When comparing this project to a paper, normalize the comparison in this order:

1. Match threat model first.
2. Match adversary access level next.
3. Match conservative vs point reporting convention.
4. Match upper-bound semantics.
5. Only then compare tightness values.

If these are not aligned, report the comparison as qualitative rather than quantitative.

## The Current Best Explanation

The current repo evidence supports the following explanation:

- In stronger-privacy regimes, the observed gap is heavily influenced by loose upper-bound accounting.
- In weaker-privacy regimes, the remaining gap is increasingly explained by auditor strength and limited statistical support.
- Passive auditing is fundamentally weaker than the theorem's worst-case adversary, so a residual passive gap is expected even when the pipeline is correct.
- Some newer very-strong attack outputs are more likely signs of estimator/setup mismatch than evidence that the true lower bound has suddenly become near-tight.

## What This Framework Implies

The project should stop treating the gap as a single mystery number.

Instead, the standard output of the project should become:

- which upper bound was used
- which threat model and attack family were used
- whether the lower bound is conservative or point
- which trust tier the result belongs to
- which gap bucket is most responsible for the remaining discrepancy

That is the right structure for a proper framework, and it is compatible with both the existing pipeline and the newer exploratory work.
