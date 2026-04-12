# Support-Scaled Pilot Findings

## Scope

This pilot reran the sidecar 3-dataset matrix with exact Google PLD enabled and scaled the audit evidence budget while keeping the underlying training regime fixed at the same subset-sized smoke configuration.

- Datasets: `mnist`, `cifar10`, `adult`
- Passive attacks: `negative_loss`, `score_fusion`
- Active attack: `canary_random` for image datasets only
- Support levels:
  - passive `smoke`: budget `64` with `1` seed
  - passive `medium`: budget `256` with `3` seeds
  - passive `large`: budget `512` with `5` seeds
  - canary `smoke`: `8` canaries with `1` seed
  - canary `medium`: `16` canaries with `3` seeds
  - canary `large`: `32` canaries with `5` seeds

Main artifact:
- `codex/results/support_scaled_pilot/support_scaled_pilot_summary.json`

## Main Readout

Only one condition produced a non-zero conservative lower bound:

- `cifar10 + passive_negative_loss + large`
  - `epsilon_upper_pld_exact = 2.635023`
  - `epsilon_lower_conservative = 0.118652`
  - `epsilon_lower_point = 0.612024`
  - implied exact-PLD tightness ratio `~4.5%`

Everything else remained at conservative `epsilon_lower = 0.0`.

## What This Suggests

### 1. Finite-sample limitations are real, not hypothetical

The smoke matrix mostly had positive point estimates but zero conservative lower bounds. In the scaled pilot, one attack/dataset combination crossed into a positive conservative lower bound only after increasing support to `1280` member and `1280` non-member scores.

That means part of the earlier gap was caused by weak statistical support, not just by weak attacks or bad accounting.

### 2. Support alone does not fix everything

Several conditions still stayed at conservative zero even after support scaling.

- `mnist` remained conservative-zero for both passive attacks and canary
- `adult` remained conservative-zero for both passive attacks
- `cifar10 + score_fusion` remained conservative-zero despite a large point estimate

This means the residual gap is not purely a sample-size artifact.

### 3. Some attacks are still relying on sparse-tail events

Many runs still carried warnings about the selected threshold being in a sparse tail region. In those cases the point estimate can look large, but the confidence-supported lower side collapses.

Important example:

- `cifar10 + passive_score_fusion + large`
  - point estimate `1.787484`
  - conservative lower bound `0.0`
  - selected `TPR = 0.00234375`
  - selected `FPR = 0.0`

This is exactly the kind of result that looks exciting in a point-estimate table but is too tail-fragile for a robust claim.

### 4. Adult currently looks attack-limited or regime-limited

On `adult`, support scaling did not unlock a conservative lower bound. The best thresholds tended to have member and non-member event rates that were too close to each other.

Example:

- `adult + passive_negative_loss + large`
  - point estimate `0.026359`
  - `TPR = 0.51015625`
  - `FPR = 0.496875`

That is not a sample problem anymore; it is a weak-separation problem.

### 5. Random canaries are still too weak at this scale

Even after increasing the canary counts and repeated seeds, canary auditing stayed conservative-zero on both image datasets.

This suggests one of:

- too few canaries even in the current "large" sidecar setting
- random canaries are weak for this regime
- canaries need optimization or a stronger insertion design

## Current Implied Story

The current best story is:

- exact PLD fixed the upper-bound semantics
- part of the smoke-run gap was caused by low audit support
- after support scaling, a small but real conservative lower bound appears in at least one condition
- however most of the gap remains, so the project is still facing a combination of:
  - finite-sample limitations
  - attack weakness
  - regime mismatch

## Next Best Steps

1. Freeze `cifar10 + passive_negative_loss + large` as the first clean proof that support scaling can matter.
2. Run a stronger passive family next, ideally a properly integrated Raw LiRA sidecar, because the current packaged attacks remain weak.
3. Strengthen canary auditing separately:
   - more canaries
   - stronger canary construction than pure random
4. Keep reporting both:
   - point estimate
   - conservative lower bound

Point estimates alone are too easy to over-interpret in this project.
