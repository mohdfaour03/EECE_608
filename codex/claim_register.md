# Claim Register

## Purpose

This register separates current project claims into:

- trusted
- provisional
- exploratory
- invalidated or historical-only

It is meant to stop strong claims from spreading faster than the evidence supports them.

## Trusted

### Claim

The project has a functioning end-to-end privacy auditing pipeline.

### Why it is trusted

- the repo has real training, auditing, estimation, reporting, and validation code
- multiple debugging cycles already happened on real outputs

### Next step

- keep this claim as background, not as the research contribution

## Trusted

### Claim

The gap between upper and lower privacy bounds is substantial in the current MNIST/MLP setting.

### Why it is trusted

- supported by the paper draft
- supported by the findings log
- supported by multiple passive and canary experiments

### Next step

- refine the explanation, not the existence, of the gap

## Trusted

### Claim

Simple passive heuristic scores are weak in this setup.

### Why it is trusted

- repeated low or zero conservative performance in prior passive experiments
- paper draft already treats these as weaker than stronger attack families

### Next step

- keep them as baseline comparators only

## Provisional

### Claim

At low sigma, most of the total gap is accounting-driven.

### Why it is only provisional

- the sigma-sweep decomposition strongly supports it
- but the exact semantics of the tighter upper bound still need to be audited carefully because "PLD" may sometimes be a fallback analytical approximation

### What would upgrade it to trusted

- explicit confirmation of which upper-bound backend was used for each result

## Provisional

### Claim

The passive auditor family is nearing saturation in the current regime.

### Why it is only provisional

- `epsilon_lower` appears comparatively stable across sigma while `epsilon_upper` changes a lot
- but this could still be partly driven by finite-sample limitations rather than pure attack saturation

### What would upgrade it to trusted

- support-scaling experiments showing conservative lower bounds remain flat even as sample support grows

## Provisional

### Claim

Raw LiRA is the strongest currently credible passive baseline.

### Why it is only provisional

- newer repo evidence points in this direction
- but the project still lacks one canonical baseline artifact and trust-standardized summary

### What would upgrade it to trusted

- freeze one canonical Raw LiRA config and reproduce it under the framework reporting format

## Exploratory

### Claim

RMIA is much stronger than Raw LiRA in this project.

### Why it is exploratory

- the newest RMIA outputs are dramatically stronger than prior baselines
- many of those outputs exceed both RDP and tighter upper bounds
- this is currently more likely a validation problem than a final result

### What must happen next

- validate score semantics, support assumptions, and estimator compatibility before using RMIA in any headline claim

## Exploratory

### Claim

At high sigma, the audit may exceed the tighter theoretical upper bound because the theorem side is too aggressive.

### Why it is exploratory

- the sigma sweep shows this behavior
- but the same behavior is also consistent with finite-sample noise or fallback-bound mismatch

### What must happen next

- distinguish exact PLD from fallback results and repeat in a higher-support regime

## Historical-only

### Claim

Pre-fix canary results showing about one-third tightness are reliable headline evidence.

### Why it is historical-only

- the paper itself documents that these were entangled with a seed-propagation bug
- they remain useful as debugging history, but not as the cleanest current scientific evidence

### What should replace it

- a post-fix, conservative, statistically supported canary baseline or a clear statement that current canary support is still insufficient

## Invalidated

### Claim

GDP-based AUC lower-bound estimation is a valid default empirical estimator for this project.

### Why it is invalidated

- the repo findings explicitly say this estimator is not valid for the DP-SGD score shapes observed here
- it produced pathological overestimates

### What should replace it

- keep threshold-sweep + conservative confidence reporting as the default until a better validated estimator is established

## Current Headline Claim The Project Can Safely Make

The project can currently say, with reasonable discipline:

"In the present MNIST/MLP DP-SGD setup, the privacy tightness gap is real. Part of it is caused by loose upper-bound accounting, and part of it is caused by passive-auditor and finite-sample limitations. Raw LiRA appears to be the strongest currently credible passive baseline, while some newer stronger-looking results remain exploratory until their semantics and statistical validity are confirmed."

That is strong enough to be meaningful and cautious enough to remain defensible.
