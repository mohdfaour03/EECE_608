# Raw LiRA Pilot Findings

## Scope

This sidecar pilot reproduced the notebook-style `Raw LiRA` idea in code and then corrected its evaluation direction after a focused pathology check showed the score semantics were inverted.

- Datasets: `mnist`, `cifar10`, `adult`
- Budget: `512` queries per seed
- Seeds: `401, 402, 403, 404, 405`
- Shadow counts: `K=8`, `K=16`
- Comparator: matched `negative_loss` baseline on the exact same queried examples
- Upper bound: exact Google PLD from the target model training run
- Correct score direction for Raw LiRA: `lower_is_member`

Main artifacts:

- `codex/results/raw_lira_pilot/raw_lira_pilot_summary.json`
- `codex/results/raw_lira_pathology/raw_lira_pathology_summary.json`

## Main Readout After Direction Fix

### Matched baseline

`cifar10 + negative_loss + budget=512`

- conservative lower bound: `0.118652`
- exact-PLD upper bound: `2.635023`
- tightness: `~4.5%`

### Raw LiRA

`mnist + raw_lira + K=16`

- conservative lower bound: `0.894024`
- exact-PLD upper bound: `1.812994`
- tightness: `~49.3%`

`cifar10 + raw_lira + K=16`

- conservative lower bound: `0.309951`
- exact-PLD upper bound: `2.635023`
- tightness: `~11.8%`

`adult + raw_lira + K=16`

- conservative lower bound: `4.734196`
- exact-PLD upper bound: `1.187876`
- tightness: `~398.5%`

This last row is not scientifically credible as stated.

## What Changed

The initial Raw LiRA run was evaluated using the framework default assumption that higher scores imply stronger membership evidence.

The pathology pass showed:

- `cifar10`: preferred direction is `lower_is_member`
- `adult`: preferred direction is `lower_is_member`

So the runner was corrected to log and use explicit score direction for Raw LiRA.

## Interpretation

### 1. Raw LiRA is a real improvement on at least two datasets

Compared with the matched `negative_loss` baseline:

- `mnist` improved from conservative `0.0` to `0.894024`
- `cifar10` improved from conservative `0.118652` to `0.309951`

That is a meaningful result for the framework: some of the gap really was attack-limited.

### 2. CIFAR-10 is currently the most credible Raw LiRA result

The `cifar10 + raw_lira + K=16` row is:

- directionally consistent
- stronger than the matched passive baseline
- below the exact PLD upper bound

This is the best current candidate for a stronger passive headline result.

### 3. MNIST is promising but still needs a credibility check

The corrected `mnist` result became much larger after fixing the score direction. That could be real, but it changed enough that it should be treated as `provisional` until it is rechecked under stricter diagnostics.

### 4. Adult remains pathological

The `adult` row is still not trustworthy even after fixing direction.

Why:

- the conservative lower bound exceeds the exact theoretical upper bound by a large margin
- the score distributions are extremely heavy-tailed
- the selected threshold still sits in a sparse-tail regime

This means the attack is surfacing a regime that the current lower-bound estimator cannot safely interpret as a normal, trustworthy result.

## Current Trust View

### Most credible positive result

- `cifar10 + raw_lira + K=16`

### Promising but still provisional

- `mnist + raw_lira + K=16`

### Suspicious / not for headline use

- `adult + raw_lira + K=8/16`
- old notebook-style Gaussian LiRA outputs

## Framework Lesson

The main framework lesson from this pilot is not only that stronger attacks matter.

It is also that:

- attack outputs require explicit semantic metadata
- especially `score_direction`

Without that, the framework can compute the wrong empirical lower bound even if the attack implementation itself is fine.

## Next Best Steps

1. Freeze score direction as a required field in the framework output contract.
2. Treat `cifar10 + raw_lira + K=16` as the strongest current passive result.
3. Recheck `mnist + raw_lira + K=16` before promoting it to trusted.
4. Keep `adult + raw_lira` in the debugging bucket until the pathology is explained.
