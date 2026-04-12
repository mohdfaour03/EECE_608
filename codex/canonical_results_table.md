# Canonical Results Table

## Purpose

This table converts the current project state into one normalized view:

- what the result is
- what upper bound it uses
- what lower-bound convention it uses
- whether it is trusted enough for headline use

This is not the final paper table. It is the framework-native sidecar version.

## Project Results

| Result family | Threat model | Setting | Upper-bound reference | Lower-bound reference | Reported tightness | Trust tier | Use in narrative |
|---|---|---|---|---|---|---|---|
| Simple passive heuristics | Passive final-model-only | MNIST, small MLP, sigma ~= 1.1, query budget 128 | Opacus RDP upper bound | Conservative threshold-sweep lower bound | Mostly 0% conservative; some non-pathological small positives; many pathological cases before fixes | Trusted as weak baseline | Use as evidence that simple black-box heuristics are weak |
| Random canary pre-fix result | Evaluator-controlled canary stress test | MNIST, small MLP, sigma ~= 1.1 | Opacus RDP upper bound | Threshold-sweep lower bound | About 35.8% | Historical-only | Keep as debugging history, not as clean headline evidence |
| Raw LiRA v2 best prior result | Passive final-model-only | MNIST, small MLP, sigma ~= 1.1, K=32 shadows, budget=256, 5 seeds | Opacus RDP upper bound | Conservative threshold-sweep lower bound | About 6.1% conservative | Provisional | Strong candidate for canonical passive baseline |
| Raw LiRA sigma sweep, low-to-mid sigma | Passive final-model-only | MNIST, small MLP, sigma in {0.5, 0.8, 1.1, 1.5, 2.0, 3.0} | RDP upper bound plus a tighter reported bound currently labeled "PLD" in project notes | Conservative threshold-sweep lower bound | About 0.4% -> 31.2% vs RDP and 5.4% -> 56.7% vs tighter reported upper bound | Provisional | Use for decomposition and trend interpretation, but audit tighter-bound semantics first |
| Raw LiRA sigma sweep, sigma=5.0 | Passive final-model-only | MNIST, small MLP, sigma = 5.0 | Same as above | Conservative threshold-sweep lower bound | About 44.0% vs RDP and above the tighter reported upper bound | Exploratory | Do not use as a clean privacy-contract conclusion yet |
| RMIA sigma sweep | Passive final-model-only | MNIST, small MLP, multiple sigma values, production RMIA code | RDP upper bound plus tighter reported upper bound | Conservative threshold-sweep lower bound applied to RMIA scores | Extremely large ratios, often far above both upper bounds | Exploratory | Treat as unresolved until semantics and estimator validity are checked |
| GDP AUC lower-bound estimator | Passive final-model-only | Used in exploratory autoresearch | GDP/AUC-based lower-bound estimator | GDP bootstrap conversion | Pathological overestimates | Invalidated | Do not use as default empirical estimator |

## Literature Context Rows

These are reference rows, not direct apples-to-apples competitors.

| Literature reference | Access / threat model | Reported qualitative takeaway | Comparison status |
|---|---|---|---|
| Nasr et al. 2023 | Stronger white-box / worst-case-style auditing regime | Near-tight audits can be possible in stronger settings | Qualitative context only unless setting is normalized |
| Steinke et al. 2023 | Strong canary-based auditing regime | One-run canary methods can approach tightness under stronger constructions | Qualitative context only unless canary construction and reporting are aligned |
| Pillutla et al. 2024 | Black-box / weaker-access regime | Tightness can remain modest in weaker-access settings, especially outside best-case conditions | Useful context for passive results, but still not a numeric target without normalization |

## Current Headline-Safe Subset

If the project had to make disciplined claims today, the safest subset is:

1. The project has a functioning privacy-gap pipeline.
2. The passive gap is substantial in the current MNIST/MLP setting.
3. Simple passive heuristics are weak.
4. Raw LiRA is the strongest currently credible passive baseline.
5. A meaningful portion of the gap, especially at low sigma, appears to come from upper-bound looseness.
6. The remaining passive gap is plausibly due to attack-strength limits plus finite-sample limitations.

## Results That Should Not Yet Lead The Narrative

These should stay out of the headline claim set for now:

- RMIA appears dramatically stronger than everything else
- high-sigma lower-bound-above-upper-bound events prove the theorem side is wrong
- the canary track already provides a clean conservative near-tight result
- GDP/AUC estimation is a valid replacement for the current conservative estimator

## Immediate Use

This table should be the source of truth when deciding:

- what belongs in the paper
- what belongs in exploratory appendices
- which results should be rerun first
- which claims need explicit caveats
