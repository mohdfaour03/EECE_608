# Current State And Gap Analysis

## Scope

This note is a sidecar synthesis written under `codex/` so the existing project state is left untouched.

Primary sources inspected:

- `README.md`
- `research/findings.md`
- `paper/main.tex`
- `autoresearch/experiment.py`
- `autoresearch/prepare.py`
- `autoresearch/results/experiment_results.csv`
- `autoresearch/results/lira_v2_results.csv`
- `autoresearch/notebooks/colab_sigma_sweep.ipynb`
- `autoresearch/notebooks/colab_rmia.ipynb`
- `src/dp_audit_tightness/privacy/accounting.py`
- `src/dp_audit_tightness/privacy/pld_accounting.py`
- `src/dp_audit_tightness/privacy/gdp_estimation.py`

## What Has Already Been Done

### 1. The project has a real end-to-end auditing pipeline

The repository is no longer a bare scaffold. It already includes:

- DP-SGD training with saved checkpoints and structured JSON/CSV metadata
- canary auditing under an evaluator-controlled threat model
- passive auditing from final-model outputs
- empirical lower-bound estimation from member/non-member score distributions
- saturation tracking and aggregated reporting
- multiple datasets/models/config families beyond the earliest MNIST-only framing

### 2. Important engineering issues were already found and partly fixed

From the paper draft and code history, the team already identified:

- a canary seed-propagation bug that caused identical-looking canary results across seeds
- passive score-direction mistakes that allowed non-member-favoring events to inflate estimates
- sparse-tail threshold-selection pathologies where `epsilon_lower` could exceed `epsilon_upper`
- the need for confidence-aware conservative reporting via Wilson intervals

These are substantial fixes. They mean the team has already moved past the "does the pipeline run?" phase and into "which results are trustworthy?".

### 3. The project already has a first explanation of the gap

The strongest current explanation appears in `research/findings.md` and the sigma-sweep notebook:

- the total gap is not just one thing
- it can be decomposed into:
  - an accounting gap: the difference between a looser upper bound and a tighter upper bound
  - an auditing gap: the remaining distance between the tighter upper bound and the observed lower bound

That decomposition is the most useful conceptual step already achieved in the project.

### 4. Newer exploratory work exists beyond the paper draft

The paper draft mainly reflects:

- canary auditing
- simple passive scoring rules
- one MNIST/MLP regime around `epsilon_upper ~= 0.771`

But the newer `autoresearch` branch goes further:

- Raw LiRA experiments
- sigma sweeps over noise multiplier
- RDP vs tighter upper-bound comparisons
- RMIA experiments using production ML Privacy Meter code

This means the paper is behind the newest exploratory direction.

## Best Current Understanding Of Why The Gap Exists

Based on the current repo, the gap is best explained by a combination of four factors.

### 1. Theoretical upper-bound looseness is part of the story

At low sigma, the findings log says most of the gap is due to the upper bound itself being loose when using the standard RDP accountant. The sigma sweep explicitly reports that the accounting gap dominates in the stronger-privacy regime.

Important nuance:

- the project's `compute_theoretical_upper_bound()` uses the training accountant directly
- the project's so-called "PLD" path can fall back to an analytical GDP-style approximation if the Google `dp_accounting` PLD backend is unavailable

So the project already suspects that some of the gap comes from loose accounting, but the exact semantics of the tighter upper bound still need to be nailed down carefully.

### 2. The auditor family is probably saturating in the current setup

The sigma sweep suggests `epsilon_lower` stays relatively stable while `epsilon_upper` changes a lot with sigma. That means the observed gap is closing mostly because the upper bound comes down, not because the attack becomes much stronger.

That is a strong sign that, for this dataset/model/budget:

- the current passive auditor family may already be near its ceiling
- the remaining gap is not explained by attack scaling alone

This is one of the most important current conclusions.

### 3. Sample-budget limits and estimator instability are still a major bottleneck

Several results in the paper and notebooks become pathological when:

- sample budgets are small
- the best threshold lands on a sparse score-distribution tail
- FPR is effectively near zero

That creates unstable or inflated empirical lower bounds. The conservative Wilson-based fix is the right direction, but it also collapses some claims to zero. That does not mean leakage is absent; it means the current sample size cannot support a stable positive lower-bound claim.

### 4. Threat-model mismatch matters

The formal DP upper bound is worst-case over datasets and adversaries. The passive auditor is much weaker than that setting:

- model-only access
- limited query budget
- heuristic scores or current LiRA/RMIA variants

So a large passive gap is not, by itself, evidence that the theory is wrong. Part of it may simply be that the empirical adversary is weaker than the worst-case adversary the theorem protects against.

## What The Current Results Seem To Say

### Stable conclusions

- The project has a functioning privacy-gap measurement pipeline.
- The gap is real and substantial in the current MNIST/MLP setup.
- Simple passive heuristics are weak.
- Raw LiRA is meaningfully stronger than simple passive scores.
- Accounting choice matters a lot, especially at low sigma.
- Statistical conservatism matters; point estimates alone are not trustworthy enough.

### Unstable or not-yet-settled conclusions

- Whether the tighter upper bound is truly exact PLD or sometimes only an analytical GDP-style surrogate
- Whether the canary track can make non-zero conservative claims after the bug fixes without scaling sample budgets
- Whether RMIA is genuinely stronger here or currently producing invalid/pathological results
- Whether the high-sigma regime is revealing a real accountant mismatch or just small-sample estimation noise

## What Still Needs To Be Done

### Priority 1. Lock down upper-bound semantics

Before making stronger scientific claims, the project should clearly separate:

- Opacus RDP upper bound
- Google `dp_accounting` PLD upper bound, when actually available
- analytical GDP-style fallback used when PLD is unavailable

Right now these are close conceptually but not interchangeable. A large part of the research question depends on this distinction.

### Priority 2. Establish one canonical, trusted passive baseline

The strongest candidate appears to be Raw LiRA, not the older heuristic scores.

The next step should be to define one canonical passive benchmark with:

- fixed training config
- fixed shadow-model regime
- fixed query/sample budget
- fixed conservative estimator
- fixed artifact format

Without that, every new notebook risks shifting multiple variables at once.

### Priority 3. Investigate RMIA before treating it as evidence

The newest RMIA notebook reports extremely large conservative ratios, often far above both RDP and tighter upper bounds. That is too strong to trust at face value.

RMIA needs a focused validation pass:

- inspect whether score construction or population/reference handling is mismatched
- confirm the conservative estimator is still appropriate for RMIA scores
- verify that any FPR near zero is not just another sparse-tail artifact

Until that happens, RMIA should be treated as exploratory, not conclusive.

### Priority 4. Increase sample support for conservative lower bounds

The project already learned that confidence-aware reporting is necessary. The next step is to make that reporting informative rather than mostly zero.

That likely means increasing some combination of:

- repeated audit seeds
- query budget
- number of canaries
- number of shadow/reference models

This is probably required before the project can make strong conservative claims in the passive setting.

### Priority 5. Separate "engineering validated" from "scientifically explained"

The repo is already strong on engineering validation. The missing piece is a disciplined explanation program:

- how much of the gap is accountant looseness
- how much is attack weakness
- how much is estimator/sample-budget limitation
- how much is inherent threat-model mismatch

The sigma-sweep decomposition is the right backbone for that explanation.

### Priority 6. Update the written narrative to match the actual state

The paper draft is currently behind the newer exploratory branch. It still emphasizes older canary/simple-passive results, while the repo now has stronger LiRA and RMIA exploration.

The project needs a unified narrative that says:

- what is the stable baseline
- what is exploratory
- which claims are trustworthy now
- which claims remain provisional

## Recommended Safe Next Actions

These are the highest-value next steps that should not require disruptive changes to the main project structure:

1. Build a canonical result table that lists, for each reported finding:
   - config
   - threat model
   - attack family
   - upper-bound type
   - lower-bound estimator
   - conservative vs point result
   - trust status

2. Audit the accounting backend used in every "tight" upper-bound result and relabel anything that was actually produced by the analytical fallback rather than exact PLD.

3. Reframe the current scientific claim as:
   "the gap is partly an accounting artifact and partly an auditor/statistical limitation, with the balance depending strongly on sigma."

4. Treat Raw LiRA as the current best passive baseline and RMIA as an unresolved follow-up, not yet a headline result.

5. Design the next experiment set specifically to answer:
   "Does conservative `epsilon_lower` rise when sample support increases, or has the passive attack family already saturated?"

## Bottom Line

The project already did the hard work of building the pipeline and uncovering the first real explanation of the gap.

The best current interpretation is:

- in strong-privacy regimes, much of the gap is due to loose upper-bound accounting
- in weaker-privacy regimes, the remaining gap is increasingly about auditor strength and sample-limited estimation
- the passive threat model is fundamentally weaker than the worst-case theorem setting
- some of the newest "very strong" exploratory results are likely not ready to trust yet

So the next stage is not to rebuild the project. It is to formalize one trusted baseline, validate the accounting semantics, and run targeted experiments that isolate whether the remaining gap is attack-limited, sample-limited, or theory-limited.
