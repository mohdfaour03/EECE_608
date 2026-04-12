# Framework Work Program

## Goal

Turn the project from "a set of experiments with partial explanations" into "a reusable framework that explains why the gap exists and how much each factor contributes."

All work below is intended to live alongside the current project, not replace it.

## Workstream 1. Standardize Result Semantics

### Objective

Make sure every reported result says exactly what kind of upper bound and lower bound it uses.

### Actions

- Create a canonical label set for upper bounds:
  - `upper_rdp_opacus`
  - `upper_pld_exact`
  - `upper_gdp_fallback`
- Create a canonical label set for lower bounds:
  - `lower_point_threshold_sweep`
  - `lower_conservative_threshold_sweep`
  - any future LiRA- or RMIA-specific conservative forms if needed
- Relabel historical findings accordingly before using them in a paper-quality comparison

### Acceptance criteria

- No result table uses the word "PLD" unless exact PLD was actually used
- Every figure/table row can be traced to a precise upper/lower-bound semantic pair

## Workstream 2. Freeze One Canonical Passive Baseline

### Objective

Pick one passive setup that becomes the trusted baseline for comparison.

### Recommended baseline

- dataset: MNIST
- model: current small MLP
- training regime: current 1-epoch DP-SGD baseline
- attack family: Raw LiRA
- reporting mode: conservative lower bound

### Actions

- Fix one training config
- Fix one shadow-model protocol
- Fix one query budget
- Fix one seed/repetition policy
- Save results in a repeatable artifact format

### Acceptance criteria

- The same passive baseline can be rerun and summarized without notebook-specific interpretation
- All future passive claims are compared against this baseline first

## Workstream 3. Separate Attack Strength From Statistical Power

### Objective

Determine whether the remaining gap is attack-limited or sample-limited.

### Actions

- Scale repeated audit seeds while holding attack family fixed
- Scale query budget while holding attack family fixed
- Scale shadow/reference model count while holding attack family fixed
- Compare point and conservative lower bounds under each scaling step

### Readout

- If conservative `epsilon_lower` rises with sample support, the bottleneck is partly statistical
- If conservative `epsilon_lower` stays flat despite more support, the attack family is likely saturating

### Acceptance criteria

- One plot or table shows `epsilon_lower_conservative` as a function of sample support for the canonical passive baseline

## Workstream 4. Quantify Accounting Gap Versus Auditing Gap

### Objective

Make the decomposition from the findings log a standard project output.

### Actions

- For every canonical run, record:
  - RDP upper bound
  - exact-PLD upper bound when available
  - fallback upper bound when exact PLD is unavailable
  - conservative audited lower bound
- Compute:
  - total gap
  - accounting gap
  - residual auditing gap

### Acceptance criteria

- Every headline result reports not just tightness, but also the gap decomposition
- The decomposition is comparable across sigma values and attack families

## Workstream 5. Validate Or Reject RMIA

### Objective

Resolve whether RMIA is a true improvement or currently an invalid result family.

### Actions

- Verify score semantics and reference/population construction
- Check whether the conservative estimator is appropriate for RMIA score distributions
- Inspect whether near-zero FPR events are driving extreme ratios
- Compare RMIA against Raw LiRA under the same sample-support budget and trust convention

### Decision rule

- If RMIA still produces massive bound violations after semantic and statistical checks, classify it as invalid for the current framework
- If it remains stronger but stable under conservative reporting, promote it from exploratory to provisional

### Acceptance criteria

- RMIA ends in one of two states:
  - accepted as provisional evidence
  - explicitly excluded from headline claims

## Workstream 6. Build A Literature-Normalized Comparison Table

### Objective

Compare this project to prior work in a way that is actually fair.

### Actions

- Create a table where each literature comparison row includes:
  - threat model
  - access level
  - dataset/model regime
  - upper-bound semantics
  - lower-bound convention
  - trust note

### Acceptance criteria

- No literature comparison is presented as direct numeric competition unless the settings are aligned enough to justify it

## Workstream 7. Promote The Framework To The Default Reporting Format

### Objective

Make future outputs framework-native rather than notebook-native.

### Required fields for every future result

- experiment id
- dataset/model/training regime
- threat model
- attack family
- upper-bound type
- lower-bound type
- conservative or point
- trust tier
- gap attribution note

### Acceptance criteria

- New results can be added to the project without requiring a custom narrative each time

## Immediate Next Sequence

This is the recommended order.

1. Relabel upper-bound semantics.
2. Freeze Raw LiRA as the canonical passive baseline.
3. Run support-scaling experiments to separate statistical limits from attack saturation.
4. Recompute the accounting-gap versus auditing-gap decomposition with the corrected semantics.
5. Validate or reject RMIA.
6. Update the paper narrative only after the above are settled.

## Bottom Line

The next proper step is not "try random stronger attacks."

It is:

- standardize semantics
- freeze one trusted passive baseline
- isolate statistical power from attack strength
- validate or reject RMIA
- only then elevate new conclusions into the project's main narrative
