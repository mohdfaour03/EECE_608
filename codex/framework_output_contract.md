# Framework Output Contract

## Purpose

For each evaluated configuration, the framework should emit a single structured result that is sufficient to compare:

- the theoretical privacy upper bound
- the empirical auditing lower bound
- the remaining gap

without hiding the assumptions that make the comparison valid.

## Required Inputs

- dataset
- model
- DP training regime
- auditor family
- auditor hyperparameters
- audit sample-support settings

## Required Outputs

### Experiment identity

- `dataset`
- `model_name`
- `training_regime_id`
- `attack_name`
- `attack_family`

### Theoretical side

- `epsilon_upper_rdp`
- `epsilon_upper_tighter`
- `upper_bound_backend`
  - e.g. `google_dp_accounting`, `analytical_gaussian`

### Empirical side

- `epsilon_lower_conservative`
- `epsilon_lower_point`
- `selected_tpr`
- `selected_fpr`
- `num_member_samples`
- `num_nonmember_samples`

### Attack semantics

- `score_direction`
  - `higher_is_member` or `lower_is_member`
- `score_name`
- `calibration_method`

This field is mandatory because the Raw LiRA sidecar work showed that using the wrong score direction can materially distort the empirical lower bound.

### Gap metrics

- `privacy_loss_gap`
- `tightness_ratio`

### Trust / diagnostics

- `warning`
- `trust_tier`
  - `trusted`
  - `provisional`
  - `exploratory`
  - `invalidated`
- `diagnostic_tags`
  - examples:
    - `sparse_tail`
    - `low_support`
    - `score_direction_sensitive`
    - `pathological_distribution`
    - `attack_limited`

## Minimum Framework Guarantees

1. The upper-bound backend must be explicit.
2. The attack score direction must be explicit.
3. Conservative and point lower bounds must both be kept.
4. Support size must be reported.
5. Pathological or non-credible results must not be mixed with trusted results without a trust tag.

## Current Lessons Driving This Contract

### 1. Upper-bound backend matters

Switching from analytical fallback to exact Google PLD changed the narrative materially.

### 2. Support size matters

Some results only became conservative-positive after support scaling.

### 3. Score direction matters

Raw LiRA produced materially different lower bounds once evaluated with `lower_is_member` instead of the default higher-score convention.

### 4. Trust tags matter

The `adult + raw_lira` results currently exceed the exact PLD upper bound, which makes them useful for debugging but unsafe for headline interpretation.
