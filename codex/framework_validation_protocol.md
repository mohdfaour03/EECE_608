# Framework Validation Protocol

## Goal

Validate the framework as an end-to-end system that, for a given setup, emits:

- a theoretical upper bound
- an empirical auditing lower bound
- explicit attack semantics
- diagnostics that explain whether the comparison is trustworthy

This is not the same as proving the final scientific claim. It is a framework-validation pass.

## Canonical Validation Matrix

### Datasets

- `mnist`
- `cifar10`
- `adult`

### Audit families

- `passive_negative_loss`
  - threshold-sweep passive baseline
- `passive_raw_lira`
  - stronger passive sidecar attack
- `canary_random`
  - active canary stress-test

### Validation-only consistency row

- `passive_negative_loss_matched`
  - same support and seed layout as the Raw LiRA sidecar path
  - used to check that the baseline agrees across implementation paths

## Canonical Support Settings

- passive query budget per seed: `512`
- passive audit seeds: `401, 402, 403, 404, 405`
- Raw LiRA shadow count: `K=16`
- canary seeds: `101, 102, 103, 104, 105`
- canaries per seed: `32`

## Validation Requirements

### Execution checks

1. Training succeeds on all three datasets.
2. Passive baseline succeeds on all three datasets.
3. Raw LiRA succeeds on all three datasets.
4. Canary succeeds on image datasets and returns an explicit expected limitation on `adult`.

### Accounting checks

1. Exact upper-bound backend is `google_dp_accounting`.
2. `epsilon_upper_tighter` is positive.
3. `epsilon_upper_tighter <= epsilon_upper_rdp`.

### Semantic checks

1. Every supported attack row records explicit `score_direction`.
2. Raw LiRA uses `lower_is_member`.
3. Threshold-style attacks use `higher_is_member`.
4. Matched negative-loss agrees with the canonical passive baseline up to a tight numerical tolerance.

### Sanity checks

1. `epsilon_lower_conservative <= epsilon_lower_point`.
2. `selected_tpr` and `selected_fpr` lie in `[0, 1]` when present.
3. `num_member_samples` and `num_nonmember_samples` are positive for supported audit rows.
4. Any empirical lower bound that materially exceeds the theoretical upper bound must be flagged as pathological rather than accepted silently.

## Expected Outcome Shape

- framework execution: should pass
- scientific trust: may remain provisional on some rows
- pathologies: should be surfaced explicitly, not hidden

## Deliverables

- canonical validation summary JSON
- canonical validation CSV
- explicit validation checks JSON
- human-readable validation report
