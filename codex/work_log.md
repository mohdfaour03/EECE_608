# Codex Work Log

## Goal

Build a sidecar framework for explaining the gap between:

- the theoretical privacy upper bound
- the empirical auditing lower bound

and determine whether the remaining gap comes mainly from:

- loose accounting
- weak attacks
- limited statistical support
- regime mismatch

## What Has Been Done

### 1. Repo state and gap framing

Created sidecar synthesis documents:

- `codex/current_state_gap_analysis.md`
- `codex/gap_explanation_framework.md`
- `codex/framework_work_program.md`
- `codex/claim_register.md`
- `codex/canonical_results_table.md`

These established the main working framework:

- accounting gap
- attack gap
- statistical-evidence gap
- regime mismatch gap
- implementation gap

### 2. Exact-PLD enablement

Installed `dp-accounting` into `codex/.venv` and patched:

- `src/dp_audit_tightness/privacy/pld_accounting.py`

Outcome:

- the project now uses the Google backend successfully in the isolated venv
- previous smoke results that used analytical fallback were preserved for comparison

### 3. Smoke matrix

Ran:

- `codex/run_smoke_matrix.py`

Artifacts:

- `codex/results/smoke_matrix/smoke_matrix_summary.json`
- `codex/results/smoke_matrix/smoke_matrix_summary.pre_google_fix.json`

Main finding:

- the pipeline works end-to-end across `mnist`, `cifar10`, and `adult`
- positive point estimates appeared
- conservative lower bounds mostly stayed at `0.0`

### 4. Support-scaled pilot

Ran:

- `codex/run_support_scaled_pilot.py`

Artifacts:

- `codex/results/support_scaled_pilot/support_scaled_pilot_summary.json`
- `codex/support_scaled_pilot_findings.md`

Main finding:

- increasing audit support matters
- first clean non-zero conservative result:
  - `cifar10 + passive_negative_loss + large`
  - `epsilon_lower_conservative = 0.118652`
  - exact-PLD tightness about `4.5%`
- most other conditions remained conservative-zero

### 5. Raw LiRA pilot

Ran:

- `codex/run_raw_lira_pilot.py`

Artifacts:

- `codex/results/raw_lira_pilot/raw_lira_pilot_summary.json`
- `codex/raw_lira_pilot_findings.md`

Main finding:

- initial run showed `Raw LiRA` can help, but later diagnostics found the score direction was inverted
- the sidecar runner was then corrected to use explicit `score_direction`
- corrected rerun results:
  - `mnist + raw_lira + K=16`: conservative `0.894024`
  - `cifar10 + raw_lira + K=16`: conservative `0.309951`
  - `adult + raw_lira + K=16`: conservative `4.734196`
- the `adult` result is not credible and remains flagged as pathological

### 6. Raw LiRA pathology / directionality check

Ran:

- `codex/analyze_raw_lira_pathology.py`

Artifacts:

- `codex/results/raw_lira_pathology/raw_lira_pathology_summary.json`

Main finding:

- for `raw_lira`, the preferred score direction is `lower_is_member`, not the framework default
- this is true on both `cifar10` and `adult`
- therefore the framework needs explicit score-direction metadata for every attack family
- the `adult + raw_lira` result remains suspicious even after correcting the direction, because it exceeds the exact PLD upper bound by a wide margin

### 7. Canonical framework ledger

Ran:

- `codex/build_framework_ledger.py`

Artifacts:

- `codex/results/framework_ledger/framework_ledger.json`
- `codex/results/framework_ledger/framework_ledger.csv`
- `codex/results/framework_ledger/framework_ledger_summary.md`

Main finding:

- the pilots are now merged into one sidecar ledger with explicit:
  - upper-bound backend
  - score direction
  - support profile
  - trust tier
  - diagnostic tags
- no audit row is fully trusted yet
- the current best rows are still `provisional`, while `adult + raw_lira` is explicitly `invalidated`

### 8. Full framework validation pass

Added:

- `codex/framework_validation_protocol.md`
- `codex/run_framework_validation.py`

Ran:

- `codex/run_framework_validation.py`

Artifacts:

- `codex/results/framework_validation/framework_validation_summary.json`
- `codex/results/framework_validation/framework_validation_rows.csv`
- `codex/results/framework_validation/framework_validation_checks.json`
- `codex/results/framework_validation/framework_validation_report.md`

Main finding:

- the framework now validates operationally end-to-end on the canonical matrix
- exact Google PLD is active for all training runs
- score direction is explicit for every supported attack row
- matched negative-loss agrees with the canonical passive baseline
- the framework explicitly catches the `adult + raw_lira` overshoot as pathological rather than treating it as a valid headline result
- validation run time was about `194` seconds and all `70` validation checks passed

### 9. GPU-scale validation runner

Added:

- `codex/framework_validation_gpu_protocol.md`
- `codex/run_framework_validation_gpu_scale.py`

Verified:

- the script compiles successfully
- the launcher exits immediately with a clear CUDA-only guard on the current machine, which has no visible GPU

Purpose:

- this is the larger-scale validation run meant for a real GPU environment
- it scales:
  - passive query budget to `2048`
  - passive seeds to `10`
  - Raw LiRA to `K=32`
  - canaries to `128` per seed
  - training to multi-epoch, near-full/full dataset workloads

Current status:

- prepared but not executed here because `torch.cuda.is_available()` is `False`

## Current Best Supported Story

1. Exact PLD is working and gives the correct upper-bound side of the comparison.
2. Part of the observed gap was due to low audit support.
3. Stronger passive attacks can improve the lower bound materially in some settings.
4. Attack semantics such as score direction are framework-critical and must be logged explicitly.
5. Some result families are still too fragile or pathological to trust.

## Current Trusted / Provisional Status

### Trusted enough to use in the narrative

- exact-PLD upper bounds from the venv-backed runs
- `cifar10 + passive_negative_loss + large`

### Provisionally interesting

- `mnist + raw_lira + K=16`
- `cifar10 + raw_lira + K=16`

### Suspicious / not for headline use

- `adult + raw_lira`
- old notebook-style Gaussian LiRA outputs

## Immediate Next Tasks

1. Run `codex/run_framework_validation_gpu_scale.py` on a CUDA-enabled machine.
2. Explain why `adult + raw_lira` overshoots the exact upper bound.
3. Decide what evidence threshold should move a row from `provisional` to genuinely trusted.
