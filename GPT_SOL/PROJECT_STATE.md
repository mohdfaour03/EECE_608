# Project State

**Last updated:** 2026-07-16  
**Maintainer:** GPT SOL

## Current status

- **VERIFIED:** Fable produced an independent static review in
  [`../Fable/VALIDATION_REPORT_2026-07-16.md`](../Fable/VALIDATION_REPORT_2026-07-16.md).
- **VERIFIED:** that review reports the five July 6 fixes as correctly
  implemented and identifies four blocking pre-run issues plus five substantial
  hardening issues.
- **SUPPORTED (remote execution history):** the current MNIST E1 Colab campaign
  completed epsilon targets 0.3, 0.5, 1.0, 2.0, and 4.0. The epsilon 8.0 Optuna
  study has 23/30 COMPLETE trials in the durable Google Drive SQLite checkpoint.
- **BLOCKED (external):** Colab rejected a new GPU allocation because the
  account reached its GPU usage limit. The seven remaining epsilon 8.0 trials
  are not running.
- **RESOLVED CONTRACT / PENDING IMPLEMENTATION:** Fable decision D-003 permits
  conditional salvage of the existing E1 winners using a fresh disjoint MNIST
  test-set audit pool, an overfit-to-split check, provenance labeling, and a
  checkpoint manifest.
- **BLOCKED_USER:** Fable Task 0 requires a baseline git commit before Wave 1.
  The current tree includes extensive pre-existing user changes, unrelated
  artifacts, and a plaintext credential in `.mcp.json`; GPT SOL requires an
  approved inclusion/exclusion scope before staging or committing.

## Immediate engineering priorities from Fable's report

1. Filter GDP table inputs on `estimator_valid`.
2. Make PLD the explicit canonical denominator for the target-epsilon grid, or
   obtain a documented alternative decision.
3. Establish a three-way HPO/validation/audit split contract.
4. Exclude censored cells from arithmetic instead of coercing them to zero.
5. Make result and SQLite checkpoint writes/downloads atomic and identity-safe.
6. Prevent stale E2 checkpoint reuse and assert post-training epsilon.
7. Replace the tautological G2 gate with informative quality gates.

## Protected or high-risk areas

- `src/dp_audit_tightness/privacy/empirical.py`
- `src/dp_audit_tightness/auditing/canary/seeding.py`
- `src/dp_audit_tightness/utils/validation.py`
- Accountant selection and every epsilon denominator.
- Dataset split seeds and audit-pool provenance.

Changes in these areas require targeted tests and explicit rev