# GPT SOL Implementation Queue

Source: [`../Fable/VALIDATION_REPORT_2026-07-16.md`](../Fable/VALIDATION_REPORT_2026-07-16.md)

No item is complete merely because code was edited. Completion requires the
listed acceptance evidence.

## Phase A - Scientific contract

- [x] **A1: Choose the canonical epsilon denominator.**
  - Proposed default: PLD, because the sigma solver targets PLD epsilon.
  - Resolved by `Fable/DECISIONS.md` D-002: PLD/google is canonical; RDP is a
    secondary diagnostic.
  - Acceptance: every target-grid result records `epsilon_pld`, `epsilon_rdp`,
    and the named denominator used for tightness; no ambiguous
    `epsilon_upper_theory` division remains in target-grid reporting.

- [ ] **A2: Define a three-way data split.**
  - Separate HPO utility selection, final utility evaluation, and passive audit
    non-members.
  - Acceptance: split identities and seeds are persisted; tests prove the index
    sets are pairwise disjoint.

- [x] **A3: Decide whether existing E1 HPO studies are reusable.**
  - Candidate salvage: freeze winners and construct a fresh, disjoint audit
    non-member pool that never influenced HPO.
  - Alternative: rerun E1 under the three-way split.
  - Acceptance: Fable review plus user decision recorded in `DECISION_LOG.md`.
  - Resolved by `Fable/DECISIONS.md` D-003: conditional salvage using a fresh
    disjoint audit pool and four required gates.

## Phase B - Blocking correctness fixes

- [ ] **B1: Filter invalid GDP estimates from Table 1 aggregation.**
  - Files: `experiments/aggregate_results.py` and tests.
  - Acceptance: a row with `quality_flag="ok"`, non-null GDP epsilon, and
    `estimator_valid=false` cannot enter GDP means or tightness summaries.

- [ ] **B2: Apply the canonical PLD denominator consistently.**
  - Files: audit runners, gap decomposition, records/aggregation, tests.
  - Acceptance: target epsilon and reported denominator agree within 1e-3;
    RDP remains available as a secondary diagnostic.

- [ ] **B3: Implement the approved split contract.**
  - Files: dataset preparation, HPO runner/config schema, audit runners, tests.
  - Acceptance: deterministic pairwise-disjoint sets and persisted provenance.

- [ ] **B4: Preserve censoring semantics in decomposition arithmetic.**
  - Replace `or 0.0` coercion with explicit missing/censored representation.
  - Acceptance: censored rows do not contribute to means or shares; counts and
    table footnotes report exclusions.

## Phase C - Operational hardening

- [ ] **C1: Atomic JSON persistence.**
  - Implement temporary write, flush/fsync as appropriate, then `os.replace`.
  - Acceptance: interruption simulation leaves either the old valid file or the
    new valid file, never partial JSON.

- [ ] **C2: Safe Drive checkpoint identity and download.**
  - Pin a Drive file ID or scoped folder; download to a temporary file; validate
    SQLite; atomically replace the local checkpoint.
  - Acceptance: duplicate filenames cannot redirect the checkpoint and a failed
    download cannot truncate the current DB.

- [ ] **C3: Bind E2 checkpoint reuse to the full training contract.**
  - Include dataset/split, hyperparameters, target epsilon, accountant, code or
    config digest, and seed.
  - Acceptance: any contract mismatch refuses reuse with a clear diagnostic.

- [ ] **C4: Enforce post-training epsilon.**
  - Persist `target_epsilon`; recompute achieved PLD epsilon from actual run
    parameters; fail outside tolerance.
  - Acceptance: a changed batch size, epoch count, or dataset size triggers the
    gate instead of producing an apparently on-target record.

- [ ] **C5: Replace the tautological G2 gate.**
  - Gate on estimator validity, censoring coverage, epsilon agreement, seed
    completeness, and artifact integrity.
  - Acceptance: deliberately bad inputs fail at least one informative gate.

## Phase D - Review hardening and cleanup

- [ ] Assert sigma-solver calls use `backend="google"`.
- [ ] Remove or repair vacuous artifact-based tests.
- [ ] Correct double-counted audit/censor counts and quality-flag precedence.
- [ ] Restore the documented 123/124/125 MNIST training seed contract.
- [ ] Make sigma tie-breaking conservative (prefer more noise).
- [ ] Correct stale PLD fallback documentation.
- [ ] Resolve repository housekeeping without overwriting unrelated user work.

## Phase E - Experiment continuation

- [ ] Generate reviewed bundles and record hashes.
- [ ] Determine whether the current 23-trial epsilon 8.0 checkpoint is reusable
  under decision A