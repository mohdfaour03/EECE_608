# Independent Validation Report — 2026-07-16

**Reviewer:** Claude Fable 5 (independent session, no prior involvement in this repo)
**Scope:** Full static code review of the repository, focused on the most recent AI-assisted work: the 2026-07-06 estimator/scorer fixes (HANDOFF_OPUS_FIXES_2026-07-06) and the 2026-07-11/13 additions (decomposition runner, HPO tooling, rebuilt bundles). No experiments or tests were executed (per request: code review only). Two sub-reviews were fanned out and their findings spot-verified against the source before inclusion.

---

## Verdict

**Yes — the recent model did a good job, with caveats.** The five blocking fixes from the 2026-07-06 handoff are all correctly implemented, the deployed code now byte-matches the Colab bundles (closing the "stale bundle" failure mode that caused the original pathological results), and the paper honestly quarantines every retracted number. The new July-11/13 tooling (decomposition sweep runner, HPO harness) is well-engineered and respects all the hard-won invariants — but it contains **4 issues that should be fixed before the GPU campaign runs**, one of which is the same anti-conservative class of bug this project has already retracted results over once.

Scorecard: handoff fixes **5/5 implemented correctly** (one cosmetic remnant). New tooling: **sound core, 4 pre-run fixes needed**. Documentation honesty: **excellent**. Reproducibility hygiene (pinned hashes, byte-matched bundles, deterministic seeds): **verified, excellent**.

---

## 1. What was verified correct

### 1.1 The five 2026-07-06 handoff fixes (all confirmed in source)

| Fix | File | Status |
|---|---|---|
| 1. Sample-split holdout threshold selection | `src/dp_audit_tightness/privacy/empirical.py:308-466` | **Correct.** Opt-in via `threshold_selection="holdout"`; in-sample path untouched and default; selection on one half, Wilson certification at the single fixed threshold on the untouched half (coverage-valid); `<4`-per-half → conservative zero; seeded (`20260706`); new dataclass fields all defaulted. Matches the spec exactly. |
| 2. OUT-count matching in `raw_lira_scores` | `codex/run_raw_lira_pilot.py:409-490` | **Correct.** Symmetric scorer (`mean(OUT) − target_loss` for both branches), zero-OUT points skipped, non-member reference count drawn from the empirical member OUT-count distribution, per-point seeded, gated by `match_out_counts=True`. |
| 3. Verdict/censoring semantics | `codex/run_mnist_saturation.py:104-156`, `run_raw_lira_pilot.py:257` | **Correct.** `<2` non-censored valid cells → "NO VERDICT", never the STILL-CLIMBING fallback; censored cells excluded from the plateau ladder; flag renamed to `conservative_zero_below_floor_or_no_signal`. |
| 4. Disjoint query sampler in `main()` | `codex/run_raw_lira_pilot.py:740` | **Correct.** `sample_query_indices_disjoint` used; eval limits raised so its distinct-point requirement is met. |
| 5. Hard PLD backend | `src/dp_audit_tightness/privacy/pld_accounting.py:44, 85-96` | **Correct.** `backend="google"` is the default and raises if `dp_accounting` is missing; the anti-conservative GDP-CLT fallback can no longer be silently recorded as "PLD". Loud warning on the opt-in `"auto"` path. |

- **Tests:** `tests/test_empirical.py` contains all specified holdout tests (null coverage over 200 reps, signal sanity, determinism, tiny/odd inputs, in-sample-is-default). `.pytest_cache` shows the suite was exercised on 2026-07-11.
- **`empirical.py` integrity:** full 745 lines on disk — the OneDrive truncation blocker from VALIDATION_2026-07-06 §8.3 has resolved. All reviewed files compile (`py_compile` clean).

### 1.2 Bundle reproducibility (independently re-verified today)

- `saturation_bundle_v4.zip`, `hpo_bundle_v1.zip`, `decomposition_bundle_v1.zip` — **sha256 hashes match the values pinned in RUN_QUEUE_2026-07-11.md exactly.**
- Unzipped `decomposition_bundle_v1.zip` and `saturation_bundle_v4.zip` and diffed the load-bearing files (`empirical.py`, `pld_accounting.py`, both runners) against the repo: **byte-identical.** The deployed-code ≠ repo-code failure mode that produced the original pathological σ-sweep is closed.

### 1.3 The new decomposition runner (`codex/run_decomposition_sweep.py`) — invariants wired correctly

Verified by sub-review and spot-checked: passive and canary ε_lower both use `threshold_selection="holdout"` (in-sample kept only under `*_insample` diagnostic keys); PLD is google-or-crash (`enforce_accounting_invariant`); LiRA is the symmetric OUT-matched scorer with `score_direction="higher"`; queries are disjoint across audit seeds; seeds are deterministic end-to-end; smoke budgets correctly land below the Wilson floor and get flagged `exploratory`. The telescoping share algebra is correct (max deviation ~4e-12 on 200k synthetic rows).

### 1.4 The sigma solver and HPO harness — core design sound

`sigma_solver.py` inverts the true PLD (hard google backend), with defensible bracketing/monotonicity checks and a ±1e-3 tolerance enforced at solve time. `tune_hyperparams.py` tunes on **utility only** (Optuna objective = eval accuracy; imports nothing from `auditing/` or `privacy/empirical`), uses deterministic per-trial seeds, independent per-ε studies, and does not touch the protected files.

### 1.5 Scientific honesty

`paper/main.tex:234` carries an explicit draftnote quarantining all pre-2026-07-06 ε_lower values and superseding the retracted "93% accounting share" figure with the corrected ~21%/55–60% numbers. findings/validation docs consistently forbid quoting pre-fix numbers. This is exemplary handling of a retraction.

---

## 2. Issues found (new — not previously documented)

### Blocking before the GPU campaign / before quoting Table 1

1. **`experiments/aggregate_results.py` (~lines 106-132): GDP columns in `decomposition_table1` are not filtered on `estimator_valid`.** The filter is only `quality_flag == "ok"` + non-null ε_lower, but `quality_flag` reflects the *Wilson* bound; a pathological GDP estimate exceeding the PLD upper bound would be averaged into the table as a lower bound, and `tightness_*_gdp_pld` can exceed 1. This is the same anti-conservative class of bug already retracted once. Filter on `estimator_valid` before this table is ever quoted.

2. **Accountant inconsistency: solver targets PLD, downstream tightness divides by RDP.** `sigma_solver` holds ε_PLD = target, but `run_audit_canary.py:89`, `run_audit_passive.py:92`, and `evaluation/gap_decomposition.py` compute tightness against `epsilon_upper_theory` (RDP). A "target ε=1.0" cell will report an ε_upper ≠ 1.0 that varies across cells with the HPO-chosen (batch, epochs), since the RDP–PLD gap is (q,T)-dependent. Pick one accountant for the denominators (PLD is the natural choice) before running the ε-grid.

3. **HPO objective / passive non-member pool coupling.** The Optuna objective maximizes accuracy on `bundle.eval_dataset` — the same split (same `split_seed`) the passive auditor uses as its non-member pool. Selecting hyperparameters that maximize accuracy on the future non-member pool biases ε_lower downward for tuned configs. Needs at least a disclosed limitation; better, a three-way split.

4. **Censored zeros enter the share arithmetic as literal 0.0** (`run_decomposition_sweep.py:465,471` via `or 0.0`), and `aggregate_shares` averages those rows into cell means with no censoring exclusion. This violates the project's own "censored ≠ 0" rule (fix #3's spirit) at the decomposition level — exactly where the paper's Table 1 comes from.

### Major (hardening — cheap to fix, real failure modes)

5. **`utils/results.py::save_json` is not atomic** (in-place `open("w")`). `wide_rows.json` — the sole resume state for the decomposition sweep — is rewritten after every row; a Colab disconnect mid-write corrupts it and loses all resume state. Use write-temp + `os.replace`.
6. **E2 checkpoint reuse is keyed only on (experiment_name, seed, split_seed)** — no hyperparameter check; a re-run of E1 with a different winning config silently reuses stale target checkpoints against fresh shadows. The ε-vs-target mismatch check exists but only logs; it is not persisted into any output row.
7. **`experiments/colab_drive_checkpoint.py`: upload matches Drive files by name only** across the whole corpus (including shared files) and overwrites the match; download streams directly into the final path (truncation on failure corrupts the Optuna SQLite DB). Pin a fileId / scope to a folder; download via temp + rename.
8. **No post-training ε gate:** tuned YAMLs freeze `sampling_rate`, and `dp_sgd.py` prefers the config value; later edits to batch size/epochs/dataset silently produce a wrong-but-"on-target" reported ε. Store `target_epsilon` machine-readably and assert achieved ε_PLD ≈ target after final training.
9. **The "G2 telescoping gate" is a tautology** — the shares sum to 1 identically for *any* inputs (the algebra cancels by construction), so "G2 passed" carries no information about run quality. Don't cite it as evidence; the real quality signals are `validity`/`quality_flag`/`gdp_valid_lower_bound`.

### Minor

10. `pld_accounting.py::_compute_analytical_gaussian` docstring still says the fallback is "comparable to (though slightly looser than) exact PLD" — the exact mislabel the handoff's Fix 5 ordered removed (it's anti-conservative at low σ). The module- and API-level docs are corrected; this inner docstring was missed. Also `compute_epsilon_pld`'s parameter docs still call `"auto"` the default.
11. Canary track records the *target* model's ε_upper, but the canary retrain uses an augmented dataset (different q, steps) — ~1-2% mismatch in the validity gate's bound.
12. `tests/test_sigma_solver.py` never asserts the mock was called with `backend="google"` — a regression to `"auto"` would pass the suite.
13. Two artifact-based tests in `test_empirical.py` glob `results/substantive_smoke_3seed/` and `results/canary_seedfix_3seed/`, which no longer exist → they pass vacuously.
14. `tidy_rows` double-counts audits (2 rows each) in `num_censored`/`num_invalid`; `quality_flag_for` checks `exploratory` before `censored`, undercounting censored smoke rows.
15. MNIST tuned configs restore `training_seeds=None` (base YAML has no seed list), contradicting the docstring's "123/124/125" promise — final MNIST runs would be single-seed.
16. Solver tie-breaking can return achieved ε up to target + 1e-3 (should break toward more noise for a guarantee).
17. Housekeeping: `autoresearch/` is deleted from the working tree but CLAUDE.md still documents it as current; stray file `notebooks/ziDAofCC`; duplicate `...pptx.pptx`; July work is entirely uncommitted (git history ends 2026-04-13) — commit it.

---

## 3. Project state (facts, not judgment)

- **None of the planned experiments (E1–E7) have run yet.** `results/hpo/`, `results/decomposition/`, and the codex sweep result dirs are empty/absent — consistent with RUN_QUEUE_2026-07-11's own status. Every ε_lower currently in the paper is pre-fix and flagged as such.
- The only post-fix GPU run is the K=512 saturation ladder (2026-07-06): conservative ε_lower stayed on the detection floor at every K → the saturation claim is currently **not licensed by data**, as VALIDATION_2026-07-06 §7 correctly concluded.
- Both original headline numbers (93% accounting share; 44.3% canary tightness) are retracted/quarantined. The paper's surviving contribution is the decomposition **protocol** + the measured estimator detection floor — which the validation docs correctly identify.

---

## 4. Recommended order of operations

1. Fix issues **1–4** (half a day of code changes; none touch protected files).
2. Apply hardening **5–9** before any long Colab session.
3. Commit everything (17).
4. Run the queue as ordered in RUN_QUEUE_2026-07-11 (E1+E2 → E3/E4 → sigma sweep v3 → matched canary v2).
5. Only then regenerate paper tables — with the estimator_valid filter in place.

*Written by Claude Fable 5. Findings 1–9 were each verified against source lines before inclusion; sub-review agent reports are summarized, not pasted.*
