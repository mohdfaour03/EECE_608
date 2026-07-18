# Handoff: implement audit-estimator fixes (2026-07-06 review)

Context: independent code review on 2026-07-06 (session 2) confirmed the symmetric scorer and
disjoint sampling fixes are correctly deployed, but found the issues below. You are asked to
implement the fixes. Read `research/VALIDATION_2026-07-05.md` and `research/VALIDATION_2026-07-06.md`
first — they are the ground truth on project state.

Decision already made by Mohamad: **Fix #1 = sample-split holdout** (not Bonferroni-grid,
keep Wilson intervals for now).

Ground rules
- `src/dp_audit_tightness/privacy/empirical.py` is on the "do not change without care" list.
  Make ADDITIVE changes only: the existing in-sample path must remain byte-for-byte reachable
  and default, all existing tests must still pass, and every new behavior must be opt-in via a
  new keyword argument.
- Everything must stay deterministic and seeded. No new hyperparameters that require tuning.
- After coding, rebuild the Colab bundle(s) so deployed code == repo code, and extend the
  pre-flight sanity checks to cover the new paths.

---

## Fix 1 (blocking): sample-split holdout in `empirical.py`

Problem: `_estimate_member_aligned_threshold_sweep` selects the threshold by maximizing the
point ε over ALL unique scores, then reports the Wilson lower bound computed on the SAME
sample. Maximizing over thousands of dependent candidates voids 95% coverage — the
"conservative" bound is anti-conservative by construction.

Implementation:
1. Add keyword `threshold_selection: str = "in_sample"` to `estimate_empirical_lower_bound`
   (values: `"in_sample"` = current behavior, `"holdout"` = new path), plus
   `holdout_split_seed: int = 20260706` and `holdout_fraction: float = 0.5`.
2. Holdout path:
   a. Deterministically shuffle member and nonmember score lists with
      `random.Random(holdout_split_seed)` (independent shuffles), split each into
      SELECTION half and ESTIMATION half per `holdout_fraction`.
   b. Run the existing member-aligned sweep (with `require_member_favoring`) on the
      SELECTION half only, choosing the threshold that maximizes the point ε there.
   c. On the ESTIMATION half, evaluate ONLY that single fixed threshold: compute
      member/nonmember success counts, Wilson intervals (keep z=1.96: one-sided 97.5%
      per rate, joint ≥95% by union bound), and
      ε_lower = log((TPR_lo − δ) / FPR_hi) via the existing `_evaluate_member_aligned_threshold`.
      Because the threshold is fixed w.r.t. the estimation half, coverage is valid.
   d. If the selection half finds no member-favoring threshold, return the existing
      conservative-zero result (`no_member_favoring_event_found=True`).
   e. New `estimation_method` string, e.g. `"threshold_sweep_holdout_member_aligned"`.
      Populate the estimate dataclass with the ESTIMATION-half tpr/fpr/counts and record the
      selection-half threshold. Add fields (with defaults, so the dataclass stays
      backward-compatible): `selection_half_sizes`, `estimation_half_sizes`,
      `threshold_selection`.
3. Point estimate reporting: keep reporting the estimation-half point ε alongside the
   conservative value. Do NOT report the selection-half ε anywhere except diagnostics.
4. Callers: switch `build_result_row` in `codex/run_raw_lira_pilot.py` to
   `threshold_selection="holdout"` (this propagates to the saturation and sigma-sweep
   runners, which reuse it). Keep the in-sample value in the row too, under
   `epsilon_lower_conservative_insample`, so old and new tables diff cleanly.
5. Tests (add to `tests/test_empirical.py`, do not modify existing tests):
   - Null coverage: 200 repetitions of pure-noise member/nonmember scores (n=640 each,
     varying seeds) → holdout conservative ε must be 0 in ≥95% of reps (expect ~100%).
     Also assert the in-sample path violates this (documents why the fix exists).
   - Signal sanity: strong-signal Gaussian case → holdout conservative ε > 0 and below the
     in-sample point estimate.
   - Determinism: same inputs + seed → identical output.
   - Odd-length inputs and tiny inputs (n<4 per half → return conservative zero with a
     warning, never crash).

## Fix 2 (blocking for small-K interpretation): OUT-count matching in `raw_lira_scores`

Problem: member OUT-reference averages ~K/2 shadows (as few as 1), nonmember reference
averages all K → member scores have ~2× reference-noise variance under the null → tails
alone can fake TPR>FPR. Severe at K≤8; plausibly explains point estimates 0.97 (K=2) and
1.87 (K=4) in the 2026-07-06 saturation run.

Implementation (in `codex/run_raw_lira_pilot.py`, function `raw_lira_scores`):
- For each NONMEMBER query point, do not average all k shadows. Instead draw a count
  c ~ the empirical distribution of member OUT-counts at this k (simplest faithful version:
  collect the multiset of member OUT-counts actually observed at this k, and for each
  nonmember sample one count from it with `random.Random(matching_seed + pos)`), then average
  c randomly-chosen shadows (same rng). This equalizes reference-noise variance between the
  two branches.
- Keyword-gate it: `match_out_counts: bool = True` with the old behavior available for
  diff runs.
- Add a null test mirroring T3b of `research/validate_estimator_scorer_20260705.py` but at
  K ∈ {2, 4, 8}: synthetic zero-signal shadows, member ref = 1–4 shadows vs nonmember
  ref matched → ε must be ~0. Also run the UNmatched variant and record that it inflates —
  that asymmetry finding is quotable in the paper.

## Fix 3: censoring semantics + saturation verdict default

- In `_saturation_verdict` (`codex/run_mnist_saturation.py`): when the filtered ladder has
  <2 rows, the verdict must be
  `"NO VERDICT -- fewer than 2 non-censored valid cells; cannot assess plateau"` —
  never the STILL CLIMBING string (that is currently a fallback masquerading as a finding).
- In `apply_quality_flags`: rename the flag value to
  `"conservative_zero_below_floor_or_no_signal"` (or add a second field) so a conservative 0
  with a positive point estimate is not asserted to be "below detection floor" — the point
  estimate is almost always > 0 from noise, so the current label presumes signal exists.

## Fix 4 (one line): stale entry point

`run_raw_lira_pilot.main()` still calls the overlapping `sample_query_indices`. Switch to
`sample_query_indices_disjoint` so rerunning the pilot cannot reproduce the anti-conservative
pooling bug. Note: MNIST pilot eval_limit=512 < 5·256=1280 needed → also raise the pilot's
eval limits or lower its QUERY_BUDGET so the disjoint sampler's ValueError doesn't fire.

## Fix 5 (carry-over from VALIDATION_2026-07-06 §6.3, still open)

In `src/dp_audit_tightness/privacy/pld_accounting.py`: make `backend="google"` the hard
default and RAISE if `dp_accounting` is missing instead of silently falling back to the
analytical GDP-CLT approximation; fix the docstring that calls the fallback "slightly looser"
(it is anti-conservative at low σ — this mislabeling is what invalidated Finding 1).

## Verification before declaring done

1. All existing tests pass unmodified; new tests pass.
2. Re-run `research/validate_estimator_scorer_20260705.py` — all 7 checks still PASS.
3. Run the null-coverage simulation for the holdout path and paste its output into a new
   `research/VALIDATION_<date>.md` section.
4. Rebuild `notebooks/saturation_bundle_v4.zip` (+ sigma sweep bundle) from the fixed tree;
   verify the bundle's `empirical.py`, `gdp_estimation.py`, and both runners byte-match the
   repo; extend the notebook pre-flight checks with: (a) holdout path active,
   (b) OUT-count matching active, (c) hard-PLD backend assertion.
5. Do NOT reinterpret old numbers: all published tables must be regenerated on GPU with the
   fixed bundle before any ε_lower is quoted.

## Explicitly out of scope

- No changes to `auditing/canary/seeding.py` or `utils/validation.py`.
- Do not replace Wilson with Clopper–Pearson in this pass (Mohamad chose split+Wilson);
  leave a TODO noting CP would make "conservative" exact.
- Do not touch the legacy `_estimate_legacy_threshold_sweep` path.
