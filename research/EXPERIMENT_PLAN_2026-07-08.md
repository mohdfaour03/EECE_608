# EXPERIMENT PLAN — implementation-ready specification (2026-07-08)

**Audience:** a coding model (Opus or similar) implementing runners without further context.
**Read first:** `CLAUDE.md`, `research/VALIDATION_2026-07-06.md`, `notebooks/RUNBOOK.md`.
**Prime directive:** never weaken an invariant in §0 to make an experiment "work."

---

## §0. Global contracts and invariants (apply to EVERY experiment)

**I1 — Estimator.** All conservative ε_lower values come from
`estimate_empirical_lower_bound(..., threshold_selection="holdout")`
(`src/dp_audit_tightness/privacy/empirical.py`). The in-sample value may be stored
only under a key suffixed `_insample` for diffing. Point estimates are stored but
NEVER quoted as findings (null point estimates range 0.59–3.16; see VALIDATION §9).

**I2 — Accounting.** ε_upper is computed twice per run: RDP (Opacus default α-grid) and
PLD via `compute_epsilon_pld(..., backend="google")`. If `dp_accounting` is missing the
run must CRASH, not fall back. Record `pld_accounting_backend` in every result row.

**I3 — Threat models never mix.** Canary (evaluator-controlled) and passive
(observer-only) code paths, configs, and result files stay disjoint. A result row has
exactly one `threat_model` ∈ {"canary", "passive"}.

**I4 — Scoring.** Raw-LiRA scoring uses the SYMMETRIC scorer with
`match_out_counts=True` (`codex/run_raw_lira_pilot.py::raw_lira_scores`), and disjoint
query sampling (`sample_query_indices_disjoint`). Never the legacy asymmetric scorer.

**I5 — Canary design.** Inserted and reference canaries must be exchangeable draws from
ONE generation pool (matched design). The unmatched design may be run ONLY inside E5
(the validity case study), clearly flagged `design="unmatched_legacy"`.

**I6 — Seeds.** Canonical seed triple {123, 124, 125} for (split_seed, training_seed)
pairs of final audited models. HPO trials use 900000+trial_number (MNIST) / 910000+
(diabetes) and never touch canonical seeds. Every stochastic function takes an explicit
seed; no global RNG state may leak between phases (`set_global_seed` at phase start).

**I7 — Records.** Every result row is schema-validated before save
(`utils/validation.py`); JSON + config snapshot per run. Rows carry:
`quality_flag` ∈ {ok, conservative_zero_below_floor_or_no_signal,
invalid_exceeds_upper_bound, exploratory}, and validity gate
`epsilon_lower_conservative <= epsilon_upper_pld` enforced by
`apply_quality_flags` — a violation is a FLAG, never a reported finding.

**I8 — Budgets.** Minimum 1000 member + 1000 nonmember observations per certified bound
(Wilson floor). Anything smaller → row is `exploratory`.

**I9 — Reporting.** Censored ≠ zero: a conservative 0 with the below-floor flag is
reported as "below detection floor," never "attack recovers 0%."

**Quantities.** For accountant A ∈ {RDP, PLD}, threat model θ ∈ {canary, passive},
estimator E ∈ {wilson_holdout, gdp}:
Δε(A,θ,E) = ε_upper^A − ε_lower^{θ,E};  ρ = ε_lower/ε_upper^A.
accounting_share = (ε_upper^RDP − ε_upper^PLD) / Δε(RDP,θ,E).
estimator_share  = (ε_lower^{θ,gdp} − ε_lower^{θ,wilson_holdout}) / Δε(RDP,θ,E), gdp
diagnostic-only when it violates the validity gate.
threat_share     = (ε_lower^{canary,E} − ε_lower^{passive,E}) / Δε(RDP,passive,E).
residual = Δε(PLD,canary,E); attributable beyond auditor family ONLY if E3 verdict
is PLATEAUED (else report "residual ≥ partly weak-auditor").

---

## §E1. Per-ε hyperparameter optimization — IMPLEMENTED, run only

Code: `experiments/tune_hyperparams.py`, `privacy/sigma_solver.py`,
configs `configs/hpo/{mnist,diabetes}_eps_grid.yaml`. Tests green (2026-07-08).

```
for dataset in [mnist, diabetes]:
  for target_eps in [0.3, 0.5, 1.0, 2.0, 4.0, 8.0]:
    study = optuna TPE(seed=20260708), 30 trials
    trial: sample (lr, batch, epochs, clip); sigma = solve_sigma_for_epsilon(target_eps, q=batch/N, T=ceil(N/batch)*epochs, delta)
           train DP-SGD (seed 900000+trial) -> maximize eval accuracy
    save results/hpo/<study>.json + best_configs/<study>.yaml (canonical seeds restored)
```

Acceptance: 12 studies × 30 trials; every best config's achieved ε within 1e-3 of target;
JSON parses; YAML loadable by `load_train_config`.

## §E2. Final audited models

For each `results/hpo/best_configs/*.yaml`: run `experiments/run_train.py --config <yaml>`
(3 canonical seeds). Store checkpoints. Acceptance: 2 datasets × 6 ε × 3 seeds = 36
checkpoints; per-cell accuracy std over seeds < 2%; recorded ε_PLD == target ± 1e-3.

## §E3. Saturation K-ladder (passive Raw-LiRA)

On the ε=1.0 MNIST tuned model (extend to diabetes if budget allows):

```
shadows: train K_max=512 shadow models once (50% subsets, sigma inherited from tuned config)
for K in [2,4,8,16,32,64,128,256,512]:
  scores = raw_lira_scores(K shadows, match_out_counts=True)   # I4
  row(K) = build_result_row(threshold_selection="holdout")      # I1
verdict = _saturation_verdict(filtered ladder)                  # censored/invalid cells excluded
```

Outputs: `codex/results/mnist_saturation/*.json` + ladder CSV + verdict string.
Acceptance: no `invalid_exceeds_upper_bound` at K≥32; verdict ∈ {PLATEAUED,
STILL CLIMBING, NO VERDICT} — all three are publishable outcomes (pre-registered).
Runtime ~1–1.5 h T4 (existing notebook `colab_saturation_k512_v4.ipynb`).

## §E4. Decomposition sweep (the paper's Table 1)

For each (dataset, ε) cell from E2, using the SAME 3 models per cell:

```
upper:   eps_rdp, eps_pld                                       # I2
passive: K=32 shadows, budget>=5120 disjoint queries            # I4, I8
         eps_lower[passive, wilson_holdout], [passive, gdp]
canary:  1000 matched canaries (I5), 5 audit seeds, logit-margin scoring
         eps_lower[canary, wilson_holdout], [canary, gdp]
shares:  accounting_share, estimator_share, threat_share, residual   # §0 formulas
```

Outputs: one row per (dataset, ε, seed, θ, E) → `results/decomposition/rows.csv`;
aggregated shares (mean over seeds, min/max as error bars) → `shares.csv`.
Acceptance: every cell carries all 4 lower bounds + 2 upper bounds; shares sum sanity:
accounting_share + threat_share + estimator_share + residual_share = 1 ± 0.02 per cell
(document any cell where gdp is diagnostic-invalid — exclude estimator_share there and
renormalize, flagging `estimator_share_excluded=true`).
Runtime: dominated by shadows: 6 ε × 32 shadows × 2 datasets. Batch per-ε on Colab
(results written after every ε — disconnect loses one cell).

## §E5. Matched-canary validity case study (paper §Validity)

MNIST, ε=1.0 tuned config, 3 seeds:

```
armA = audit with design="matched"          # I5 pool
armB = audit with design="unmatched_legacy" # positions/intensities differ  (FLAGGED)
report per arm: eps_lower_conservative, quality_flag counts, score AUC between
                inserted-vs-reference distributions BEFORE training (appearance signal)
```

Expected: armB pre-training AUC >> 0.5 (appearance separability), armB may violate the
validity gate (that is the finding); armA pre-training AUC ≈ 0.5 and no violations.
Acceptance: pre-training AUC computed on raw pixel scores; both arms use identical
budgets/seeds; one figure (two score histograms) per arm.

## §E6. Cross-dataset replication

Repeat E3 (K to 128 suffices) + E4 on diabetes. CIFAR-10 only if time: tuned configs
first via a `configs/hpo/cifar10_eps_grid.yaml` clone (epochs_max=30, cnn model),
else declare out of scope in the paper. Acceptance: identical schemas; no
dataset-specific code forks outside `data/datasets.py`.

## §E7. Aggregation + paper artifacts

```
python experiments/aggregate_results.py       # extend to read results/decomposition
figures: tightness-vs-epsilon (per dataset, per θ), shares stacked bars per ε,
         K-ladder with censoring markers, E5 histogram pair
tables:  decomposition Table 1; hyperparameter table (per-ε winning configs, appendix)
```

Every figure script reads ONLY schema-validated JSON/CSV (no notebook state). Numbers
in the paper regenerate from `paper/figures/make_paper_figures.py` — no hand edits.

---

## Execution order and gates

E1 → E2 → {E3, E4, E5 in any order} → E6 → E7.
Gate G1 (after E2): accuracy sanity vs untuned baselines — tuned ≥ untuned per cell.
Gate G2 (after first E4 cell): shares sum sanity passes before launching the full grid.
Gate G3 (before paper numbers freeze): rerun `tests/` (all), `research/validate_estimator_scorer_20260705.py` (7/7), and the E4 pipeline on one smoke cell with 64 queries expecting `exploratory` flags — the pipeline must refuse to certify from tiny budgets.

## Explicitly forbidden

- Quoting any pre-2026-07-06 ε_lower (estimator was anti-conservative).
- The "93% accounting share" figure (mislabeled-PLD artifact; retracted).
- Tuning hyperparameters on audit outcomes (tune on utility ONLY — I6 separation).
- Comparing ε_lower across cells with different budgets without stating budgets.
