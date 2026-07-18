# RUNBOOK — GPU runs

## 2026-07-11 UPDATE — full run queue assembled; E4 runner + E1/E2 + E4 notebooks added

**Read `research/RUN_QUEUE_2026-07-11.md` first** — it is the single ordered list of every
run still missing from `research/EXPERIMENT_PLAN_2026-07-08.md`, with handoffs and shas.

New this update:

| Run | Notebook | Bundle | Est. T4 time |
|---|---|---|---|
| E1+E2 (HPO + final models, per dataset) | `colab_hpo_e1_e2_v1.ipynb` | `hpo_bundle_v1.zip` | ~3–6 h (per-ε resumable) |
| E4 decomposition (Table 1, per ε-cell) | `colab_decomposition_e4_v1.ipynb` / `kaggle_decomposition_e4_v1.ipynb` | `decomposition_bundle_v1.zip` | ~1–2 h per (dataset, ε) cell |

- E4 runner: `codex/run_decomposition_sweep.py` (holdout-Wilson primary, GDP diagnostic,
  matched canaries, telescoping shares, gates G2/G3 built in, resume after disconnect).
- `saturation_bundle_v4.zip` / `sigma_sweep_bundle_v3.zip` / `matched_canary_bundle_v2.zip`
  were rebuilt to pick up the diabetes loader in `data/datasets.py` (added 2026-07-08 after
  the bundles were zipped). New saturation sha
  `754fe0e98433b95bd74ddee20b1674396c5a4220b516eb5d4210861d1746a564` is re-pinned in both
  saturation notebooks. **Discard any locally downloaded copies of the old zips.**
- `experiments/aggregate_results.py` now folds E4 output into
  `results/summaries/decomposition_table1.{csv,json}` (E7).

## 2026-07-08 — Per-ε hyperparameter optimization (professor feedback)

New pipeline, runs BEFORE the audit sweeps once adopted:

```bash
pip install optuna dp-accounting            # plus the usual torch/opacus
python experiments/tune_hyperparams.py --config configs/hpo/mnist_eps_grid.yaml
python experiments/prepare_diabetes.py --fetch-uci        # one-time dataset build
python experiments/tune_hyperparams.py --config configs/hpo/diabetes_eps_grid.yaml
```

- Grid: ε ∈ {0.3, 0.5, 1, 2, 4, 8}; 30 TPE trials each; σ re-solved per trial by
  `privacy/sigma_solver.py` so eps_PLD == target exactly (google backend, hard).
- Outputs: `results/hpo/<study>.json` (all trials) and
  `results/hpo/best_configs/<study>.yaml` — feed the latter to `run_train.py`
  then the audit runners. Winning configs restore canonical seeds 123/124/125.
- Compute: ~30 trials × (1–15 epochs) per ε cell. MNIST cell ≈ 20–60 min on T4;
  diabetes cell similar (253k rows, MLP). Run `--epsilon 1.0` style single cells
  if a session is short.
- Paper hooks: Preliminaries §"Per-ε hyperparameter optimization" states the
  out-of-band-tuning caveat (Papernot & Steinke 2022) and cites Empirical
  Privacy Variance (Hu et al. 2025). Do not delete those hedges.
- Tests: `tests/test_sigma_solver.py` (7 checks, CPU-only, dependency-injected
  accountant) — all passing 2026-07-08.

## 2026-07-07 UPDATE — v4/v3 packages SUPERSEDE everything below

The 2026-07-05 packages (saturation v3, sigma sweep v2) are **obsolete**: their bundles
predate the 2026-07-06 estimator fixes (holdout threshold selection, OUT-count matching,
hard-PLD backend). Current packages, rebuilt 2026-07-07 and byte-verified against the repo:

| Run | Notebook | Bundle | Est. T4 time |
|---|---|---|---|
| 1 (first) | `colab_saturation_k512_v4.ipynb` | `saturation_bundle_v4.zip` | ~1–1.5 h |
| 2 | `colab_sigma_sweep_v3.ipynb` | `sigma_sweep_bundle_v3.zip` | ~2.5–4.5 h |
| 3 (last) | `colab_matched_canary_sweep_v2.ipynb` | `matched_canary_bundle_v2.zip` | ~2.5–4 h |

### Run 3 — Matched canary sweep v2 (added 2026-07-08)

Rebuilt from the stale 2026-07-02 package: bundle now carries the fixed `empirical.py` /
`pld_accounting.py`, and `estimate_row` in `run_matched_canary_sweep.py` uses the
selection-valid **holdout** Wilson bound as PRIMARY (in-sample kept per row as
`epsilon_lower_conservative_insample`, diagnostic only). Pre-flight cells mirror v4
conventions (bundled 13-test estimator suite + fix markers + true-PLD live check) and
ABORT on a stale upload. Verified locally 2026-07-08: 13/13 estimator tests green on the
staged bundle tree, runner compiles, all runner-consumed estimator fields exercised.

This run regenerates the canary-track numbers (incl. the 44.3%-tightness figure and the
legacy-vs-matched artifact quantification) — it gates only the paper's canary case-study
section, not the decomposition spine. Expectations: matched-design eps_lower <= eps_upper
everywhere; legacy design at sigma in {3.0, 5.0} may exceed eps_upper — that is the
documented appearance artifact, report it as such; holdout conservative values will sit at
or below the old in-sample numbers, and low-support cells may be censored — report
censoring, never "0%". Priority: run AFTER Runs 1–2; the paper can ship without Run 3 by
dropping quantitative canary claims (qualitative pitfall narrative already cites Cebere
et al.). Result file: `matched_canary_results_v2.zip`.

Both saturation/sigma notebooks carry extended pre-flight cells that ABORT unless the uploaded bundle has:
(a) holdout path active and primary in `build_result_row`, (b) `match_out_counts=True`
default in the scorer, (c) `compute_epsilon_pld` hard-defaulting to the google backend
(raises without dp-accounting — the GDP-CLT mislabeling that invalidated Finding 1 can no
longer occur silently), plus the full bundled test suite (13 tests incl. null coverage).

Corrected expectations (per VALIDATION_2026-07-06):
- The old "σ=0.5 accounting share ≈ 93%" prediction is DEAD. With true PLD the accounting
  share peaks ~55–60% at mid-σ and is ~21% at σ=0.5.
- Conservative ε_lower cells at low σ will likely be `conservative_zero_below_floor_or_no_signal`
  — report as censored, never as "attack recovers 0%".
- Saturation verdict may legitimately be `NO VERDICT` (fewer than 2 non-censored cells) or
  `STILL CLIMBING`; per the pre-registered hedge both are publishable outcomes.
- Never quote point estimates: under the pure null they ranged 0.59–3.16 across K.
- Holdout halves the effective sample per estimate; expect conservative bounds ≤ the old
  in-sample values (which remain in the row as `epsilon_lower_conservative_insample` for diffing).

Kaggle instead of Colab: replace the `google.colab` upload cell with a Kaggle dataset
attach (upload the bundle as a private dataset, then `!unzip -o /kaggle/input/<ds>/bundle.zip`),
and enable GPU T4 in notebook settings. Everything else is unchanged.

---

# (SUPERSEDED) RUNBOOK — Colab runs, 2026-07-05 packages

Two independent packages. Run the saturation sweep first (cheaper, and it validates the whole corrected pipeline end-to-end before you commit to the long sweep).

## Package 1 — Saturation v3
**Files:** `colab_saturation_k512_v3.ipynb` + upload `saturation_bundle_v3.zip` when prompted.

| Stage | Expected on T4 |
|---|---|
| pip install | 1–2 min |
| Sanity cell (10 checks) | < 30 s |
| Target model (2048-sample subset, 1 epoch) | < 1 min |
| 512 shadow models (1024 samples each) | ~40–80 min (the long stage; ~5–10 s/shadow) |
| Scoring all 9 K values (shadows reused) | ~2–5 min |
| **Total** | **~1–1.5 h** |

**Good output looks like:** no `invalid_exceeds_upper_bound` cells at K≥32 (the v2 stale-bundle pathology is fixed); small honest ε_lower at low K (possibly `censored`); a rising then flattening curve; verdict either `PLATEAUED` (→ the attribution claim is licensed) or `STILL CLIMBING` (→ extend K_LADDER to 1024, which roughly doubles the shadow stage, or write the honest "gap remains partly weak-auditor at K=512" framing — still publishable per the pre-registration hedge).

## Package 2 — Sigma sweep v2
**Files:** `colab_sigma_sweep_v2.ipynb` + upload `sigma_sweep_bundle_v2.zip` when prompted.

| Stage | Expected on T4 |
|---|---|
| pip + sanity | ~2 min |
| Per σ: target (full 54k MNIST, 1 epoch, batch 256) | ~1–2 min |
| Per σ: 32 shadows (27k samples each) | ~20–35 min |
| Per σ: scoring + 3 estimators (pooled 5120/5120, GDP bootstrap 2000) | ~2–4 min |
| **Total, 7 sigmas** | **~2.5–4.5 h** |

Results are written after **every** σ, so a disconnect loses at most one σ — re-download whatever `codex/results/sigma_sweep_v2/` holds.

**Good output looks like:** ε_RDP/ε_PLD close to (not identical to) the old table — the split here is 54000/6000 vs the old ~57k, so bounds shift slightly but the table is self-consistent; σ=0.5 accounting share ≈ 93% (expected to survive); low-σ ε_lower cells likely `censored_below_detection_floor` (report as "below detection floor", never "0%"); NO cell with ε_lower > ε_PLD outside the validity gate (the old σ=5.0 anomaly should be gone).

## Abort conditions (both notebooks)
1. **Sanity cell fails** — the notebook raises SystemExit; do not proceed, re-download the bundle (a stale upload is exactly what invalidated the 2 Jul run).
2. **Pathological cell** (`invalid_exceeds_upper_bound`) **at K≥32 / any σ** — stop and investigate; the fixed scorer shouldn't produce these except at tiny K where sparse tails are expected and gated.
3. **CUDA unavailable** (pip cell prints "CPU ONLY") — abort; switch Runtime → T4 GPU. CPU runs are 10–20× slower.

## What was verified locally today (no torch in the sandbox)
- Both runners and the pilot compile (`py_compile`).
- Both runners import cleanly from the actual bundles, with the right constants.
- Disjoint sampler: distinctness, reproducibility, small-pool failure — under both configs.
- Quality flags: 4 cases (censored, invalid, honest zero, normal).
- Saturation verdict: censored + invalid cells excluded from the plateau ladder.
- The sanity cell itself executes green against the extracted v3 bundle (10/10).
- Notebook JSON structure + syntax of every code cell.

## What CANNOT be verified until Colab (torch/opacus unavailable here)
- Actual DP-SGD training (target + shadows) and Opacus/dp-accounting versions interop.
- `train_shadow_losses` σ-inheritance producing sensible shadow models at σ≠1.1.
- Real MNIST download inside `load_dataset_bundle`.
- End-to-end runtime estimates above (extrapolated from the 2 Jul T4 session).
- The GDP estimator's behavior on real (non-Gaussian) score distributions — diagnostic only regardless.

## One deliberate deviation from the prompt
`make_train_config` hardcoded σ=1.1/batch=128 into **shadow** configs, so σ-sweep shadows would have been trained at the wrong noise level. Fixed in `run_raw_lira_pilot.py` (`train_shadow_losses` now inherits noise/batch/epochs/clipping from the target config). No-op for the saturation run; essential for the sweep. Protected files untouched.
