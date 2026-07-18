# PROMPT DRAFT — send back to Claude when ready
*(Every factual claim below was validated against the repo on 2026-07-05. Validation notes in italics after each section — delete them before sending, or leave them, they don't hurt.)*

---

You are my research assistant on the DP audit tightness project (EECE_608 folder). Prepare the two corrected GPU experiment packages so I can run them on Colab today. Do not modify the protected files (`privacy/empirical.py`, `auditing/canary/seeding.py`, `utils/validation.py`). Work only in `codex/` and `notebooks/`.

## Task A — Finalize the saturation rerun (extend v2)

`notebooks/saturation_bundle_v2.zip` already contains the corrected symmetric scorer (`score = mean(OUT-shadow loss) − target loss`, `score_direction="higher"` at both call sites of `run_raw_lira_pilot.py`), the validity gate (`invalid_exceeds_upper_bound`), and GDP columns. It has never been executed — no results exist on disk. Update it as follows and rebuild the bundle + notebook as **v3**:

1. Extend `K_LADDER` from max 256 to `[..., 256, 512]` in `codex/run_mnist_saturation.py` (keep 1024 commented with a note on expected extra runtime), since the 2 Jul run's verdict was STILL CLIMBING at K=256 (+0.082 nats on the last doubling, threshold 0.05).
2. Fix the non-member pool: currently `EVAL_LIMIT = 512` while budget is 512 × 5 seeds → the 5 seeds resample from only 512 distinct eval points, so pooled non-member counts are not independent and the Wilson CI is anti-conservative. Raise `EVAL_LIMIT` to at least 5 × 256 = 1280 distinct points (MNIST eval has 10k, so use 2560 for margin) and make the per-seed eval draws disjoint.
3. Add a first notebook cell that runs the sanity checks before anything trains: the bundle contains no `tests/`, so embed the seven-check validation from `research/validate_estimator_scorer_20260705.py` (pure Python, seconds) and abort if any check fails.
4. In the summary output, add a `censored` flag: any cell where `epsilon_lower_conservative == 0` AND the point estimate or GDP estimate is positive must be labeled `censored_below_detection_floor`, not treated as "attack recovers 0%". Justification: verified today that an oracle attack extracting all true leakage at ε = 0.77 is certified as 0 by Wilson-conservative in 5/6 seeds at n = 640–1280.
5. Keep the plateau verdict computed on valid, non-censored cells only.

## Task B — Rebuild the σ-sweep (the paper's main table)

The original sweep notebook (`colab_sigma_sweep.ipynb`, git 4ca8e88, deleted from the working tree) scored members as `mean(OUT) − mean(IN)` without ever querying the target model — the asymmetric bug. All ε_lower values in the findings.md table are untrustworthy. Build a new self-contained `notebooks/colab_sigma_sweep_v2.ipynb` + bundle that:

1. Imports the corrected scorer from `codex/run_raw_lira_pilot.py` as a module (same pattern `run_mnist_saturation.py` uses) — do NOT reimplement LiRA scoring.
2. Same design otherwise: full MNIST, 1 epoch, σ ∈ {0.5, 0.8, 1.1, 1.5, 2.0, 3.0, 5.0}, K = 32 shadows trained once per σ, 5 audit seeds.
3. Raise the query budget from 256 to 2048 per seed (pooled ≈ 5120/5120) — at the old budget the Wilson floor sits above every ε_PLD ≤ 1.0 in the grid, making sub-1-ε cells structurally uncertifiable. Draw non-members from disjoint eval slices per seed.
4. Report three estimators per cell: Wilson-conservative, point estimate, GDP (diagnostic only — never as the headline bound, per the 2 Jul code audit). Apply the same validity gate and `censored` flag as Task A.
5. Regenerate the decomposition table (ε_RDP, ε_PLD, ε_lower, accounting share, audit share) in the same format as findings.md so the old and new tables diff cleanly. Note in a comment: the σ=0.5 accounting share (~93%) is expected to survive (verified insensitive to ε_lower); σ≈2.0 shares are sensitive and were the reason for the rerun.

## Deliverables
- `notebooks/saturation_bundle_v3.zip` + `notebooks/colab_saturation_k512_v3.ipynb`
- `notebooks/sigma_sweep_bundle_v2.zip` + `notebooks/colab_sigma_sweep_v2.ipynb`
- Both notebooks: upload-cell → pip cell (`opacus dp-accounting pyyaml scipy`) → sanity-check cell → run cell → results table/plot cell → zip-and-download cell.
- A short RUNBOOK.md: expected runtime per stage on a T4, what "good" output looks like, and the three abort conditions (sanity check fails, pathological cell appears, CUDA unavailable).

Compile-check every Python file you touch (`python -m py_compile`) and dry-run the notebook JSON for validity before delivering. You cannot run torch here — say so explicitly in the runbook and list exactly what remains unverified until Colab.

---

*Validation notes (checked 2026-07-05):*
- *Bundle v2 scorer/direction/gate/GDP claims: verified by inspecting the zip contents directly (symmetric formula present, `score_direction="higher"` at lines 660 and 703, `invalid_exceeds_upper_bound` and `epsilon_lower_gdp` in run_mnist_saturation.py).*
- *v2 never executed: `codex/results/` contains only April/May outputs; no `mnist_saturation/` directory exists.*
- *K_LADDER=[2..256], QUERY_BUDGET=512, AUDIT_SEEDS=5, TRAIN_LIMIT=2048, EVAL_LIMIT=512: read from the zipped `run_mnist_saturation.py`.*
- *EVAL_LIMIT=512 overlap issue: follows arithmetically from budget 512 (→256 eval draws/seed × 5 seeds from 512 points).*
- *Asymmetric scorer in original σ-sweep: verified in git blob 4ca8e88.*
- *Oracle-at-ε=0.77 → 0 floor result: measured today against the actual `empirical.py` (see VALIDATION_2026-07-05.md §5).*
- *No tests/ in bundle, no sanity cell in v2 notebook: verified from zip listing and notebook JSON.*
- *"STILL CLIMBING, +0.082 nats": from `research/saturation_run_2026-07-02.md`.*
- *One judgment call, not a verified fact: budget 2048/seed for the σ-sweep. It fits MNIST's data (needs 5×1024=5120 distinct members of 60k and 5120 eval of 10k) but the floor at n=5120 was not measured today (the test timed out) — it will certify roughly the ε≳0.3–0.5 region, still not σ=5.0's ε_PLD=0.04. Sub-floor cells stay censored; that's inherent to Wilson, not fixable by this prompt.*
