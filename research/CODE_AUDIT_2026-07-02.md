# Code Audit — 2026-07-02

Full logic/validity pass over the load-bearing code, triggered by the pathological
(eps_lower > eps_upper) cells in the first saturation run.

## Root cause of the pathological cells

**Not** the estimator, and **not** a subtle statistics bug — a stale bundle plus a
direction mismatch:

1. **Stale bundle (primary).** The bundle uploaded to Colab contained the OLD
   `raw_lira_scores`, which scored members and non-members with *different*
   formulas:
   - members: `mean(OUT shadow loss) - mean(IN shadow loss)` — never queries the
     target model at all;
   - non-members: `mean(shadow loss) - target loss`.
   Two different quantities on two different scales → the score distributions are
   separable by construction, independent of real membership. Perfect artificial
   separation → the estimator (correctly, given its inputs) returns a large
   eps_lower that can exceed eps_upper. The working-tree scorer had already been
   corrected to the symmetric form `score = mean(OUT) - target_loss` (identical
   for members and non-members), but that fix wasn't in the zip that ran.

2. **Direction bug (secondary, live).** With the corrected symmetric scorer,
   MEMBERS score HIGHER, but both call sites passed `score_direction="lower"`.
   Verified empirically: on well-separated synthetic data, `"lower"` returns
   eps_lower = 0 (no member-favoring event), `"higher"` returns the correct
   bound. Left unfixed, `"lower"` would have zeroed every cell. **Fixed** at
   `run_raw_lira_pilot.py:660,703` and `run_mnist_saturation.py`.

## Fixes applied

- `run_raw_lira_pilot.py`: confirmed corrected symmetric scorer (skip points with
  zero OUT shadows — removes the small-K fallback artifact); `score_direction`
  → `"higher"` at both call sites.
- `run_mnist_saturation.py`: `score_direction="higher"`; added a **validity gate**
  that labels any cell with eps_lower_conservative > eps_upper_tighter as
  `invalid_exceeds_upper_bound`; the plateau verdict now ignores invalid cells so a
  pathological cell can never drive attribution.
- Rebuilt the run bundle as `saturation_bundle_v2.zip` and notebook
  `colab_saturation_k256_v2.ipynb` (robust upload path, valid-cell-only plots).

## Files audited and verdicts

| File | Verdict |
|---|---|
| `privacy/empirical.py` (protected) | **Sound.** Wilson CI, epsilon = log((TPR-δ)/FPR) with FPR floored, conservative bound uses tpr_lower/fpr_upper (correct direction). Validated: no membership signal → eps_lower = 0. Not the source of the pathology. |
| `privacy/gdp_estimation.py` | **Logic correct** (Mann-Whitney AUC, μ = √2·Φ⁻¹(AUC), GDP→(ε,δ)). Inflated outputs were bad inputs, correctly gated invalid by the runner. Limitation: assumes Gaussian scores → treat as diagnostic, never a reported bound. |
| `auditing/canary/generation.py` + `one_run.py` | Matched-design functions correct and compile; wiring correct. Unit tests need torch (see below). |
| `run_raw_lira_pilot.py` / `run_mnist_saturation.py` / `run_matched_canary_sweep.py` | Fixed as above; all compile. |

## Documented limitations (not bugs — for the paper's validity section)

1. **Estimator selection bias.** The threshold is chosen to maximize the point
   estimate, then a Wilson lower bound is reported at that threshold. This is
   mildly anti-conservative (selection over the grid). Mitigation: large query
   budget (FPR floor ≈ 0.5/n becomes negligible) and the validity gate. A fully
   rigorous alternative (held-out threshold selection or Bonferroni over the grid)
   is future work; `empirical.py` is protected and was left unchanged.
2. **FPR floor.** With FPR = 0 on a finite sample, eps is capped at ~log(2·n_non).
   At genuinely strong separation this can exceed a small eps_upper — the validity
   gate catches these; real *passive* LiRA is weak, so it won't bite in practice.
3. **GDP Gaussianity.** Diagnostic only.

## Not runnable in this sandbox

`torch`/`opacus` could not be installed here (proxy blocks the wheel), so the
torch-dependent tests (`tests/test_matched_canaries.py`, end-to-end training)
were **not** executed locally. All pure-Python logic WAS validated directly
(estimator no-signal→0, direction correctness, synthetic LiRA scorer behavior).
Run `pytest tests/ -q` on Colab as the first cell before the sweep.

## Action for next run

Use `saturation_bundle_v2.zip` + `colab_saturation_k256_v2.ipynb`. Expect: no
pathological cells, small honest eps_lower at low K, and either a plateau or a
clean climbing curve to extrapolate.
