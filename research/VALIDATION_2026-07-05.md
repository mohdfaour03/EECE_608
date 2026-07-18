# Project Validation — 2026-07-05

**Question answered:** Is the project sincerely valid, real, novel, and working?
**Short verdict:** **Novel: yes (confirmed, fresh check today). Valid: mostly — with one serious evidence problem found and quantified below. Working: estimator core verified correct today; the σ-sweep results were produced with the buggy scorer and need one rerun.**

---

## 1. What I read (fresh literature check, today)

Two targeted searches over arXiv/alphaXiv (2.5M papers) against the exact claims the paper makes, focusing on anything published since the 2 Jul novelty assessment. The two 2026 papers that could plausibly scoop us were read in depth:

- **Wang et al. 2026, "Rethinking the Security of DP-SGD" (arXiv:2605.15648, May 2026).** Shows the SGM formalization used by Opacus-style DP-SGD ignores the gradient-normalization step (EASGM/ASGM/FEASGM); audited leakage can *exceed* the claimed SGM bound in high dimensions / small datasets. **Not a decomposition paper** — it attacks the soundness of ε_upper itself. *Obligation:* cite it; it means our ε_upper (Opacus, loss_reduction=mean → FEASGM) is itself contested at small dataset sizes — one hedging sentence in the validity section covers us, and our small-data saturation runs (N=2048) are exactly the regime they flag.
- **Ertan & van Dijk 2026, "Fundamental Limitations of Favorable Privacy-Utility Guarantees for DP-SGD" (arXiv:2601.10237, Jan 2026).** Pure f-DP theory: lower bound on trade-off separation for shuffled/Poisson DP-SGD. No empirical decomposition, no saturation. Cite in related work only.
- Also surfaced, none scooping: **Cebere et al. 2026** grey-box library auditing (arXiv:2602.17454 — this is the correct "Cebere" attribution target for the paper), **Revisiting LiRA under realistic assumptions** (arXiv:2603.07567 — cite when defending Raw LiRA choices), SeMI sequential MIA (2602.16596), AutoMIA (2604.01014), metagradient canaries (2507.15836), quantile-regression one-run auditing (2506.15349).

**Novelty verdict (re-confirmed today):** no paper performs a controlled same-runs decomposition of the audit gap into accounting (RDP vs PLD) / threat-model (canary vs passive) / estimator (Wilson vs GDP) components, and none uses auditor saturation as an attribution criterion. The 2 Jul assessment stands: **the decomposition protocol is the paper; "framework" phrasing stays dead; σ/T-dependence and epoch-effect findings must be cited as known (Annamalai-DC 2024, hidden-state line), with our contribution being quantitative attribution.**

## 2. What I ran today (two code validations)

**(a) σ-sweep scorer provenance — FAILED provenance check.** Recovered the exact notebook that produced the paper's main decomposition table (`colab_sigma_sweep.ipynb`, git 4ca8e88, 5 Apr) and inspected its scoring code. It uses the **asymmetric buggy scorer**: members scored as `mean(OUT shadows) − mean(IN shadows)` (never queries the target model), non-members as `mean(shadows) − target_loss`. This is precisely the bug the 2 Jul code audit identified as the root cause of pathological cells. **Every ε_lower in the findings.md σ-sweep table is untrustworthy**, and the σ=5.0 anomaly (ε_lower > ε_PLD) is now fully explained.

**(b) Estimator + corrected-scorer logic — ALL 7 CHECKS PASS** (`research/validate_estimator_scorer_20260705.py`, pure Python, run locally against the current working tree):

| Check | Result |
|---|---|
| No membership signal → ε_lower = 0 | PASS (0.000) |
| Real signal + direction="higher" → ε > 0 | PASS (4.18) |
| Real signal + direction="lower" → ε = 0 (direction bug reproduces) | PASS (0.000) |
| **Buggy asymmetric scorer, ZERO signal → spurious ε** | **PASS (ε = 0.24 from pure noise)** |
| Fixed symmetric scorer, zero signal → ε = 0 | PASS (0.000) |
| Symmetric scorer + real signal → ε > 0 | PASS (3.23) |
| Validity gate fires when ε_lower > ε_upper | PASS |

The estimator core (`privacy/empirical.py`) and the corrected scorer are **verified correct**. The buggy scorer demonstrably manufactures ε from nothing — which is both the problem and a publishable cautionary result (the "matched-canary artifact" finding generalizes: *scorer asymmetry silently inflates audits*).

## 3. Damage assessment: does the buggy sweep kill the main result?

No — the **accounting-share conclusion is robust** because it barely depends on ε_lower (which is tiny relative to the accounting gap). Sensitivity check run today, varying ε_lower from 0 to 2× reported:

- σ=0.5: accounting share 92.7–93.4% (headline "93%" survives any plausible ε_lower)
- σ=1.1: 74.0–78.7% (survives)
- σ=2.0: 47.4–69.2% (**sensitive** — the mid-σ crossover point needs the rerun before being quoted)

The *tightness ratios* and all ε_lower values, however, must be regenerated with the fixed scorer before appearing in the paper.

## 4. Verdict and required actions

- **Novel:** yes — confirmed against literature through today. The unclaimed territory is exactly the decomposition protocol + saturation criterion.
- **Valid/real:** the estimator machinery is sound (verified). The novelty positioning is honest. But two headline claims currently rest on zero clean data: the saturation criterion (K-ladder ran on the stale buggy bundle; v2 never executed) and the σ-sweep ε_lower values.
- **Working:** yes at the code level, as of the 2 Jul fixes — verified independently today.

**Blocking actions, in order:**
1. Run `colab_saturation_k256_v2.ipynb` (fixed bundle) — licenses or reframes the saturation claim.
2. Rerun the σ-sweep with the corrected symmetric scorer (port the fixed `raw_lira_scores` from `codex/run_raw_lira_pilot.py` into the sweep) — regenerates the main table.
3. Then freeze results and write. Add Wang et al. 2026 + Cebere et al. 2026 (arXiv:2602.17454) + LiRA-revisited 2026 to related work.

---

## 5. Addendum (same day): the zero-ε_lower floor — quantified

Follow-up on the observation that sweeps kept yielding ε_lower = 0. There are two distinct zero mechanisms, and both were verified today:

**(a) Artifact zeros (bug, already fixed).** With `align_event_to_score_direction=True`, a wrong `score_direction` negates all scores, no member-favoring threshold exists, and the estimator returns exactly 0 with `no_member_favoring_event_found`. This is the 2 Jul direction bug; it would have zeroed every cell. Verified reproducible today (T2b). Fixed at both call sites.

**(b) Honest-but-uninformative zeros (NOT a bug — a fundamental estimator floor, and it is large).** Detection-floor experiment run today (`Gaussian scores, μ-GDP ground truth, Wilson-conservative at the project's exact settings`):

| true ε (δ=1e-5) | certified ε_lower @ n=640 | @ n=1280 |
|---|---|---|
| 0.34 | 0.000 | 0.000 |
| 0.73 | 0.000 | 0.000 |
| 1.13 | 0.151 | 0.042 |
| 1.99 | 0.319 | 0.286 |

**Killer datapoint:** an ORACLE attack extracting 100% of the true leakage at ε = 0.77 (the paper's ε_upper at σ=1.1) is certified as **ε_lower = 0 in 5 of 6 seeds** at n=640–1280. The Wilson-conservative ceiling under *perfect* separation is ~5.1 (n=640) — but its floor swallows the entire sub-1-ε regime at these budgets.

**Consequences for the results:**
1. Every "0% conservative" cell (all simple passive scores, reference-model attack, K=16 saturation cell) is **uninterpretable as attack weakness** — the estimator cannot certify anything below ε≈1 at budget 640–1280 even for a perfect attack. Zeros must be reported as *censored* ("below detection floor"), never as "attack recovers 0%".
2. This does NOT invalidate the paper — it **is** the estimator-gap component of the decomposition, now measured. In the sub-1-ε regime the estimator gap dominates everything else at these budgets. That's a quotable, novel-flavored quantitative result.
3. It strengthens the case for the planned GDP/density estimator (which certifies from AUC, not tail counts) and/or budget scaling. Fix options for the rerun: raise budget (floor drops roughly with log of tail counts), report GDP alongside Wilson, and label sub-floor cells as censored.
4. Wariness note: near the floor the estimate is unstable in the seed (0.79 vs 0.03 at same true ε across seeds) — single-seed values near the floor should never be quoted.
