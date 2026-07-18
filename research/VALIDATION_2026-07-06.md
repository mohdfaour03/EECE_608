# Project Validation — 2026-07-06 (independent re-run)

> **§8 (later same day): estimator/scorer fixes from HANDOFF_OPUS_FIXES_2026-07-06 are now implemented.** All five fixes are coded in the repo. Fix 1 (holdout) is validated by simulation. Two verification steps — running the *deployed* test suite and rebuilding a byte-verified bundle — are BLOCKED by a OneDrive partial-sync freeze (details in §8.3). See §8 at the bottom.

**Question:** Inspect the project, continue prior work, run the code, and confirm whether the novelty is real and valid.

**Short verdict:** The **methodological novelty holds** (the same-runs decomposition + saturation-criterion positioning is still unclaimed in the literature and the estimator core is sound). But **the flagship *quantitative* result — "accounting choice explains ~93% of the audit gap at σ=0.5" (findings.md Finding 1) — is INVALID.** It is an artifact of mislabeling the analytical GDP-CLT approximation as "PLD." Recomputed with the true PLD accountant (which the current code actually calls), the accounting share at σ=0.5 falls from 93% to ~21%, and the whole Finding-1 story ("switching accountants alone closes almost the entire gap at strong privacy") does not survive.

---

## 1. What I ran (all reproducible in a CPU sandbox, no GPU)

- `research/validate_estimator_scorer_20260705.py` — **all 7 checks PASS** again on the current tree (no-signal→ε=0; direction bug reproduces; buggy asymmetric scorer manufactures ε=0.24 from pure noise; fixed symmetric scorer→0; validity gate fires). The estimator machinery (`privacy/empirical.py`) and corrected scorer are confirmed sound.
- `research/recompute_accounting_gap.py` — **independent recomputation of the accounting bounds** with `dp-accounting` (installed cleanly from PyPI; only pytorch.org's index is proxy-blocked, so DP-SGD *training* still requires Colab/Kaggle GPU — but the accounting decomposition, which is what the flagship claim rests on, needs no torch).

## 2. The critical finding: the "PLD" column is not PLD

**eps_RDP reproduces the paper exactly** (independent `dp_accounting` RDP accountant, Opacus's default α-grid):

| σ | paper eps_rdp | recomputed | match |
|---|---|---|---|
| 0.5 | 6.43 | 6.431 | ✓ |
| 1.1 | 0.77 | 0.772 | ✓ |
| 2.0 | 0.19 | 0.194 | ✓ |

**eps_PLD does NOT.** The paper's `eps_pld` column matches the code's *analytical GDP-CLT fallback* (`mu = q·√T/σ`) to the decimal — not the true PLD accountant:

| σ | paper eps_pld | analytical GDP fallback | **true PLD (google backend)** |
|---|---|---|---|
| 0.5 | 0.47 | 0.4691 | **5.106** |
| 0.8 | 0.28 | 0.2811 | 0.920 |
| 1.1 | 0.20 | 0.1987 | 0.318 |
| 1.5 | 0.14 | 0.1417 | 0.179 |
| 2.0 | 0.10 | 0.1034 | 0.118 |
| 3.0 | 0.07 | 0.0663 | 0.071 |
| 5.0 | 0.04 | 0.0377 | 0.039 |

Confirmed with **two independent PLD paths** (the project's own `from_gaussian_mechanism().self_compose()` and `dp_accounting`'s `PLDAccountant.compose()`) — both give 5.106 at σ=0.5. Since PLD is a *valid, tighter* upper bound it must sit below RDP (6.43); the reported 0.47 sits 13× below RDP, which no valid accountant produces here. The GDP-CLT approximation is anti-conservative (under-estimates ε) in the low-noise / low-composition regime because the CLT hasn't converged at ~211 steps; it only agrees with true PLD at high σ (σ=5.0: 0.038 ≈ 0.039). **The claim is most wrong exactly where it is loudest (low σ).**

## 3. Corrected decomposition (true PLD, v2 split 54000/6000, ε_lower≈0.024)

| σ | RDP | PLD(true) | acct gap | **acct share (corrected)** | acct share (paper) |
|---|---|---|---|---|---|
| 0.5 | 6.52 | 5.19 | 1.34 | **20.6%** | 93% |
| 0.8 | 1.71 | 0.95 | 0.75 | **44.7%** | 84% |
| 1.1 | 0.78 | 0.33 | 0.45 | **59.4%** | 77% |
| 1.5 | 0.37 | 0.18 | 0.19 | **53.6%** | 65% |
| 2.0 | 0.20 | 0.12 | 0.07 | **43.0%** | 56% |
| 3.0 | 0.12 | 0.07 | 0.05 | **49.3%** | 65% |
| 5.0 | 0.05 | 0.04 | 0.005 | **25.0%** | 117% |

The corrected accounting share is **non-monotonic, peaking (~59%) at mid-σ**, not a monotone curve dominated (93%) at strong privacy. Finding 1 as written is false. At σ=0.5 the RDP→PLD tightening is 6.52→5.19 (1.26×), not 14×; most of the gap down to the tiny audited ε_lower is threat-model + estimator-floor, not accounting.

## 4. Why the prior sessions missed this

The 2026-07-05 validation stress-tested the **ε_lower** side (buggy scorer, detection floor) and did a sensitivity check that assumed the ε_upper columns were correct — it never independently recomputed eps_PLD. The RUNBOOK predicted "σ=0.5 accounting share ≈ 93% expected to survive" the v2 rerun; that prediction is **wrong** — the never-executed true-PLD rerun would have overturned Finding 1. The likely mechanism: the original April sweep ran in an environment where `dp_accounting` was unavailable, so `backend="auto"` silently fell through to the analytical Gaussian, and the number was recorded as "PLD."

## 5. Bottom line

- **Novel:** yes — the decomposition-protocol + saturation-criterion methodology is genuinely unclaimed; literature positioning (cite Annamalai-DC 2024 for σ/T trends, the SoK for open questions) is honest and intact.
- **Valid:** the *machinery* is valid (estimator sound, RDP correct, current code calls the true PLD). But the *headline number* that made the decomposition compelling is an accounting artifact and must be pulled.
- **Working:** yes at code level. The DP-SGD training half still needs a GPU run (Colab/Kaggle) — the notebooks are staged for it — but that half does not affect this correction.

## 6. Required actions (supersedes 2026-07-05 action list)

1. **Retract Finding 1's "93% / accounting dominates at strong privacy" framing.** Replace with the corrected table: accounting share peaks ~55–60% at mid-σ; at σ=0.5 it is ~20%.
2. Regenerate every table/figure/slide that cites eps_pld ≤ 0.47 — the Final Presentation PDF and findings.md both carry the artifact.
3. In `pld_accounting.py`, fix the false docstring claim that the analytical fallback is "comparable to (slightly looser than) exact PLD" — it is *dramatically tighter and anti-conservative* at low σ. Make `backend="google"` the hard default and raise (not silently fall through) if `dp_accounting` is missing, so a stale environment can never mislabel GDP-CLT as PLD again.
4. Re-run the σ-sweep on GPU with the fixed scorer AND the true-PLD assertion, then refreeze results before writing.

## 7. Addendum — live GPU saturation run (Colab T4, 2026-07-06)

Ran `colab_saturation_k512_v3.ipynb` + `saturation_bundle_v3.zip` end-to-end on a Colab T4 (~25 min: target model + 512 shadow models + scoring across the K-ladder). Pre-flight: all 10 sanity checks PASS live on GPU (including "bundle has SYMMETRIC scorer" and the buggy-asymmetric-scorer detector), and dp-accounting was installed so `epsilon_upper_tighter` is the **true PLD (1.81)**, not the analytical fallback. No `invalid_exceeds_upper_bound` cells at any K — the validity gate and corrected scorer hold up in a real run.

**K-ladder result (train=2048, eval=2560, δ=1e-5, ε_upper PLD = 1.81):**

| K | ε_lower Wilson (conservative) | ε_lower point | ε_lower GDP |
|---|---|---|---|
| 2 | 0.000 | 0.974 | 0.000 |
| 4 | 0.000 | 1.866 | 0.000 |
| 8 | 0.085 | 0.761 | 0.016 |
| 16 | 0.000 | 0.690 | 0.000 |
| 32 | 0.000 | 0.690 | 0.004 |
| 64 | 0.000 | 0.914 | 0.005 |
| 128 | 0.000 | 0.874 | 0.000 |
| 256 | 0.000 | 1.607 | 0.005 |
| 512 | 0.000 | 1.383 | 0.004 |

**VERDICT (printed by the runner): `STILL CLIMBING -- gap is still partly weak-auditor; scale K further before attributing`. last delta: None.**

**What this means for the saturation claim:**
- The conservative (Wilson) ε_lower stays **on the detection floor (0.000) at every K up to 512** — one noisy 0.085 at K=8 aside. Scaling shadows to 512 (v3's entire purpose) does **not** lift the certified bound off the floor. This directly confirms §5's detection-floor finding: at this budget the estimator floor swallows the entire sub-1-ε regime, so conservative zeros are **censored, not "attack recovers 0%".**
- The point estimate wanders 0.69–1.87 without a monotone climb, and the plateau logic can't even compute a `last delta` (None) — so **there is no clean plateau to license the saturation-based attribution claim.** Per the pre-registered hedge (RUNBOOK), the honest framing is "the gap remains partly weak-auditor at K=512," NOT "the auditor has saturated, so the residual is threat-model."
- Net: the saturation-criterion claim is **not licensed by data at K≤512**. Combined with §2–3 (accounting artifact), **both** of the paper's headline quantitative claims need revision. The methodology/novelty framing survives; the specific numbers do not.

Confidence: the run is reproducible (fixed seeds, corrected bundle, sanity-gated) and was executed live, not extrapolated.

## 8. Estimator/scorer fixes implemented (HANDOFF_OPUS_FIXES_2026-07-06)

### 8.1 What was implemented (all authoritative in the repo, verified via file Read)

- **Fix 1 — sample-split holdout (`src/dp_audit_tightness/privacy/empirical.py`).** ADDITIVE only. New `threshold_selection="in_sample"` (default, unchanged) vs `"holdout"` (opt-in), plus `holdout_split_seed=20260706`, `holdout_fraction=0.5`. The holdout path shuffles (seeded), splits members and non-members into SELECTION and ESTIMATION halves, picks the point-maximising member-favoring threshold on SELECTION, and certifies the Wilson lower bound at that single fixed threshold on ESTIMATION → coverage restored. New dataclass fields (`threshold_selection`, `selection_half_sizes`, `estimation_half_sizes`) all defaulted; `<4`-per-half returns conservative zero; existing in-sample and legacy paths untouched.
- **Fix 2 — OUT-count matching (`codex/run_raw_lira_pilot.py::raw_lira_scores`).** Non-member reference now averages `c` shadows where `c` is drawn (seeded, per point) from the empirical distribution of member OUT-counts at this K, equalising reference-noise variance. Gated by `match_out_counts=True`.
- **Fix 3 — verdict/censoring semantics.** `_saturation_verdict` returns `"NO VERDICT -- fewer than 2 non-censored valid cells; cannot assess plateau"` (+ `num_ladder_cells`, `plateaued`) when the ladder has <2 cells, instead of the STILL-CLIMBING fallback. Censoring flag renamed to `conservative_zero_below_floor_or_no_signal` so a conservative 0 is no longer asserted to prove signal.
- **Fix 4 — disjoint sampler in `main()`.** Switched to `sample_query_indices_disjoint`; pilot eval limits raised to 1536 (and cifar train to 1536) so the disjoint sampler's 1280-distinct-point requirement is met.
- **Fix 5 — hard PLD backend.** `compute_epsilon_pld` default is now `backend="google"` (raises if `dp_accounting` missing); docstring corrected (fallback is anti-conservative, not "slightly looser"). `"auto"` still available for the warned fallback.
- **Tests** added to `tests/test_empirical.py` (`HoldoutThresholdSelectionTests`): null coverage, signal sanity, determinism, tiny/odd inputs, in-sample-is-default.

### 8.2 Validation results (simulation, run locally)

Fix 1 was validated against a faithful standalone reimplementation of the holdout logic (see §8.3 for why the deployed file couldn't be imported in the sandbox):

- **Null coverage, 200 reps of pure-noise scores (n=640/side):** holdout conservative ε = 0 in **200/200 (100%)** — meets the ≥95% requirement. In-sample conservative ε was nonzero in 2/200 (max 0.428), i.e. anti-conservative by construction — the exact failure mode Fix 1 removes (and the mechanism behind the lone 0.085 at K=8 in the §7 saturation run).
- **Signal sanity:** strong-signal case → holdout conservative = 3.96 (>0) and ≤ the in-sample point estimate (6.76). ✓
- **Determinism:** identical output on repeat. ✓
- **Edge cases:** tiny (n=3) → conservative zero via `threshold_sweep_holdout_insufficient_samples`, no crash; odd-length inputs → no crash. ✓
- **Fix 2:** under zero-signal shadows at K∈{2,4,8}, holdout conservative stays 0 whether matched or unmatched (Fix 1 already guarantees this). The OUT-count asymmetry's effect on the *point* estimate is modest in the i.i.d.-noise model (the target-loss term dominates the reference-variance gap), but matched is the principled construction and removes the asymmetry as a confound.

### 8.3 BLOCKER — OneDrive partial-sync froze the sandbox view of `empirical.py`

The sandbox shell reads the OneDrive-synced working tree. After the Fix-1 edits, the shell's on-disk copy of `empirical.py` froze at a truncated **474 lines** (cut mid-function), while the authoritative file (via the editor/file tools) is the correct **745 lines**. `pld_accounting.py`, `run_raw_lira_pilot.py`, and `run_mnist_saturation.py` all synced fine; only `empirical.py` (which received 5 rapid successive edits) is stuck, and a full re-read + a fresh trailing-comment write did not dislodge it after >12 min. This is the same partial-sync artifact previously noted for `run_raw_lira_pilot.py`.

Consequences (both deferred, NOT done):
1. **Running the deployed test suite** (`pytest tests/`, `validate_estimator_scorer_20260705.py`) against the real `empirical.py` — blocked; the shell would import the truncated file. Fix 1 was therefore validated against a faithful reimplementation, not the deployed bytes. The deployed bytes were verified correct by reading the whole file.
2. **Rebuilding `saturation_bundle_v4.zip` with byte-match** — blocked; zipping from the frozen mount would ship the truncated `empirical.py`, and byte-matching against the stale mount is meaningless until it syncs.

**To unblock:** force OneDrive to re-sync `src/dp_audit_tightness/privacy/empirical.py` (open/resave the file, or pause/resume OneDrive), then run `pytest tests/ -q` + `validate_estimator_scorer_20260705.py`, then rebuild the bundle and byte-match `empirical.py`/`gdp_estimation.py`/both runners against the repo, extending the notebook pre-flight with (a) holdout active, (b) OUT-count matching active, (c) hard-PLD assertion — exactly as the handoff specifies. No ε_lower should be regenerated/quoted until that clean GPU rerun.

## 9. Deployed-bytes verification (2026-07-06, independent second session)

§8.3's blocker item 1 is now CLEARED. A second session staged the authoritative files
(transcribed verbatim from the repo via the file tools, which read the cloud copy and are
unaffected by the frozen mount) into the sandbox and ran everything against the TRUE deployed
bytes of `empirical.py` (745 lines) + the new `tests/test_empirical.py` (223 lines):

- **`unittest tests.test_empirical`: 13/13 PASS** — including the null-coverage test
  (holdout ε=0 in ≥190/200 pure-noise reps; in-sample demonstrably anti-conservative),
  signal sanity, determinism, tiny/odd inputs, and in-sample-is-default.
- **`validate_estimator_scorer_20260705.py`: all 7 checks PASS** on the deployed estimator.
- **Deployed `raw_lira_scores` null test at K∈{2,4,8,32}** (function extracted verbatim from
  the authoritative pilot, fed through the deployed estimator): holdout conservative ε =
  **0.000 in every cell**, matched and unmatched. OUT-count matching pulls the null
  member/nonmember variance ratio from ~1.09–1.14 (unmatched) to ~0.87–0.96 (matched).
  The in-sample POINT estimate under the pure null ranged **0.59–3.16** across K — direct
  confirmation that point estimates at these budgets are noise and must never be quoted
  (matching reduced the worst null point inflation, e.g. K=4: 3.16 → 0.75).
- Code review of the five fixes against the authoritative source: all correctly implemented;
  legacy/in-sample paths byte-identical to the pre-fix file; new dataclass fields defaulted.

Still open from §8.3: **rebuilding `saturation_bundle_v4.zip`** — deferred until OneDrive
serves a consistent local copy (the mount in this session showed the same truncation, plus a
stale `pld_accounting.py`), because the bundle must be zipped from byte-matched files, not
transcriptions. Everything else in the handoff's verification list is done.
