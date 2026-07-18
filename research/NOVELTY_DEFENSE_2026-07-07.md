# Novelty Defense — Adversarial Verification, 2026-07-07

**Purpose:** Update and stress-test `NOVELTY_ASSESSMENT.md` (2026-07-02) against everything
published or posted through July 2026, before committing to the PoPETs 2027.2 submission.
Method: four parallel adversarial literature sweeps (gap decomposition; saturation +
estimator validity; canary-design artifacts; full venue sweep June 2025 – July 2026), each
explicitly instructed to KILL our claims. Key kill-candidates verified by direct fetch of
full texts. This document supersedes the claim-by-claim scorecard in NOVELTY_ASSESSMENT.md §4.

---

## 1. Executive verdict

Of the paper's four claimed contributions, **one survives intact, one survives narrowly with
mandatory rescoping, and two are scooped as stated and must be repositioned** — but both
repositioned forms remain publishable and, importantly, *strengthen* the honest version of
the paper. The "unclaimed combination" verdict of 2026-07-02 still holds for the joint
protocol: **no paper through July 2026 runs a single controlled protocol on the same training
runs that quantitatively splits ε_upper − ε_lower into accounting, threat-model, and
estimator shares.** That joint protocol is the paper. Nothing else is.

| # | Claim | Verdict (2026-07-07) | Required action |
|---|---|---|---|
| 1 | Same-runs decomposition protocol (accounting / threat-model / estimator shares) | **SURVIVES — narrowest reading only** | Claim only the *joint controlled quantification*; each individual axis is separately established (see §2.1) |
| 2 | Auditor saturation (ε_lower plateau in K) as attribution criterion | **SURVIVES NARROWLY — must be rescoped as a heuristic** | Cite the three papers that undermine plateau⇒optimality inference (§2.2); present as cheap necessary-condition diagnostic, never as proof |
| 3 | Matched-canary failure mode (appearance vs membership signal) | **SCOOPED as a phenomenon** (Cebere et al. Zero-Run, May 2026; + MIA-evaluation confound literature) | Reposition as practitioner-facing *case study/pitfall report* in the evaluator-controlled canary pipeline, citing the formalization (§2.3) |
| 4 | Sample-split holdout fix for anti-conservative threshold-sweep CI | **FIX SCOOPED (concurrent)** — Michel, Basu & Kaufmann, Feb 2026, Appendix C | Reposition as *empirical quantification of the coverage violation's magnitude* in standard DP-SGD audit pipelines + cite Michel et al. as concurrent; note their DKW-envelope alternative (§2.4) |

---

## 2. Claim-by-claim analysis

### 2.1 Claim 1 — the decomposition protocol (the paper's spine)

**Survives, but every individual axis is separately anticipated:**

- **Accounting share:** Nasr et al., *Tight Auditing of DPML* (USENIX Sec 2023,
  arXiv:2302.07956) already demonstrated that part of the theory–practice gap is an
  accounting artifact (empirical leakage matches f-DP/PLD-style accounting while
  RDP-reported ε is loose). Our corrected accounting-share table (VALIDATION_2026-07-06 §3:
  peaks ~55–60% at mid-σ, ~21% at σ=0.5) is a *quantified share on shared runs*, not a new
  phenomenon. Never write "we discover the accounting gap."
- **Threat-model share:** Nasr et al., *Adversary Instantiation* (S&P 2021,
  arXiv:2101.04535) is a threat-model ladder on the same training setup — the direct
  ancestor of our canary-vs-passive split. Annamalai & De Cristofaro (NeurIPS 2024,
  arXiv:2405.14106) attribute the black-box gap to average-case initialization by
  controlled intervention. Both must be cited as the attribution precedents.
- **Estimator share:** Nasr et al. 2023 already swap Clopper-Pearson-style estimation for
  GDP; Koskela & Mohammadi (SaTML 2025, arXiv:2406.04827), *Let's Ask Gauss*
  (arXiv:2606.12733, Jun 2026), and *Optimal Guarantees for Auditing RDP ML*
  (arXiv:2605.21938, May 2026 — explicitly separates statistical-estimation error from
  leakage) own the estimator axis.
- **The conceptual three-way frame itself** is the SaTML 2026 SoK's systematization (threat
  models / attacks / evaluation functions). We adopt their vocabulary; we do not compete
  with it.

**What remains ours:** the joint, controlled, same-training-runs quantification — dual
accountants × dual threat models × dual estimators on identical runs, with shares reported
as fractions of the same measured gap. No sweep found this combination. One near-miss to
watch: *Empirical Privacy Variance* (ICML 2025, arXiv:2503.12314) shows same-(ε,δ) models
vary widely in empirical privacy — cite it as evidence the gap is not a single number, which
motivates decomposition.

**Sentence to write:** "Prior work established each gap source in isolation [2101.04535,
2302.07956, 2405.14106, 2406.04827]; we contribute the first controlled quantification of
all three shares on shared training runs."

### 2.2 Claim 2 — saturation as attribution criterion

No paper uses the ε_lower plateau over a shadow-count K-ladder as a formal attribution
criterion. But three results constrain what a plateau can prove, and reviewers will know them:

1. **Haghifam, Smith & Ullman** (arXiv:2508.19458): the attacker may need Ω(n + n²ρ²)
   reference samples to match the fully-informed adversary — saturation at feasible K does
   **not** certify convergence to the optimal attack.
2. ***MIA on Sequence Models*** (arXiv:2506.05126): plateaus are attack-class-specific —
   univariate attacks plateau near K=32 while richer estimators keep improving. A Raw-LiRA
   plateau therefore bounds Raw-LiRA, not the threat model.
3. **Keinan & Yehudayoff / Keinan, Shenfeld & Ligett** (arXiv:2503.07199, NeurIPS 2025):
   canary interference is a *fundamental* barrier in one-run-style audits — a theoretical
   account of why saturation may reflect audit-design limits rather than threat-model limits.

**Also directly relevant:** Nasr et al. 2023 achieved attribution *constructively* (show a
stronger adversary reaches the bound, so the residual is threat model). Our criterion is the
cheap, non-constructive complement. The 2026-07-06 live run returned **STILL CLIMBING at
K=512 with conservative ε on the detection floor** — so per the pre-registered hedge, the
current honest sentence is "the gap remains at least partly weak-auditor at K=512," and the
saturation *criterion* is presented as methodology with a negative/censored instance, not as
a licensed attribution.

**Rescoped claim:** "We propose auditor saturation as a *necessary-condition heuristic* for
threat-model attribution — if ε_lower still climbs in K, attribution is premature — and show
it correctly withholds attribution in our own experiments. It cannot be sufficient
[2508.19458, 2506.05126]."

### 2.3 Claim 3 — matched-canary failure mode: SCOOPED, reposition

**Verified by direct fetch:** Cebere, Bleistein, Even & Bellet, *Privacy Auditing with Zero
(0) Training Run* (arXiv:2605.14591, May 2026) formalizes exactly this mechanism: when
member and non-member populations differ in distribution (ℙ₁ ≠ ℙ₀), "an auditor may appear
to perform well by exploiting distribution-shift signals rather than privacy leakage from
the algorithm's output." They model it causally (leakage = composition of a
distribution-shift mechanism with the algorithm's true leakage), and give propensity-score
corrections restoring valid lower bounds. Additionally: Steinke et al.'s one-run framework
(NeurIPS 2023) makes exchangeable canary inclusion its core validity assumption (our
artifact is a violation of a *documented* assumption); PANORAMIA (NeurIPS 2024,
arXiv:2402.09477) engineered a baseline-subtraction correction for the same confound; and
the MIA-evaluation literature (Blind Baselines, arXiv:2406.16201; Meeus et al. SoK, SaTML
2025; Duan et al., COLM 2024) documents the distribution-shift confound extensively.

**What survives:** ours is, as far as found, the only *empirical demonstration inside an
evaluator-controlled DP-SGD canary audit* that a plausible implementation choice
(structurally different inserted vs reference canaries) silently produces ε_lower > ε_upper
(the 243%/657% cells), plus the matched-pool fix. Present it as a cautionary case study and
pitfall report — "a concrete instance, in the canary-insertion pipeline, of the confound
formalized by [2605.14591]" — never as a newly discovered failure mode. This also connects
to Tramèr et al. *Debugging DP* (arXiv:2202.12219): ε_lower > ε_upper is the field's bug
signal, and we show it can fire spuriously.

### 2.4 Claim 4 — holdout estimator fix: FIX SCOOPED (concurrent), reposition

Michel, Basu & Kaufmann, *Sequential Membership Inference Attacks* (arXiv:2602.16596,
Feb 2026), Appendix C.1, implements the same construction with a coverage proof: split into
calibration and evaluation phases "so the thresholds at which Clopper–Pearson intervals are
computed are independent of the data used to estimate the errors," with a union bound over
the threshold grid. Their Appendix C.2 adds a **DKW uniform envelope** that validly maximizes
over *all* thresholds with no split at all — strictly more sample-efficient than our
half/half split. Prior art that already treats event/threshold selection as the validity
obstacle: Askin, Kutta & Dette (S&P 2022, arXiv:2108.09528), Askin et al. (USENIX Sec 2025,
arXiv:2502.07066), Zanella-Béguelin et al. (ICML 2023, Bayesian route), sequential f-DP
auditing (arXiv:2602.06518 — burn-in calibration then fresh-data testing).

**What survives:** (a) nobody has *quantified the magnitude* of the coverage violation in
the standard DP-SGD multi-run audit pipeline — our null-coverage simulation (in-sample
conservative ε up to 0.43 on pure noise; null *point* estimates 0.59–3.16 across K; 13/13
tests) is exactly that; (b) Liu et al. (TPDP 2025, arXiv:2506.15349) documents that
max-and-report is still standard practice in the one-run line — so the pitfall is live, not
historical. Cite Michel et al. as concurrent, adopt or at least discuss the DKW envelope as
the better fix (TODO already notes CP would make "conservative" exact — extend that TODO to
DKW), and present our contribution as the *empirical anatomy* of the failure plus a
deployed, tested fix in an open pipeline.

---

## 3. Papers the reference list must add (beyond NOVELTY_ASSESSMENT.md's list)

**Blocking (reviewers will expect these):**
- arXiv:2605.14591 — Cebere et al., Zero-Run auditing (scoops claim 3's phenomenon)
- arXiv:2602.16596 — Michel, Basu & Kaufmann, Sequential MIA (scoops claim 4's fix; App. C)
- arXiv:2101.04535 — Nasr et al., Adversary Instantiation (threat-model ladder precedent)
- arXiv:2302.07956 — Nasr et al., Tight Auditing (accounting + estimator precedent)
- arXiv:2508.19458 — Haghifam, Smith & Ullman (limits of saturation inference)
- arXiv:2503.07199 — Keinan et al., one-run audit limits (NeurIPS 2025)
- arXiv:2206.05199 — Zanella-Béguelin et al., Bayesian estimation of DP
- arXiv:2502.07066 — Askin et al., f-DP estimation/auditing (USENIX Sec 2025)

**Strongly recommended:**
- arXiv:2506.05126 (attack-specific plateaus); arXiv:2503.12314 (Empirical Privacy
  Variance); arXiv:2406.04827 (density-estimation auditing); arXiv:2606.12733 (Let's Ask
  Gauss); arXiv:2605.21938 (optimal RDP auditing); arXiv:2506.15349 (max-and-report is
  standard practice — motivation); arXiv:2402.09477 (PANORAMIA baseline correction);
  arXiv:2406.16201 (Blind Baselines); arXiv:2202.12219 (Debugging DP); arXiv:2411.10614
  (shuffling audit — accounting-mismatch cousin); arXiv:2108.09528 (event-selection
  obstacle, 2022); arXiv:2410.22235 (f-DP one-run, ICML 2025); arXiv:2509.08704 (tight
  one-run; fixes CP *dependence*, orthogonal to our *selection* fix — say so in one line).

**Canary-design context (related-work paragraph):** arXiv:2605.27292 (Detectability in
Diversity); arXiv:2507.15836 (metagradient canaries); arXiv:2503.06808 (LLM audit canaries);
OptiFluence (ICML 2026). These show the field optimizes canary *strength* under randomized
inclusion — orthogonal to our matched-design validity point, which is why the pitfall matters.

---

## 4. Updated positioning paragraph (replaces NOVELTY_ASSESSMENT.md §5)

> Prior work has established, in isolation, that each of three factors contributes to the
> gap between DP-SGD's accounted ε and audited lower bounds: accounting choice [Nasr et al.
> 2023], threat-model restriction [Nasr et al. 2021; Annamalai & De Cristofaro 2024], and
> statistical estimation [Zanella-Béguelin et al. 2023; Askin et al. 2025; Koskela &
> Mohammadi 2025]. The recent SoK [Annamalai et al. 2026] systematizes these axes and poses
> attribution as an open problem. We contribute the first controlled decomposition that
> quantifies all three shares on the *same* training runs, using dual accountants (RDP vs
> PLD), dual threat models (evaluator-controlled canaries vs a passive observer), and dual
> estimators (selection-valid Wilson bounds vs GDP estimation), together with an auditor
> K-saturation diagnostic that withholds attribution when the attack is still improving —
> as it demonstrably is at K=512 in our setting. Along the way we document two validity
> pitfalls that silently inflate audit results in this pipeline: threshold selection and
> confidence estimation on the same sample (concurrently corrected, with proof, by Michel
> et al. 2026 — we quantify the violation's magnitude in DP-SGD audits and deploy a
> sample-split fix), and unmatched canary/reference design, a concrete evaluator-controlled
> instance of the distribution-shift confound formalized by Cebere et al. 2026, which drove
> our audited ε_lower above ε_upper by up to 6.6×.

Everything in that paragraph is licensed by verified results already in hand (given the GPU
rerun regenerates the final numbers).

## 5. Scoop-risk watchlist (re-check before submission)

The Inria Lille/Bellet cluster (2605.14591, 2605.27292, 2602.16596) and the UCL/DeepMind
cluster (SoK, 2405.14106, 2411.10614) are both fast-moving and adjacent. Re-run this sweep
(a) when the GPU results are frozen, and (b) the week before the PoPETs deadline. Anything
new combining "decomposition"/"attribution" with "auditing DP-SGD" is a red alert.

## 6. Method note

Sweeps executed 2026-07-07 by four parallel research agents (search + full-text fetch;
kill-candidates 2605.14591 and 2602.16596 verified by direct fetch of full texts; the
Michel et al. Appendix-C quote was grepped verbatim from the arXiv HTML). PoPETs 2025 (all
issues) contained no DP-SGD-audit papers; S&P/CCS/USENIX 2025-26 sweeps surfaced only the
items above. Caveat: arXiv submissions from late June 2026 may be undersampled by search
indexing; re-sweep per §5.
