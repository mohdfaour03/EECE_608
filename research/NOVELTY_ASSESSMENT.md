# Novelty Assessment — Is the Framework Actually New?

**Date:** 2026-07-02
**Purpose:** Adversarial validation of the project's novelty claim before committing to the
PoPETs 2027.2 submission. Complements `LITERATURE_REVIEW.md` (2026-06-15); this document
specifically tries to KILL the framework claim and records what survives.

---

## 1. Method

Targeted searches (arXiv, alphaXiv, web) against the specific claims we intend to make,
looking for: (a) prior "auditing frameworks," (b) explicit gap decomposition/attribution,
(c) prior observation of the sigma-dependence and epoch effects, (d) 2025–2026 work that
could have scooped us. Papers examined in depth are cited inline.

---

## 2. The verdict, up front

**The framework claim survives, with qualifications.** No prior work performs a controlled
empirical decomposition of the audit gap into accounting / threat-model / estimator
components with a saturation criterion for attribution. But three of our individual
*findings* are anticipated qualitatively elsewhere, and the word "framework" is crowded.
The paper must be positioned as **a measurement methodology answering open questions the
field has explicitly posed** — not as a new attack, a new estimator, or the first
"auditing framework."

---

## 3. Closest prior work — threat-by-threat

### 3.1 The Hitchhiker's Guide to Efficient, End-to-End, and Tight DP Auditing
(Annamalai, Balle, Hayes, Kaissis, De Cristofaro — arXiv:2506.16666, June 2025)

**The single most important related paper. Read it before writing anything.**

- It is an SoK: a *review framework* (three desiderata: efficiency, end-to-end-ness,
  tightness) plus a threat-model taxonomy (Model Visibility × Canary Type). It contains
  **no controlled experiments and no empirical gap decomposition**.
- It *explicitly states the questions our project answers*:
  - **RQ2:** "Are end-to-end and tight audits possible? ... for complex algorithms like
    DP-SGD, very little work has come close." Black-box (Final Model, Sample Canary)
    audits are "still far from tight" — our passive track quantifies exactly how far, and
    our decomposition says *why*.
  - **Takeaway 5:** valid confidence intervals are routinely mishandled; "further research
    could focus on reliably estimating full privacy regions with meaningful confidence
    intervals" — our Wilson-vs-GDP comparison is directly on this.
  - One-run audits remain loose even in strong threat models, and (per Tight Privacy
    Audit in One Run, arXiv:2509.08704) "it is unclear what causes the suboptimality" —
    attribution is an acknowledged open problem.
- **Risk:** same group (UCL/DeepMind) is active and fast. **Mitigation:** we cite their
  taxonomy, adopt their (Model Visibility, Canary Type) terminology, and frame our
  contribution as an empirical instrument for their open questions. Do NOT invent
  competing terminology.

### 3.2 Lu et al., "A General Framework for Auditing Differentially Private Machine
Learning" (NeurIPS 2022, arXiv:2210.08643)

Name collision only. Content = stronger influence-based poisoning attacks + improved
hypothesis-testing machinery to maximize audit power. No gap attribution, no
accounting-vs-threat-model separation. **Must cite; must differentiate in one sentence;
avoid titling our contribution "a general framework for auditing DP."**

### 3.3 DP-Auditorium (Google, arXiv:2307.05608)

Library for testing DP of *mechanisms* via divergence estimation + dataset finders
(property testers, black-box). Not about DP-SGD training runs, not about tightness
attribution. Cite as tooling-related work.

### 3.4 Annamalai & De Cristofaro, "Nearly Tight Black-Box Auditing of DP-SGD"
(NeurIPS 2024, arXiv:2405.14106)

**Partially anticipates two of our findings.** They report the empirical/theoretical gap
varies with noise multiplier σ and number of steps T — "notably larger for larger T ...
and smaller σ." That is qualitatively our Finding 1 (tightness increases with σ) and the
direction of our epoch effect. **We cannot claim discovery of the σ- or T-dependence.**
What they do NOT do: decompose the gap (their fix is a stronger adversary — worst-case
init), quantify the accounting share (they audit against a single accountant), or study
saturation. Our contribution on these findings = *quantitative attribution*, not the
phenomenon itself.

### 3.5 The hidden-state / last-iterate line
(Cebere et al. ICLR 2025; Annamalai AISec 2024 "It's Our Loss"; "The Last Iterate
Advantage" arXiv:2410.06186)

The observation "final-model leakage does not grow the way composition-based ε_upper
grows" is the *thesis of an entire subfield*. Our epoch effect (ε_lower flat while
ε_upper composes) is the black-box shadow of this. **Frame as: independent black-box
confirmation + decomposition of how much of the epoch-induced gap is accounting vs.
auditor weakness.** Novel packaging, known phenomenon.

### 3.6 Estimator-side prior art
(Koskela & Mohammadi 2024; Askin et al. 2025; Zanella-Béguelin et al. ICML 2023;
Epsilon* arXiv:2307.11280)

GDP/f-DP/density/Bayesian estimation of empirical ε is published. Our GDP-vs-Wilson
comparison is method *adoption* used as a decomposition instrument (isolating the
estimator gap). Never present it as a new estimator.

### 3.7 Searched for and NOT found (as of 2026-07-02)

- Any paper performing a controlled decomposition: same runs, dual accountants (RDP vs
  PLD), dual threat models (canary vs passive), estimator swap (Wilson vs GDP), with the
  residual attributed via auditor saturation. **This combination appears unclaimed.**
- Any paper using auditor saturation (ε_lower plateau in K) as an attribution criterion.
- Any paper reporting the *fraction* of the gap due to accounting choice (our "93% at
  σ=0.5" style numbers).

---

## 4. Claim-by-claim scorecard for the paper

| Claim | Status | How to write it |
|---|---|---|
| "We built an auditing framework" | **DEAD as stated** — collides with Lu 2022, DP-Auditorium, SoK | "a measurement methodology / decomposition protocol" |
| Tightness increases with σ | **Anticipated** (Annamalai-DC 2024) | cite them; we add the accounting/auditor split per σ |
| Epoch effect (ε_lower flat, ε_upper composes) | **Known in spirit** (hidden-state line) | black-box confirmation + decomposition; cite Cebere, Last Iterate |
| RDP vs PLD accounting gap dominates at small σ | **Novel as an audited, quantified share** (theory known: PLD tighter) | "X% of the observed gap is removed by accounting choice alone" |
| Canary ≫ passive quantified on same runs | **Adjacent** to Nasr 2021/2023 threat-model ladders | our version is same-pipeline, controlled; emphasize strict separation |
| Saturation as attribution criterion | **Novel** (no prior use found) | formalize; connect to Keinan 2025 interference for the *why* |
| Matched-canary artifact + fix (appearance vs membership signal) | **Novel as a documented failure mode** | present as a cautionary methodological result; relate to Steinke 2023 design |
| Full decomposition protocol (all of the above on shared runs) | **Novel — this is the paper** | lead with it |

---

## 5. Recommended positioning (revised)

> *Prior work either strengthens attacks (raising ε_lower) or tightens accounting
> (lowering ε_upper), and the recent SoK (Annamalai et al., 2025) observes that
> end-to-end black-box audits of DP-SGD remain far from tight without knowing why. We
> contribute a controlled decomposition protocol that, for a fixed training pipeline,
> attributes the audit gap to (i) accounting choice, (ii) threat-model restriction, and
> (iii) statistical estimation, using auditor saturation as the criterion for declaring
> the residual attributable. On MNIST at σ=0.5, accounting choice alone explains ~X% of
> the gap; under a deployment-realistic passive adversary the threat-model share
> dominates; and a matched-canary redesign shows that naive canary asymmetries can
> silently inflate ε_lower above ε_upper.*

Verifiable, cites the field's own open questions, and none of it requires beating
DeepMind on raw ε_lower.

## 6. Mandatory hedges / obligations for the draft

1. Cite and adopt the SoK's threat-model vocabulary: our canary track ≈ (Final Model,
   Sample Canary, evaluator-controlled insertion); passive track ≈ (Final Model, Sample
   Canary, observer-only). State both precisely in their Table-3 terms.
2. Cite Annamalai-DC 2024 wherever σ/T dependence is mentioned. Never say "we discover"
   for those trends.
3. Cite Lu 2022 in related work with explicit differentiation.
4. The saturation claim requires the K=256 sweep to actually plateau — the claim is
   conditional on that experiment. If it doesn't plateau, the honest framing flips to
   "the gap remains partly weak-auditor even at K=256," which is still publishable but a
   different sentence.
5. Single-dataset risk: MNIST-only decomposition will draw fire. CIFAR-10 at 20+ epochs
   is needed as a second line, even if numbers are less clean.
6. Scope the abstract to small models / black-box; do not generalize to LLM-scale.
