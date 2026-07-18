# DP-SGD Audit Tightness — Literature Review & Novelty Map

**Project:** DP Audit Tightness Study (EECE 608, AUB)
**Compiled:** 2026-06-15
**Focus:** Tightness & the empirical–theoretical gap (`tightness_ratio = ε_lower / ε_upper`)
**Purpose:** Map what the literature actually contains, and pin down where this project can claim genuine novelty.

---

## 0. How to read this document

This is organized in three layers:

1. **The landscape** (§2–§6): an annotated, thematic survey of the literature that matters for tightness. Each theme has a short synthesis plus a table of the key papers.
2. **The novelty map** (§7): an honest assessment of the project's stated "next steps" against what is *already published*, and where real openings remain.
3. **References** (§8): every cited work with a link.

The single most important takeaway is in §7: **several of the project's listed next steps are already published results, not open problems.** Reorienting toward measurement/decomposition under a realistic threat model is where the differentiated contribution lives.

---

## 1. The problem, stated precisely

DP-SGD ships with a **theoretical upper bound** `ε_upper` from privacy accounting (RDP / PLD / f-DP). **Auditing** produces an **empirical lower bound** `ε_lower` by instantiating an adversary (a membership-inference or canary attack) and converting its best (TPR, FPR) into an ε. The gap

```
gap = ε_upper − ε_lower        tightness_ratio = ε_lower / ε_upper ∈ [0, 1]
```

can come from three distinct sources, and conflating them is the central confusion in the field:

- **Accounting gap** — the upper bound itself is loose (the analysis over-charges privacy).
- **Threat-model gap** — the audited adversary is weaker than the worst-case adversary DP protects against (e.g., black-box vs. gradient-canary white-box; natural data vs. crafted canaries; hidden intermediate states).
- **Residual / estimator gap** — finite samples + a conservative confidence interval (Clopper–Pearson / Wilson) leave ε_lower below what the adversary actually achieves.

The project already encodes this decomposition (`accounting_gap` / `threat_model_gap` / `residual_gap`). **Most of the literature does not** — it chases a bigger ε_lower without attributing *why* it was small. That asymmetry is a clue to where novelty sits (see §7).

---

## 2. Theme A — Foundations of ML privacy auditing

**Synthesis.** Auditing began as a way to test whether DP-SGD's guarantee is *vacuous or real*, then matured into a tool for asking whether the guarantee is *tight*. The arc: poisoning-based lower bounds (Jagielski) → concrete adversary instantiation across threat models (Nasr 2021) → provably tight audits in the strongest threat model (Nasr 2023). The headline finding of the foundational line: **DP-SGD accounting is essentially tight when the adversary is given worst-case power (gradient canaries, white-box, every-step insertion); the large gaps appear only when the adversary is realistically constrained.**

| Paper | Year / Venue | What it does | Result relevant to tightness |
|---|---|---|---|
| Jagielski, Ullman, Oprea — *Auditing DP ML: How Private is Private SGD?* | 2020, NeurIPS | First poisoning/clipping-based auditing; data-poisoning adversary | Establishes that empirical ε can be measured; finds large gaps to ε_upper, motivating the whole field |
| Nasr, Song, Thakurta, Papernot, Carlini — *Adversary Instantiation* | 2021, IEEE S&P | Instantiates the DP adversary at varying strengths (input-space → gradient-space, black/white-box) | Gradient-space, white-box adversary gets **nearly tight** bounds; weaker adversaries do not — first clean demonstration that the gap is *threat-model-driven* |
| Nasr, Hayes, Steinke, et al. — *Tight Auditing of DP ML* | 2023, USENIX Sec. | Uses f-DP + tight composition; audits with **two runs** instead of thousands | Tight in the worst case (crafted gradient canaries, known intermediate steps); explicitly **loose for natural data / black-box** — quantifies the realistic-setting gap |

---

## 3. Theme B — One-run auditing (the efficiency revolution and its limits)

**Synthesis.** Steinke–Nasr–Jagielski (NeurIPS 2023, Outstanding Paper) collapsed the cost of auditing from hundreds of training runs to **one**, by inserting many *independent* canaries and guessing membership of each. This is now the dominant paradigm — but a 2025 wave of papers shows it has an **intrinsic ceiling**: because the model output aggregates all canaries, their effects *interfere*, and one-run auditing does not faithfully reproduce the DP threat model (where the adversary knows all-but-one record). This is a live, unresolved tension.

| Paper | Year | Contribution | Tightness implication |
|---|---|---|---|
| Steinke, Nasr, Jagielski — *Privacy Auditing with One (1) Training Run* | 2023, NeurIPS (Outstanding) | Many independent canaries → membership guesses → ε_lower from a single run; uses DP↔generalization to avoid group-privacy cost | Makes auditing cheap enough to run routinely; foundation the project's single-run canary path resembles |
| Keinan et al. — *How Well Can DP Be Audited in One Run?* | 2025, NeurIPS | Formalizes **interference** as the barrier; shows one-run precision is algorithm-dependent and inherently limited | The single-run gap is *partly structural*, not just weak attacks — crucial caveat for any single-run ε_lower |
| *Privacy Audit as Bits Transmission* | 2025 | Information-theoretic (im)possibility results for one-run audit | Bounds how much one run can ever reveal |
| *Enhancing One-run Privacy Auditing* | 2025 | New conceptual tricks to reduce the interference barrier | Partial mitigation; gap not closed |
| *Tight Privacy Audit in One Run* | 2025 | Pushes one-run audits toward tightness | Recent attempt to close the single-run gap |

---

## 4. Theme C — Pushing lower bounds tighter (stronger adversaries)

**Synthesis.** This is the **most crowded and fastest-moving** part of the field, and the part most relevant to the project's "stronger attack" instincts. The big levers found so far: **worst-case model initialization**, **crafted gradient sequences in the hidden state**, **input-space adversarial canaries**, and **metagradient-optimized canaries**. Note carefully: several of these directly correspond to the project's *planned* next steps — meaning those steps are already taken (see §7).

| Paper | Year / Venue | Lever | Reported result |
|---|---|---|---|
| Annamalai, De Cristofaro — *Nearly Tight Black-Box Auditing of DP ML* | 2024, NeurIPS | **Worst-case initial parameters** (accounting is agnostic to init) | At ε=10: ε_emp ≈ 7.21 (MNIST), 6.95 (CIFAR-10) on 1k-record samples; ~3× tighter than prior black-box work |
| Annamalai — *Tighter Privacy Auditing of DP-SGD in the Hidden State Threat Model* | 2024 (ICLR 2025) | **Crafted gradient sequence** maximizing final-model leakage without intermediate states | Shows inserting the crafted gradient *every step* ⇒ hidden state gives **no amplification**; partial-insertion regimes still show a gap (upper bounds may be improvable) |
| *Adversarial Sample-Based Approach for Tighter Privacy Auditing (Final-Model-Only)* | 2024/25 | **Input-space, loss-based adversarial canaries** (no extra assumptions) | At ε=10: ε_lower 4.914 vs. 4.385 baseline (MNIST) |
| *The Last Iterate Advantage* | 2024/25 | Heuristic analysis + auditing of last-iterate DP-SGD | Connects hidden-state empirics to principled heuristics |
| Boglioni, Liu, Ilyas, Wu — *Optimizing Canaries for Privacy Auditing with Metagradient Descent* | 2025 | **Metagradient-optimized canary sets** | >2× improvement in ε_lower in some settings; canaries optimized on small non-private models **transfer** to large DP-SGD models |
| *Tighter Privacy Auditing of DP-SGD in the Hidden State Threat Model* (Sci. Reports) | 2026 | Hidden-state auditing refinements | Continued narrowing in the hidden-state regime |

---

## 5. Theme D — Better estimators: f-DP / GDP / density-estimation auditing

**Synthesis.** A parallel line attacks the **residual/estimator gap** rather than the adversary. Instead of converting one (TPR, FPR) point through a conservative binomial CI (Clopper–Pearson / Wilson), these methods **estimate the full trade-off curve / densities** and read tightness off f-DP or GDP. This is directly relevant because the project's CLAUDE.md lists "GDP density estimation (replace Wilson CI)" as a next step — **this is already a published research line.**

| Paper | Year | Approach | Note |
|---|---|---|---|
| Koskela, Mohammadi — *Auditing DP Guarantees Using Density Estimation* | 2024 | Histogram density estimation → lower bound on hockey-stick divergence | Solves open problem of auditing subsampled Gaussian without knowing parameters; improves on f-DP auditing |
| Askin et al. — *General-Purpose f-DP Estimation and Auditing in a Black-Box Setting* | 2025 | Estimate both distributions' densities → black-box f-DP | Contemporaneous, general-purpose |
| *Tighter Privacy Auditing of DP-SGD* | 2025, ICLR | Improved auditing estimators for DP-SGD | — |
| *Sequential Auditing for f-DP* | 2026 | Sequential/anytime-valid f-DP auditing | Newest estimator-side work |

**Estimator takeaway:** moving from Wilson CI on a single threshold to f-DP/GDP curve estimation is a known, expected upgrade. It will tighten the project's numbers, but on its own it is a *re-implementation of existing methods*, not a novel contribution.

---

## 6. Theme E — The membership-inference engine room

**Synthesis.** Every auditor is, under the hood, a membership-inference attack (MIA). The two reference points:

- **LiRA** (Carlini et al., 2022, S&P) — *Membership Inference from First Principles*. Likelihood-ratio test using shadow/reference models; 10× stronger at low FPR. The gold standard, but expensive (many reference models).
- **RMIA** (Zarifzadeh, Liu, Shokri, 2024, ICML) — *Low-Cost High-Power MIA*. Matches LiRA with as few as 1–2 reference models via fine-grained null-hypothesis modeling.
- **The Tail Tells All** (2025) — model-level MI vulnerability *without* reference models.

The project already found "Raw LiRA K=32" is its best attack and that reference-model and Gaussian-LiRA variants underperform in its setting. RMIA is on the project's list; it is a **drop-in attack upgrade**, valuable but incremental.

| Attack | Year | Cost | Role in auditing |
|---|---|---|---|
| LiRA | 2022, S&P | High (many shadow models) | Strongest general MIA; canary scorer |
| RMIA | 2024, ICML | Low (1–2 ref models) | Near-LiRA power cheaply; good for repeated audits |
| The Tail Tells All | 2025 | None (no ref models) | Model-level vulnerability estimate |

---

## 7. Novelty map — where THIS project actually stands

This is the section to act on. Cross-referencing the project's `CLAUDE.md` "Open Research Directions" and `autoresearch` "Next steps" against the literature above:

### 7.1 Project ideas that are ALREADY PUBLISHED (do not frame these as novel)

| Project's stated next step | Already done by | Status |
|---|---|---|
| "GDP density estimation (replace Wilson CI)" | Koskela & Mohammadi 2024; Askin et al. 2025 | **Published.** Use it as *method adoption*, cite these, don't claim it |
| "Worst-case initialization" | Annamalai & De Cristofaro, NeurIPS 2024 | **Published**, with strong MNIST/CIFAR numbers |
| "RMIA" as the auditor | Zarifzadeh et al., ICML 2024 | **Published attack.** Applying it to your harness is engineering, not research novelty |
| "Stronger canary generation (gradient-based / adversarial)" | Nasr 2021/2023; Annamalai 2024 (hidden state); Adversarial-sample 2024/25; **Metagradient canaries 2025** | **Heavily covered.** Gradient/adversarial canary optimization is the crowded frontier |
| "Reference-model / LiRA attack" | Carlini 2022 | **Published**; you've already tried it |

**Implication:** the "build a stronger attack to recover more of the bound" framing puts the project in direct, late competition with Google DeepMind / UCL groups who have GPU scale and a 2–3 year head start. On *raw tightness numbers*, catching up is unlikely to read as novel.

### 7.2 Where genuine openings remain

1. **Gap *decomposition* as the contribution, not gap *closing*.** The literature reports ε_lower; almost none **attribute** the remaining gap to accounting vs. threat-model vs. estimator components in a controlled way. The project already has this machinery (`accounting_gap` / `threat_model_gap` / `residual_gap`). A paper that says *"of the X nats of gap, A is the loose bound, B is the realistic threat model, C is the estimator"* — validated across datasets/attacks — is a **measurement contribution that no one owns**. This is the strongest novelty angle and it fits the existing code.

2. **Tightness under a *deployment-realistic (passive)* threat model.** Nearly all "tight" results rely on unrealistic power: crafted gradient canaries, worst-case init, white-box, every-step insertion. The project deliberately maintains a *strict* passive (observer-only) path. Systematically **characterizing how tightness degrades from canary → passive**, and *why*, is under-studied because the tight-auditing crowd considers the passive setting "uninteresting" (it gives small ε_lower). Reframed as **"how much privacy is actually at risk in realistic deployment vs. the worst case the bound charges for,"** it becomes a deployment-security question with real value.

3. **Saturation as a diagnostic signal.** The project's stance — *if stronger auditors stop improving ε_lower, that itself is the finding* — inverts the usual incentive. Formalizing a **saturation test** ("the empirical bound has converged; the residual gap is therefore accounting/threat-model, not weak-adversary") would give practitioners a principled stopping rule. The Keinan 2025 interference result gives theoretical backing for *why* saturation occurs in one-run audits — connect to it.

4. **The honest "open" problem the field flags:** whether the hidden-state gap in **non-convex** settings is *real amplification* or *weak adversaries* is explicitly unresolved (Annamalai 2024 settles only the every-step convex-ish case). A careful empirical study in your small-model regime that disentangles these — using your decomposition — is novel and within reach without GPU scale.

### 7.3 Recommended positioning (one sentence)

> *Not "we built a stronger DP-SGD attack," but "we provide a reproducible methodology that decomposes the audit gap into accounting, threat-model, and estimator components, and uses attack saturation as a principled signal for which component dominates — across a strictly-separated canary vs. passive threat-model pair."*

That claim is defensible, fits the existing architecture, and does not require out-competing DeepMind on raw ε_lower.

---

## 8. References

**Foundations**
- Jagielski, Ullman, Oprea. *Auditing Differentially Private Machine Learning: How Private is Private SGD?* NeurIPS 2020. https://www.semanticscholar.org/paper/eec423dd048f942d654d78c1a13dde7ff0d9516e
- Nasr, Song, Thakurta, Papernot, Carlini. *Adversary Instantiation: Lower Bounds for Differentially Private Machine Learning.* IEEE S&P 2021. https://arxiv.org/abs/2101.04535
- Nasr, Hayes, Steinke, et al. *Tight Auditing of Differentially Private Machine Learning.* USENIX Security 2023. https://arxiv.org/pdf/2302.07956

**One-run auditing**
- Steinke, Nasr, Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS 2023 (Outstanding Paper). https://arxiv.org/abs/2305.08846
- Keinan et al. (Amit Keinan, lead). *How Well Can Differential Privacy Be Audited in One Run?* 2025. https://arxiv.org/abs/2503.07199
- *Privacy Audit as Bits Transmission: (Im)possibilities for Audit by One Run.* 2025. https://arxiv.org/pdf/2501.17750
- *Enhancing One-run Privacy Auditing.* 2025. https://arxiv.org/pdf/2506.15349
- *Tight Privacy Audit in One Run.* 2025. https://arxiv.org/html/2509.08704

**Stronger lower bounds**
- Annamalai, De Cristofaro. *Nearly Tight Black-Box Auditing of Differentially Private Machine Learning.* NeurIPS 2024. https://arxiv.org/abs/2405.14106 · code: https://github.com/spalabucr/bb-audit-dpsgd
- Annamalai. *Tighter Privacy Auditing of DP-SGD in the Hidden State Threat Model.* ICLR 2025. https://arxiv.org/abs/2405.14457
- *Adversarial Sample-Based Approach for Tighter Privacy Auditing in Final Model-Only Scenarios.* 2024/25. https://arxiv.org/abs/2412.01756
- *The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of DP-SGD.* 2024/25. https://arxiv.org/pdf/2410.06186
- Boglioni, Liu, Ilyas, Wu. *Optimizing Canaries for Privacy Auditing with Metagradient Descent.* 2025. https://arxiv.org/abs/2507.15836
- *Tighter privacy auditing of DP-SGD in the hidden state threat model.* Scientific Reports 2026. https://www.nature.com/articles/s41598-026-38537-0

**Estimators (f-DP / GDP / density)**
- Koskela, Mohammadi. *Auditing Differential Privacy Guarantees Using Density Estimation.* 2024. https://arxiv.org/abs/2406.04827
- Askin et al. *General-Purpose f-DP Estimation and Auditing in a Black-Box Setting.* 2025. https://arxiv.org/pdf/2502.07066
- *Tighter Privacy Auditing of DP-SGD.* ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/95504595b6169131b6ed6cd72eb05616-Paper-Conference.pdf
- *Sequential Auditing for f-Differential Privacy.* 2026. https://arxiv.org/pdf/2602.06518

**Membership inference (auditor engines)**
- Carlini, Chien, Nasr, Song, Terzis, Tramèr. *Membership Inference Attacks From First Principles (LiRA).* IEEE S&P 2022. https://arxiv.org/abs/2112.03570
- Zarifzadeh, Liu, Shokri. *Low-Cost High-Power Membership Inference Attacks (RMIA).* ICML 2024. https://proceedings.mlr.press/v235/zarifzadeh24a.html
- *The Tail Tells All: Estimating Model-Level Membership Inference Vulnerability Without Reference Models.* 2025. https://arxiv.org/pdf/2510.19773

*Note on dates: a few entries (Scientific Reports, Sequential Auditing) carry 2026 stamps from preprint/early-access; treat venue/year as provisional until the camera-ready is confirmed.*
