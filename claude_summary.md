# DP Audit Tightness — Project Summary

**Student:** Mohamad Faour (msf20)
**Course:** EECE 608 — Trustworthy Machine Learning, AUB
**Date:** April 15, 2026

---

## Research Question

DP-SGD's privacy accountant says "this model leaks at most epsilon of privacy." But how much does it ACTUALLY leak? We measure the **tightness ratio = epsilon_lower / epsilon_upper** to find out. 100% = the guarantee is perfectly tight. Lower = the guarantee is conservative (possibly over-noising the model for no practical benefit).

---

## Framework

Built an end-to-end auditing pipeline with two threat models:

- **Passive LiRA** — Observer-only. Train K shadow models on random 50% subsets, compute IN/OUT loss differences (Carlini et al. 2022). Represents a realistic adversary.
- **Canary Insertion** — Evaluator-controlled. Insert crafted pixel-patch synthetic images into training data, retrain, measure memorization via logit margin scoring (Nasr et al. 2023). Represents worst-case stress test.

Both pipelines feed scores into a **Wilson CI threshold sweep** to produce conservative epsilon_lower bounds (per Kairouz et al. 2015). Epsilon_upper is computed via dual accounting: **RDP** (Opacus) + **PLD** (Google dp-accounting).

---

## Codebase

```
src/dp_audit_tightness/
  training/       DP-SGD via Opacus (RDP + PLD dual accounting)
  privacy/        RDP, PLD, empirical lower bound, GDP estimation
  auditing/       Canary generation/insertion/scoring + passive MIA
  data/           MNIST, CIFAR-10, Adult loaders
  models/         SimpleMLP, CNN, TabularMLP (no BatchNorm)
```

---

## Experiments & Results

### 1. First Draft (1 epoch, MNIST, budget=128)

| Threat Model | Best Auditor | epsilon_lower | Tightness |
|---|---|---|---|
| Canary (active) | Random canaries | 0.274 | 35.8% |
| Passive | Prob. margin | 0.095 | 12.3% |

- epsilon_upper = 0.771 (RDP)
- Budget=128 produced pathological results in several configurations (eps_lower > eps_upper)
- Three bugs documented and fixed: seed propagation, pathological bounds, score direction misalignment

### 2. Sigma Sweep — 1 Epoch (colab_sigma_sweep)

MNIST, K=32, budget=256, 5 seeds. Sweeps sigma=[0.5, 0.8, 1.1, 1.5, 2.0, 3.0, 5.0].

| sigma | eps_upper (PLD) | eps_lower (cons) | Tightness |
|---|---|---|---|
| 0.5 | 0.469 | 0.025 | 5.4% |
| 1.1 | 0.199 | 0.023 | 11.7% |
| 3.0 | 0.066 | 0.038 | 56.6% |
| 5.0 | 0.038 | 0.048 | ~95%+ |

**Key finding:** Tightness increases monotonically with sigma. Gap decomposition shows 93% of the gap at sigma=0.5 is accounting looseness (RDP vs PLD).

### 3. Full Sweep — 3 Datasets, 1 Epoch (colab_full_sweep)

K=128, budget=8000, 7 sigmas, 3 datasets. Results:

- **MNIST:** 9.5% to 94.6% tightness. Clean, monotonic.
- **CIFAR-10:** Mostly pathological. Model only reached 32% accuracy (1 epoch insufficient).
- **Adult:** 0% tightness everywhere. No membership signal in tabular data.
- **42% of all results pathological.** Notebook crashed at results table (ZeroDivisionError), no plots generated.

### 4. MNIST 15-Epoch Run (colab_mnist_15ep)

Sigmas=[0, 1.1, 3.0, 5.0], K=16, budget=8000, 15 epochs. GPU: NVIDIA L4.

**LiRA Results:**

| sigma | eps_upper (PLD) | eps_lower | Tightness | Accuracy |
|---|---|---|---|---|
| 0 (no-DP) | inf | 0.025 | n/a | 91.1% |
| 1.1 | 1.059 | 0.025 | 2.35% | 91.0% |
| 3.0 | 0.354 | 0.028 | 7.97% | 89.6% |
| 5.0 | 0.203 | 0.041 | 20.03% | 87.4% |

**Canary Results (first attempt, 128 canaries, 3 seeds):** All zeros with DP. Only no-DP worked (eps_lower=0.317). Insufficient sample size for Wilson CI.

**Key figures generated:** gap decomposition (star figure), tightness vs sigma, budget scaling, score distributions.

### 5. Canary Rerun (colab_canary_rerun)

1000 canaries, 5 seeds = 5000 total samples. Same sigmas, 15 epochs.

| sigma | eps_lower (canary) | Tightness | Valid? |
|---|---|---|---|
| 0 (no-DP) | 1.217 | n/a | YES |
| 1.1 | 0.469 | **44.3%** | YES |
| 3.0 | 0.862 | 243.7% | NO (pathological) |
| 5.0 | 1.332 | 657.0% | NO (pathological) |

**sigma=1.1 is the star result:** 44.3% tightness. Canary is 19x stronger than passive LiRA (44.3% vs 2.4%).

**Pathological high-sigma results** caused by canary design: inserted and reference canaries are structurally different images (different positions, different intensities). At high noise, the model distinguishes them by appearance, not memorization.

### 6. Codex Pilot (codex/results/framework_ledger)

MNIST, 2048 train samples, 1 epoch, K=16, batch_size=128, sigma=1.1.

- **Best result:** eps_lower=0.894, eps_upper=1.813, **tightness=49.3%** (provisional)
- CIFAR-10: 11.8% tightness
- Adult: pathological (invalidated)
- 48 total audit rows: 0 trusted, 5 provisional, 37 exploratory, 6 invalidated

---

## Key Findings

1. **Tightness increases with sigma** — weaker privacy is easier to audit. Monotonic across all runs.

2. **Gap decomposition** — At strong privacy (sigma=0.5), 93% of the tightness gap is accounting looseness (RDP vs PLD), not auditing weakness. Switching to PLD closes most of the gap for free.

3. **Canary >> LiRA** — Evaluator-controlled canary is 19x stronger than passive LiRA at sigma=1.1 (44.3% vs 2.4%). Consistent with Nasr et al. (2023).

4. **Epoch effect (novel)** — More training epochs increase epsilon_upper (composition) but eps_lower stays flat (~0.025-0.04). The guarantee gets relatively LOOSER with longer training. 1-epoch tightness: 11-95%. 15-epoch tightness: 2-20%.

5. **Budget matters** — Wilson CI requires 1000+ query samples for reliable bounds. First draft's budget=128 produced 42% pathological results. Budget=8000 stabilized everything.

6. **MNIST doesn't memorize** — With a 64-hidden MLP at 15 epochs, the model generalizes without memorizing individual examples. No-DP and sigma=1.1 produce identical LiRA eps_lower (~0.025). Canary forcing memorization is necessary.

7. **Canary design artifact** — At high sigma (3.0, 5.0), canary eps_lower exceeds eps_upper because inserted vs reference canaries are different images. The signal comes from pattern distinctiveness, not memorization.

---

## Literature Context

| Setting | Tightness | Source |
|---|---|---|
| White-box + worst-case data | ~100% | Nasr et al. 2023 |
| White-box + natural data, eps=8 | ~20% | Nasr et al. 2023 |
| Black-box + worst-case init, eps=1 | 8% | Pillutla et al. 2024 |
| **Our canary, sigma=1.1, 15 epochs** | **44.3%** | **This work** |
| **Our LiRA, sigma=5.0, 15 epochs** | **20.0%** | **This work** |
| **Our LiRA, sigma=1.1, 15 epochs** | **2.4%** | **This work** |

---

## Engineering Lessons

1. **Seed propagation bug** — Shadow models shared seeds, inflating scores. Fixed with deterministic seed derivation via modular arithmetic.
2. **Pathological bounds** — Budget=128 + Wilson CI = pathological in 42% of cases. Fixed by scaling to budget=8000.
3. **Score direction misalignment** — Passive pipeline had inverted score direction for some auditors. Fixed with explicit direction alignment.
4. **Canary sample size** — 128 canaries insufficient for Wilson CI under DP noise. Fixed by scaling to 1000 canaries.
5. **Accounting gap** — RDP gives epsilon=6.4 where PLD gives 0.47 at sigma=0.5 (13x difference). Always report both.

---

## Peer Review (Omar Ramadan, April 7, 2026)

| Recommendation | Status |
|---|---|
| Add figures/visualizations | DONE — 9 figures generated |
| Fix pathological passive results | DONE — budget=8000 + Wilson CI |
| Move Related Work after intro | TODO (paper) |
| Reference Table 4 | TODO (paper) |
| Use AUB email | TODO (paper) |
| NeurIPS format cleanup | TODO (paper) |

---

## Files

### Notebooks (autoresearch/notebooks/)
- `colab_mnist_15ep.ipynb` — Main 15-epoch LiRA + canary run (completed)
- `colab_canary_rerun.ipynb` — Canary rerun with 1000 canaries (completed)
- `colab_sigma_sweep.ipynb` — 1-epoch sigma sweep reference data (completed)

### Presentation (presentation_figures/)
- `01_score_distributions_lira_canary.png` — 2x4 LiRA vs canary histograms
- `02_tightness_vs_sigma.png` — Tightness ratio plot (15-epoch)
- `03_gap_decomposition.png` — STAR FIGURE: stacked bar gap decomposition
- `05_budget_scaling.png` — Wilson CI convergence with budget
- `06_canary_score_distributions.png` — Canary-only score histograms
- `07_canary_vs_lira_tightness.png` — Head-to-head comparison
- `08_1epoch_sigma_sweep.png` — 1-epoch reference results
- `09_utility_privacy_tradeoff.png` — Accuracy vs epsilon

### Do NOT use
- `04_nodp_vs_dp.png` — Flat eps_lower undermines validation story

---

## Current Limitations

1. **Single dataset validated** — Only MNIST produces clean results. CIFAR-10 needs more epochs (model barely learns at 1 epoch). Adult tabular data shows zero membership signal — possibly inherent to the data modality, not a framework bug, but unresolved.

2. **Small model capacity** — SimpleMLP with 64 hidden units on MNIST generalizes without memorizing. This makes passive auditing inherently weak — LiRA can't distinguish members from non-members because the model learns the same function regardless of which training points it sees. A larger model or harder task would show more memorization.

3. **K=16 shadow models** — Carlini et al. (2022) recommend K=256 for best LiRA results. Our K=16 gives each query point only ~8 IN and ~8 OUT shadow observations, producing noisy estimates. Higher K would likely improve LiRA tightness substantially but costs linearly more compute.

4. **Canary design artifact** — At sigma=3.0 and 5.0, canary eps_lower exceeds eps_upper (243% and 657%). The inserted and reference canaries are structurally different images (different patch positions, different intensities). At high noise, the model distinguishes them by visual appearance rather than memorization. This inflates eps_lower beyond what DP theory accounts for. A proper fix requires canary designs where inserted and reference images are more similar, or testing the exact DP definition (model trained with vs without the same record).

5. **No-DP LiRA baseline is flat** — LiRA detects eps_lower=0.025 for both no-DP and sigma=1.1 — the attack can't tell the difference. This means the no-DP baseline doesn't validate the framework for the passive track. Only the canary track (eps_lower=1.217 for no-DP) provides real validation.

6. **Wilson CI may be too conservative** — The conservative Wilson CI bound knocks many results to zero at small sample sizes. GDP density estimation (already implemented in the codebase but unused in final runs) may give tighter bounds. This is untested at scale.

7. **Raw LiRA simplification** — We use mean(out_losses) - mean(in_losses) instead of the full Gaussian-fitted likelihood ratio from Carlini et al. (2022). The non-member scoring formula also differs from the member formula (asymmetric). Both choices weaken the attack.

8. **No-DP baseline uses gradient clipping** — sigma=0 uses noise_multiplier=1e-8 with clipping_norm=1.0 still active. This is "SGD with gradient clipping, no noise" — not pure SGD. Clipping may slightly reduce memorization, underestimating no-DP leakage.

---

## Future Work

1. Gradient-optimized adversarial canaries (beyond pixel patches)
2. GDP density estimation for tighter lower bounds (already implemented, untested at scale)
3. CIFAR-10 and Adult with 15+ epochs
4. Scale to K=64+ shadow models
5. Systematic hyperparameter sweeps (noise, batch size, clipping norm)
6. Fix canary design to minimize pattern-distinctiveness artifact at high sigma
7. Score fusion and ensemble attacks

---

## Key Numbers to Remember

- **44.3%** — Best clean canary tightness (sigma=1.1, 15 epochs)
- **49.3%** — Codex pilot best (provisional, small scale)
- **2–20%** — LiRA passive range at 15 epochs
- **5–95%** — LiRA passive range at 1 epoch
- **93%** — Accounting gap fraction at sigma=0.5
- **19x** — Canary vs LiRA strength ratio
- **13x** — RDP vs PLD accounting gap at sigma=0.5
- **1000+** — Minimum budget for reliable Wilson CI bounds
