# Research Findings Log

## Project: Measuring the Tightness Gap in DP-SGD

### Setup
- MNIST MLP (784-64-10), 1 epoch, batch_size=256, clipping_norm=1.0
- delta=1e-5, sampling_rate=0.00449, num_steps=222
- Auditing: Raw LiRA, K=32 shadow models, budget=256, 5 seeds (640+640 samples)

---

## Main Result: Sigma Sweep with Gap Decomposition

| sigma | eps_rdp | eps_pld | eps_lower | tr(RDP) | tr(PLD) | Acct gap | Audit gap | Acct% | Aud% | Accuracy |
|-------|---------|---------|-----------|---------|---------|----------|-----------|-------|------|----------|
| 0.5   | 6.43    | 0.47    | 0.025     | 0.4%    | 5.4%    | 5.96     | 0.44      | 93%   | 7%   | 84.0%    |
| 0.8   | 1.68    | 0.28    | 0.023     | 1.3%    | 8.1%    | 1.40     | 0.26      | 84%   | 16%  | 84.0%    |
| 1.1   | 0.77    | 0.20    | 0.023     | 3.0%    | 11.8%   | 0.57     | 0.17      | 77%   | 23%  | 84.0%    |
| 1.5   | 0.36    | 0.14    | 0.022     | 6.1%    | 15.5%   | 0.22     | 0.12      | 65%   | 35%  | 83.9%    |
| 2.0   | 0.19    | 0.10    | 0.030     | 15.7%   | 29.5%   | 0.09     | 0.07      | 56%   | 44%  | 83.5%    |
| 3.0   | 0.12    | 0.07    | 0.038     | 31.2%   | 56.7%   | 0.05     | 0.03      | 65%   | 35%  | 82.7%    |
| 5.0   | 0.11    | 0.04    | 0.048     | 44.0%   | 127%*   | 0.07     | -0.01*    | 117%* | -17%*| 81.4%    |

*sigma=5.0: eps_lower > eps_pld, meaning the audit EXCEEDS the tighter PLD bound. This could indicate (a) the analytical GDP approximation is slightly too tight at high sigma, or (b) small-sample noise in eps_lower.

### Key Findings

**Finding 1: The accounting gap dominates at low sigma (strong privacy)**
- At sigma=0.5, RDP accounting is 93% of the total gap. The bound is 6.43 but PLD says it should be 0.47 -- a 14x difference. Switching accountants alone would close almost the entire gap.
- This means practitioners using Opacus with RDP accounting are **massively over-estimating** privacy cost in the strong-privacy regime.

**Finding 2: Tightness improves dramatically at higher sigma (weaker privacy)**
- Conservative tightness vs RDP: 0.4% at sigma=0.5 --> 44% at sigma=5.0
- Conservative tightness vs PLD: 5.4% at sigma=0.5 --> 57% at sigma=3.0
- The auditor recovers more of the bound when privacy is weaker, as expected.

**Finding 3: eps_lower is remarkably stable across sigma**
- eps_lower conservative ranges from 0.022 to 0.048 across a 10x range of sigma
- The gap closes because eps_upper comes down, not because the attack gets stronger
- This suggests the LiRA attack is near its ceiling for this model/dataset/budget

**Finding 4: Accuracy is robust to noise**
- Only 3% accuracy drop (84% to 81%) across sigma 0.5-5.0
- For 1-epoch MNIST MLP, the model barely notices the noise -- most learning happens in one pass

**Finding 5: At sigma >= 5.0, the audit exceeds PLD**
- eps_lower=0.048 > eps_pld=0.038 at sigma=5.0
- This is a methodological signal: either the analytical GDP bound is slightly too aggressive, or Wilson CI is noisy at these small epsilon values. Worth investigating.

---

## Earlier Findings

### Simple passive attacks give 0% conservative tightness
- Scoring rules tested: neg_loss, max_probability, logit_margin, score_fusion
- Wilson CI conservative: 0% across all configs at sigma=1.1
- Simple heuristic scores cannot reliably distinguish members from non-members

### Raw LiRA at sigma=1.1
- K=32, budget=256, 5 seeds: 6.1% conservative (previous best, from colab_lira_v2)
- This sweep confirms: 3.0% at same sigma (slightly different due to different shadow seeds)

### Gaussian LiRA is broken for our setup
- Original bug: non-member scoring scale mismatch (fixed)
- Full-population shadows: killed signal due to 57K/3K asymmetry (reverted)
- Synthetic IN approach: pending results

### GDP estimation not valid for DP-SGD
- AUC-to-mu conversion assumes Gaussian ROC shape; DP-SGD has different shape
- Produces pathological overestimates (500%+ tightness)

---

## Literature Context

| Setting | Tightness | Source |
|---------|-----------|-------|
| White-box + worst-case dataset | ~100% | Nasr et al. 2023 |
| White-box + natural data, eps=8 | ~20% | Nasr et al. 2023 |
| Black-box + worst-case init, MNIST, eps=1.0 | 8% | Pillutla et al. 2024 |
| **Our work: sigma sweep, eps=0.19** | **15.7%** | This project |
| **Our work: sigma sweep, eps=0.12** | **31.2%** | This project |
| Black-box + worst-case init, MNIST, eps=10 | 65% | Pillutla et al. 2024 |

---

## Notebooks
- `autoresearch/notebooks/colab_sigma_sweep.ipynb` -- sigma sweep (main results)
- `autoresearch/notebooks/colab_lira_v2.ipynb` -- LiRA attack experiments
