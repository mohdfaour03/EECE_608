# Research Findings Log

## Project: Measuring the Tightness Gap in DP-SGD

### Setup
- MNIST MLP (784-64-10), 1 epoch, batch_size=256, clipping_norm=1.0
- Default: noise_multiplier=1.1, delta=1e-5
- epsilon_upper (RDP) = 0.771

---

### Finding 1: Simple passive attacks give 0% conservative tightness
- Scoring rules tested: neg_loss, max_probability, logit_margin, score_fusion
- Budget: 128-512 per seed, 3-5 seeds
- Wilson CI conservative bound: 0% across all configs
- Point estimates: up to ~24% but unreliable (small samples, no CI support)
- **Conclusion**: Simple heuristic scores cannot reliably distinguish members from non-members at epsilon=0.77

### Finding 2: Raw LiRA achieves 6.1% conservative tightness
- K=32 shadow models (each on 50% of training data), budget=256, 5 seeds
- Wilson CI conservative: 6.1% (eps_lower=0.047)
- AUC between member/nonmember scores: ~0.83
- This is consistent with Pillutla et al. 2024 (8% at eps=1.0 with stronger threat model)
- **Conclusion**: LiRA provides the first non-zero conservative lower bound

### Finding 3: Gaussian LiRA is broken for our setup
- Original bug: non-member scores used raw log-likelihood instead of log-likelihood ratio (different scales), causing FPR=0 and >500% tightness
- Fix attempt 1 (full-population shadows): killed all signal because 57K/3K train/eval asymmetry meant shadows always include ~95% of training data
- Fix attempt 2 (synthetic IN distribution): pending results
- **Conclusion**: Gaussian LiRA requires either (a) target model trained on 50% subset, or (b) balanced train/eval split

### Finding 4: GDP estimation is not valid for DP-SGD auditing
- GDP assumes mechanism's ROC follows Gaussian trade-off: AUC = Phi(mu/sqrt(2))
- DP-SGD is not a GDP mechanism (it's a composition of subsampled Gaussians)
- AUC=0.85 under GDP implies epsilon=8+, but true epsilon=0.77
- **Conclusion**: GDP converts AUC to epsilon assuming wrong functional form; produces pathological overestimates

### Finding 5: Max-threshold GDP estimator is anti-conservative
- Original implementation: take max mu = Phi^-1(TPR) - Phi^-1(FPR) over all thresholds
- This cherry-picks noise, inflating mu
- Fixed to AUC-based estimator, which is honest but still invalid for DP-SGD (Finding 4)

---

### Literature Context

| Setting | Tightness | Source |
|---------|-----------|-------|
| White-box + worst-case dataset | ~100% | Nasr et al. 2023 |
| White-box + natural data, eps=8 | ~20% | Nasr et al. 2023 |
| Black-box + worst-case init, MNIST, eps=1.0 | 8% | Pillutla et al. 2024 |
| **Our work: passive black-box, LiRA, eps=0.77** | **6.1%** | This project |
| Black-box + worst-case init, MNIST, eps=10 | 65% | Pillutla et al. 2024 |

---

### Next: Sigma Sweep (in progress)
- Sweep noise_multiplier = [0.5, 0.8, 1.1, 1.5, 2.0, 3.0, 5.0]
- For each: train target + K=32 shadows, run Raw LiRA, compute Wilson CI
- Compute both RDP and PLD epsilon_upper for gap decomposition
- Key question: does tightness improve at higher epsilon (weaker privacy)?
- Notebook: `autoresearch/notebooks/colab_sigma_sweep.ipynb`
