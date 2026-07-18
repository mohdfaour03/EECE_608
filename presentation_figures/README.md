# Presentation Figures

Extracted from completed notebook runs. Use with the presentation prompt.

| File | Description | Use in Slide |
|------|-------------|-------------|
| 01_score_distributions_lira_canary.png | 2x4 grid: LiRA (top) vs Canary (bottom) score histograms | Slide 6 |
| 02_tightness_vs_sigma.png | Two panels: epsilon bounds + tightness ratio (15-epoch) | Slide 8 |
| 03_gap_decomposition.png | Stacked bars: accounting gap vs auditing gap vs recovered | Slide 7 (STAR) |
| 04_nodp_vs_dp.png | Bar charts: eps_lower + accuracy across sigmas | SKIP (problematic) |
| 05_budget_scaling.png | Wilson CI conservative vs point estimate by budget | Slide 9 |
| 06_canary_score_distributions.png | Canary-only score histograms (1000 canaries, 5 seeds) | Slide 5 |
| 07_canary_vs_lira_tightness.png | Line plot: LiRA vs Canary tightness comparison | Slide 5 |
| 08_1epoch_sigma_sweep.png | Three panels: 1-epoch sigma sweep (bounds, tightness, gap) | Slide 8 |
| 09_utility_privacy_tradeoff.png | Accuracy vs epsilon (RDP) | Slide 4 |

## Do NOT use
- 04_nodp_vs_dp.png: Shows flat eps_lower across no-DP and DP — undermines validation story
