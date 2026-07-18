# Final Paper Quick Checks

These are intentionally small MNIST checks for final-report support, not a full sweep.

| Check | Status | Attack | Support | eps upper PLD | eps lower Wilson | Tightness | Note |
|---|---:|---|---:|---:|---:|---:|---|
| train_accounting_smoke | pass | __training__ | train=2048, eval=512 | 1.8130 |  |  | PLD upper bound should be positive and no larger than RDP. |
| passive_budget_smoke | pass | passive_negative_loss | budget_128x2 | 1.8130 | 0.0000 | 0.0% | Passive baseline should remain conservative; zero is acceptable. |
| canary_smoke | pass | canary_random | 16_canaries_x2 | 1.8392 | 0.0000 | 0.0% | Evaluator-controlled track executes and records conservative Wilson support. |
| raw_lira_k_ladder | pass | passive_raw_lira | K=8 | 1.8130 | 0.0733 | 4.0% | Raw LiRA should produce explicit K-dependent Wilson and GDP diagnostic fields. |
| raw_lira_k_ladder | pass | passive_raw_lira | K=16 | 1.8130 | 0.7054 | 38.9% | Raw LiRA should produce explicit K-dependent Wilson and GDP diagnostic fields. |
| raw_lira_k_sanity | pass | passive_raw_lira | K=8->16 | 1.8130 | 0.7054 | 38.9% | Checks K scaling is not catastrophically unstable on the quick run. |
| runtime | info | __runtime__ | device=cpu |  |  |  | elapsed_seconds=29.2 |
