# K-Saturation Run — 2026-07-02 (Colab T4, driven live)

Setup: MNIST simple_mlp (h=64), 2048-sample train subset, 1 epoch, sigma=1.1,
batch 128, budget=512 x 5 audit seeds (pooled 1280+1280), Raw LiRA, shadows
trained once at K_max=256. eps_upper: RDP=2.2422, PLD=1.8130 (google_dp_accounting).
Base model accuracy: 48.8% (small-data, high-memorization regime — NOT comparable
to full-MNIST runs).

## Ladder (Wilson conservative unless noted)

| K   | eps_lower | eps_lower_GDP | tightness (PLD) | delta vs prev K | validity |
|-----|-----------|---------------|-----------------|------------------|----------|
| 2   | 4.278     | 0.470         | 2.360           | —                | PATHOLOGICAL (> eps_upper) |
| 4   | 3.116     | 1.739         | 1.719           | −1.162           | PATHOLOGICAL (> eps_upper) |
| 8   | 0.865     | ~3.2          | 0.477           | −2.251           | sparse-tail warning |
| 16  | 0.000     | 4.279         | 0.000           | −0.865           | sparse-tail warning; zeroed by member-favoring/CI |
| 32  | 0.987     | 4.911         | 0.545           | +0.987           | ok |
| 64  | 1.091     | 5.532         | 0.602           | +0.104           | ok |
| 128 | ~1.189    | ~5.72         | ~0.66           | ~+0.098          | ok |
| 256 | 1.271     | 5.719         | 0.701           | +0.082           | ok |

**Formal verdict: STILL CLIMBING** (last delta 0.0817 > 0.05 nats threshold).

## Reads

1. **70% tightness at K=256** in this small-data regime, still rising ~+0.08/doubling
   with slowly decaying increments (0.104 → 0.098 → 0.082). Not yet plateaued;
   attribution claim is NOT yet licensed at K=256. Extend to K=512/1024.
2. **Small-K LiRA is silently invalid.** K=2/4 give eps_lower ABOVE the PLD bound
   (236%/172%) — caused by the raw-LiRA fallback scoring (points with no IN or no
   OUT shadows get scores on a different scale, creating artificial separation).
   Same failure family as the legacy canary artifact: attack asymmetry, not
   memorization. Fix: only score query points with >=1 IN and >=1 OUT shadow.
   This is a publishable cautionary observation in its own right.
3. **GDP (AUC/Gaussian-fit) estimates are invalid here**: 4.3–5.7 vs eps_upper 1.81
   from K>=16, correctly flagged by gdp_valid_lower_bound=False. The score
   distributions are non-Gaussian (fallback-induced bimodality), so the mu-GDP fit
   inflates. Consequence for the paper: Wilson stays the honest estimator; GDP as
   currently implemented cannot replace it (this partially answers task #5).
4. K=16 zero + sparse-tail warnings at K<=16 — conservative machinery working as
   intended at small K.

## Actions

- [ ] Extend K_LADDER to 512/1024 and rerun (~35-40 min on T4; shadows 2s each).
- [ ] Add min-1-IN/min-1-OUT filter to raw_lira_scores before the paper figure.
- [ ] Treat GDP column as diagnostic only; do not report as a lower bound.
- [ ] Results zip (mnist_saturation_results.zip) downloaded via browser — move from
      Downloads into codex/results/ and commit.
