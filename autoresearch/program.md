# autoresearch — DP Audit Tightness

## What this is

You are an autonomous research agent working on differential privacy auditing.
Your job is to find stronger auditing strategies that **maximize the tightness ratio** — the fraction of the theoretical privacy bound that you can empirically demonstrate.

## The problem

A DP-SGD model was trained on MNIST (simple MLP, 64 hidden units) with DP-SGD:
- noise_multiplier=1.1, clipping_norm=1.0, batch_size=256, 1 epoch
- Theoretical guarantee: `epsilon_upper ≈ 0.77` (at delta=1e-5)

Auditing tries to prove a lower bound `epsilon_lower` on the *actual* privacy leakage. The tightness ratio is `epsilon_lower / epsilon_upper`. Current best: ~12% with passive logit_margin. Your goal: push this much higher.

**Why the gap exists**: DP-SGD's theoretical bound is worst-case over *all possible* datasets and *all possible* distinguishing tests. Your auditor only tries *one specific* test. The gap measures how far your best test is from the theoretical worst case. A stronger test = higher epsilon_lower = smaller gap.

## Setup

To establish a new experiment session, do this once:

1. **Read key files**:
   - `autoresearch/prepare.py` (READ ONLY): evaluation harness, model loading, metric computation
   - `autoresearch/experiment.py` (EDITABLE): your sandbox — scoring functions, attack strategies
   - This file for context
2. **Verify the model**: Run `python autoresearch/experiment.py` once. First run trains a model (~1 min), subsequent runs are fast (~10 sec).
3. **Initialize results.tsv**: Create with header row only if it doesn't exist.
4. **Read baseline result**: Note the tightness_ratio from the initial run. This is your baseline to beat.

## The three files

- **prepare.py** — DO NOT MODIFY. Contains the immutable evaluation harness: model training, data loading, empirical epsilon estimation, and metric computation. Think of it as the measurement instrument.

- **experiment.py** — YOUR SANDBOX. Modify freely. This defines how you score membership (the `score_fn`), how you sample queries, and any preprocessing or calibration.

- **program.md** — This file. Context and research reasoning for you.

## How the evaluation works (critical to understand)

The `prepare.evaluate_audit()` function takes your member_scores and nonmember_scores and does:

1. **Threshold sweep**: For every unique score value t, compute:
   - TPR = fraction of members with score >= t
   - FPR = fraction of non-members with score >= t
2. **Privacy loss at each threshold**: `epsilon = ln((TPR - delta) / max(FPR, floor))`
3. **Best threshold wins**: The threshold giving the highest epsilon is selected
4. **Conservative reporting**: Uses Wilson CI lower bound, not the point estimate
5. **Member-favoring constraint**: Only thresholds where TPR > FPR are valid

**This means**: Your scoring function doesn't need to be a perfect classifier. It needs to find *one threshold* where members pass at a significantly higher rate than non-members. Even a small but statistically robust TPR-FPR gap at the right threshold gives a positive epsilon_lower.

**Implication for strategy**: You want scores that create *separation* in the tails. A score where the top 10% is mostly members and the bottom 10% is mostly non-members is better than a score with high average-case accuracy but overlapping tails.

## What you can change in experiment.py

Everything. Think in three categories:

### Category 1: Scoring functions (modify score_fn)

Each of these exploits a different signal that DP-SGD models leak:

- **negative_loss**: `-CrossEntropy(logits, true_label)`. Members have lower loss because the model was trained on them. The most direct signal.
- **max_probability**: `max(softmax(logits))`. Members get more confident predictions. Works because DP noise doesn't fully prevent overfitting.
- **logit_margin**: `top1_prob - top2_prob`. Members have wider gaps between their top predictions. Captures confidence beyond just the max.
- **modified_entropy**: `-sum(p * log(p))`. Members have lower entropy (more peaked distributions). Similar to max_prob but considers the full distribution shape.
- **correct_class_logit**: `logits[true_label]` (raw, before softmax). The most direct measure of how much the model "likes" the correct answer. Can be more informative than loss for small models.
- **loss_ratio**: `loss_on_true / loss_on_second_best`. Members should have a better ratio.

**Which to try first**: negative_loss is usually the strongest single signal for MIA. Try it before anything else. If your current baseline uses logit_margin, switching to negative_loss may immediately improve things.

### Category 2: Score enhancement (modify how you process scores)

Raw scores are noisy. These techniques sharpen the signal:

- **Temperature scaling**: Divide logits by T before softmax. T < 1 sharpens differences, T > 1 smooths them. Try T in [0.1, 0.5, 1.0, 2.0, 5.0].
- **Per-class normalization**: Membership signal strength varies by class. Compute z-scores within each predicted class, then combine. This removes class-difficulty confounding.
- **Score fusion**: Average the z-scored versions of multiple score types. Different scores capture different aspects of memorization. Combining them can be stronger than any single score.
- **Percentile-rank transform**: Convert each score to its percentile rank within a reference distribution (non-member scores or calibration set). Makes scores comparable across different scales.
- **Difficulty calibration**: Some examples are inherently "easy" (low loss even for non-members). Subtract the expected score for that class/difficulty level.

**Key insight**: Per-class normalization is often a big win. A score of -0.3 loss means very different things for digit "1" (easy) vs digit "8" (hard). Normalizing within classes removes this confound.

### Category 3: Query strategy (modify run_audit)

How you pick which examples to score matters:

- **Increase QUERY_BUDGET**: More samples = tighter Wilson CIs = more precise epsilon_lower. Try 256, 512, 1024. This is almost always a free win.
- **Stratified sampling**: Sample equal numbers per class. Ensures no class is over/under-represented. Important when combined with per-class normalization.
- **Multiple seeds**: Run with several seeds (401-410), concatenate all scores, then evaluate once. Larger combined sample = better statistics.
- **Oversampling hard examples**: If you can identify examples that the model is ambiguous about, those are where the membership signal is strongest.

### Category 4: Advanced attacks (larger changes to run_audit)

These are harder to implement but represent the state of the art:

- **LiRA (Likelihood Ratio Attack)**: The strongest known MIA. Train K shadow models (same architecture, same DP, different random subsets of the data). For each target example x:
  - Compute loss(x) under models that included x in training ("IN" models)
  - Compute loss(x) under models that excluded x ("OUT" models)
  - The ratio of these distributions is the membership signal
  - **Cost**: Training K shadow models (K=4-8 is enough for a demo). Each takes ~1 min for MNIST. Total: ~5-10 min. Worth it if the improvement is large.

- **Reference model attack** (simplified LiRA): Train ONE reference model on the non-member set only. Compare target model loss to reference model loss. If target loss << reference loss on example x, x is likely a member.
  - **Cost**: Training 1 extra model (~1 min). Very practical.
  - **Implementation sketch**:
    1. Split non-members: 80% for reference model training, 20% for scoring
    2. Train reference model with same architecture + DP params on the 80%
    3. Score = target_loss(x) - reference_loss(x)
    4. Members should have more negative scores (target model knows them better)

- **Augmentation consistency**: For each example, create K augmented versions (small rotations, translations, noise). Score = variance of model outputs across augmentations. Members should have more consistent outputs (the model memorized them specifically, not just their neighborhood).

## Experiment progression plan

Follow this order. Each builds on the previous:

### Phase 1: Quick wins (experiments 1-5)
1. Baseline: run current experiment.py, record tightness_ratio
2. Switch score_fn to negative_loss
3. Increase QUERY_BUDGET to 512
4. Increase QUERY_BUDGET to 1024
5. Add per-class z-score normalization to the best scoring function

### Phase 2: Score fusion (experiments 6-10)
6. Combine negative_loss + logit_margin (z-scored average)
7. Combine all three (negative_loss + max_probability + logit_margin)
8. Try temperature scaling (T=0.5) before scoring
9. Try temperature scaling (T=2.0) before scoring
10. Best fusion + best temperature

### Phase 3: Reference model attack (experiments 11-15)
11. Train a reference model on non-member data (same arch, same DP params)
12. Score = target_loss - reference_loss (basic reference attack)
13. Reference attack + per-class normalization
14. Reference attack with multiple reference models (K=2-4), average the differential
15. Best reference variant + score fusion with direct scores

### Phase 4: Advanced (experiments 16+)
16. Augmentation consistency scoring
17. LiRA with K=4 shadow models (if time allows)
18. Combine best reference attack + best augmentation + best direct score
19. Hyperparameter tuning on the best combo
20. Final push: maximize QUERY_BUDGET on the best strategy

## Constraints

- Do NOT modify `prepare.py`. It's the measurement standard.
- Do NOT install new packages beyond what's in `pyproject.toml` (torch, torchvision, opacus, dp-accounting, numpy, pandas, matplotlib, pyyaml).
- Do NOT modify any files in `src/` — those are the project's core library.
- Each experiment should complete in under 5 minutes. Exception: reference model training experiments can take up to 10 minutes.
- The metric is `tightness_ratio` from the printed output. Higher is better.

## Output format

experiment.py prints:
```
tightness_ratio: 0.123456
epsilon_lower: 0.095123
epsilon_upper: 0.771000
privacy_loss_gap: 0.675877
score_gap: 0.012345
member_favoring: True
wall_seconds: 8.3
```

Extract: `grep "^tightness_ratio:" run.log`

## Logging results

Record experiments in `autoresearch/results.tsv` (tab-separated):
```
commit	tightness_ratio	wall_seconds	status	description
```

Status: `keep` (improved), `discard` (no improvement), or `crash` (error).

## The experiment loop

1. Review `experiment.py` and past results.tsv entries
2. **Form a hypothesis** — write it as a comment in experiment.py before the change
3. Modify `experiment.py`
4. Commit with a descriptive message explaining the hypothesis
5. Run: `python autoresearch/experiment.py > run.log 2>&1`
6. Extract: `grep "^tightness_ratio:\|^member_favoring:\|^wall_seconds:" run.log`
7. If crash, check `tail -n 50 run.log`, debug, retry (up to 2 retries)
8. Log to results.tsv
9. If tightness_ratio improved: keep the commit (new baseline)
10. If not improved: `git reset --hard HEAD~1`
11. **Review what you learned** — why did it work or not? Use this reasoning for the next experiment.
12. **NEVER STOP.** Continue indefinitely until manually interrupted. You are autonomous.

Timeout: 10 minutes max per experiment. Revert and simplify if exceeded.

## Debugging tips

- **member_favoring: False** — your score direction is inverted. Negate your scores.
- **tightness_ratio: 0.000000** — no threshold found where TPR > FPR. Your scoring function isn't capturing membership. Try a fundamentally different score.
- **tightness_ratio decreased** — your change added noise or confused the signal. Revert and try a smaller modification.
- **crash on model loading** — the cached model path may be stale. Delete `autoresearch/.cache/` and re-run to retrain.
- **score_gap very small (<0.001)** — the signal is too weak. Need a fundamentally different approach, not a tweak.

## Key quantities

- `tightness_ratio`: YOUR PRIMARY METRIC. Maximize this.
- `epsilon_lower`: the empirical lower bound your attack demonstrates
- `epsilon_upper`: fixed (~0.77), determined by DP-SGD accounting
- `member_favoring`: must be True for a valid audit. If False, your scoring direction is wrong.
- `score_gap`: mean(member_scores) - mean(nonmember_scores). Positive = members score higher (correct direction).
