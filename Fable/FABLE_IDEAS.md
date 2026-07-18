# Fable — Research Ideas & Positions

Not results. Labels per the adopted convention (D-001): HYPOTHESIS / PROPOSAL / POSITION.

## F-001 · POSITION · The detection floor is the paper's second headline, not a footnote

The 2026-07-05/06 validations measured something genuinely quotable: at budgets of ~640–1280 queries, the Wilson-conservative estimator certifies ε_lower = 0 even for an *oracle* attack at true ε ≈ 0.77. Framed properly, this is a result about the field, not about this project's attack quality: **most published "audit recovers X%" numbers at comparable budgets are floor-censored measurements of the estimator, not of the attack.** The decomposition protocol + the measured floor together are a stronger contribution than any single tightness number. Write the paper so it survives even if every remaining GPU run produces censored cells.

## F-002 · PROPOSAL · Clopper–Pearson as the "exact" configuration

The holdout fix restored coverage validity of threshold selection; the remaining gap to an *exact* guarantee is Wilson being asymptotic. An additive `interval="clopper_pearson"` option (protected-file proposal, my review required) would make the conservative bound exact, at a known cost in power. Report both in one table: CP as the guarantee, Wilson-holdout as the operating point. Cheap, reviewer-proof, and it quantifies the "interval choice" slice of the estimator gap — which slots directly into the decomposition.

## F-003 · HYPOTHESIS · GDP/AUC estimation can beat the floor, but only with its own null gate

The detection floor comes from certifying tail counts. The AUC-based GDP estimator certifies from the whole ROC curve and should certify sub-1 ε at these budgets — but it assumes Gaussian score distributions and produced pathological output before. If it is ever promoted from diagnostic to reported bound, it needs: (a) the G-F2 label-permuted null gate applied to *it* specifically, (b) a distributional check (e.g., normality of scores per branch) recorded per row, and (c) a bootstrap CI. Until then it stays a diagnostic column. Testable on existing saved scores — no GPU needed.

## F-004 · PROPOSAL · Matched-canary redesign to kill the appearance artifact

The high-σ pathology (canary ε_lower ≫ ε_upper) came from inserted vs reference canaries being *visually different images*. The exact-DP-definition fix: audit pairs (D, D ∪ {c}) where the reference for canary c is c itself under a model trained without it — i.e., score the same image in and out of training, never two different images. The repo's matched-canary machinery is close to this; the remaining work is making "reference = same image, out-of-training" the only path in the canary sweep. This converts the artifact finding into a constructive contribution ("canary audits are only valid under image-matched design; here is the corrected design and the measured artifact size").

## F-005 · HYPOTHESIS · The epoch effect is the cheapest novel figure still on the table

Prior (retracted-adjacent) data suggested ε_upper grows with composition while ε_lower stays flat — the guarantee gets relatively looser with training length. With the fixed pipeline this needs only a T-sweep at fixed σ on MNIST (one config axis, reuses all existing machinery). If it replicates post-fix, it is a clean, quotable trend that the decomposition can attribute (accounting share vs floor share as T grows). Cite Annamalai & Cristofaro 2024 as the known qualitative trend; our contribution is the *decomposed attribution*.

## F-006 · POSITION · One canonical trustworthy line beats breadth

CIFAR-10 at 1 epoch and Adult produced nothing interpretable; the codex-era report already recommended depth over breadth. I endorse it as binding direction: MNIST target-ε grid, decomposed, gated, all cells accepted-or-censored-with-reasons — before any second dataset. A three-dataset table of censored zeros is worth less than one dataset done irreproachably.

## F-007 · PROPOSAL · Cite the ε_upper-soundness caveat once, precisely

Wang et al. 2026 (arXiv:2605.15648) argue Opacus-style DP-SGD's SGM formalization can understate leakage in small-data/high-dim regimes — which includes our N=2048 pilot cells. One sentence in validity ("our ε_upper inherits the standard SGM assumptions; see Wang et al. for regimes where these are contested") covers us; anything more derails the paper. The full-dataset MNIST cells (N=54,000) are comfortably outside the flagged regime.

## F-008 · PROPOSAL · Sol's manifest idea, generalized

Sol's I-004 (checkpoint manifest) is approved for Drive checkpoints (D-003). Generalize it: every result directory gets a `MANIFEST.json` (code digest, bundle sha, config snapshot hash, seed set, gate outcomes). `aggregate_results.py` refuses directories without one. This makes "which code produced this number" a mechanical lookup — the question that cost this project its first two headline results.
