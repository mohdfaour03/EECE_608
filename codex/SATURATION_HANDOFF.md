# Handoff — MNIST Saturation Runner (prepared overnight)

## What I added
- **`codex/run_mnist_saturation.py`** — a new, self-documented experiment script. Nothing else was modified. The protected files (`privacy/empirical.py`, `canary/seeding.py`, `utils/validation.py`) were **not touched**.

## Why
Our smoke results showed `eps_lower` jumping 4% (K=8) → 39% (K=16) for Raw LiRA. That means a big part of the audit gap is currently **"weak auditor / not enough compute,"** not loose accounting or a weak threat model. We can't honestly attribute the residual gap until `eps_lower` **stops moving** as we add shadow models. This script answers exactly that: *does eps_lower plateau, and at what K?*

## How it works (safe by design)
- It **reuses the already-debugged scoring/estimation functions** in `run_raw_lira_pilot.py` (imports them as a module) instead of re-implementing LiRA — so no new scoring bugs.
- It only varies **K** (shadow-model count) over `[2, 4, 8, 16, 32, 64, 128, 256]`. Everything else is held fixed (MNIST, simple_mlp, noise=1.1, 2048-example subset, budget=512, 5 seeds).
- Shadow models are trained **once** at the largest K and reused for every smaller K → the sweep is cheap after the one-time training cost.
- MNIST only — the single trustworthy line (per `framework_validation_report.md`: "scale one canonical trustworthy line rather than expand breadth").

## How to run
```bash
# from the project root, in the env with torch/opacus/dp-accounting installed
python codex/run_mnist_saturation.py
```
First run trains 1 target + 256 shadow MLPs (the slow part on CPU; fast on GPU). No network needed.

## What it produces (in `codex/results/mnist_saturation/`)
- `mnist_saturation_summary.{json,csv}` — per-K rows: `epsilon_lower_conservative` (Wilson), `epsilon_lower_gdp`, `epsilon_upper_tighter` (PLD), `tightness_ratio_tighter`.
- `mnist_saturation_verdict.json` — consecutive `eps_lower` deltas per K + a plateau verdict ("STILL CLIMBING" vs "PLATEAUED").

## How to read the result
- If `eps_lower` **plateaus** by some K → the leftover gap is now attributable to accounting / threat-model → that's the green light for the decomposition story.
- If it's **still climbing** at K=256 → the gap is still partly weak-auditor; we either scale K further or switch to a stronger auditor (e.g., the worst-case-init lever from Annamalai 2024) before claiming anything.

## Validation done overnight
- `py_compile` passes on the new runner and on `run_raw_lira_pilot.py`.
- Confirmed every pilot symbol the runner calls exists.
- **Not** run end-to-end here — that needs your local env with torch/opacus/GPU. That's the first thing to do together tomorrow.

## Talking points for tomorrow
1. Walk through the runner (5 min) and run it on MNIST.
2. Read the verdict: did eps_lower plateau? at what K? what tightness did it reach?
3. Decide: is the residual gap attributable yet, or do we need the worst-case-init lever first?
4. Continue Day-1 reading (Cebere full read + the two SoKs + survey).
