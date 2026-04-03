# DP Audit Tightness Study

## Project Overview

Academic research project (EECE 608, American University of Beirut) studying how tight DP-SGD privacy guarantees really are. Compares theoretical upper bounds (epsilon_upper from RDP/PLD accounting) against empirical lower bounds (epsilon_lower from auditing attacks). The core metric is `tightness_ratio = epsilon_lower / epsilon_upper` — closer to 1.0 means the guarantee is tighter.

## Architecture

```
configs/           YAML experiment configs (train/, canary/, passive/)
src/dp_audit_tightness/
  config.py        Dataclass-based config loading
  data/            MNIST, CIFAR-10, Purchase-100, Adult dataset loaders
  models/          SimpleMLP, CNN, TabularMLP (no BatchNorm — Opacus constraint)
  training/        DP-SGD via Opacus with RDP + PLD epsilon_upper computation
  privacy/         RDP accounting, PLD accounting (dual backend), empirical lower bound estimation
  auditing/
    canary/        Evaluator-controlled: generate pixel-patch canaries, insert, retrain, score
    passive/       Observer-only: query model on members vs non-members, compute membership scores
  evaluation/      Tightness metrics, saturation detection, gap decomposition
  reporting/       Plotting, diagnostics, summaries
  utils/           Seeds, paths, validation, result records
experiments/       Entry-point scripts: run_train.py, run_audit_canary.py, run_audit_passive.py, sweep generation, aggregation, diagnostics
tests/             test_empirical.py, test_canary_seeding.py
paper/             LaTeX paper (NeurIPS style) — draft complete with preliminary results
```

## Key Design Decisions

- **Two threat models kept strictly separate**: canary (stress-test, unrealistic but strong) vs passive (deployment-realistic but weaker). Never mix their code paths or outputs.
- **Deterministic seeding**: `CanarySeedPlan` derives reproducible streams via modular arithmetic. Every run is reproducible.
- **Threshold sweep for lower bounds**: exhaustive sweep over all unique score thresholds, convert best (TPR, FPR) pair to epsilon. No hyperparameters.
- **Wilson confidence intervals**: conservative CI bounds to avoid false positives from small samples.
- **Member-favoring constraint**: require TPR > FPR to prevent pathological lower bounds exceeding upper bounds.
- **PLD dual backend**: Google dp_accounting library (preferred) + analytical Gaussian fallback.
- **Saturation as signal, not failure**: if stronger auditors stop improving epsilon_lower, that's a meaningful result.
- **Schema validation**: required-key checks before saving/loading any result JSON.

## Tech Stack

Python 3.10+, PyTorch 2.2+, Opacus 1.5+, dp-accounting 0.4+, NumPy, Pandas, Matplotlib, PyYAML. Package installed from `src/` via pyproject.toml.

## Current Results (from paper draft)

- MNIST MLP with DP-SGD: epsilon_upper = 0.771
- Canary auditing recovers ~36% (epsilon_lower ~ 0.277)
- Passive auditing recovers ~12% (epsilon_lower ~ 0.092)

## Conventions

- Configs are dataclasses with `@dataclass(slots=True)` loaded from YAML.
- Result records use `TrainingRunRecord`, `AuditRunRecord`, `AggregatedSummaryRecord` — all validated before persistence.
- Run scripts accept `--config path/to/config.yaml`.
- Seeds are broadcast: a single seed expands to match multi-seed lists.
- All experiments save JSON results + config snapshots for reproducibility.

## Running Experiments

```bash
python experiments/run_train.py --config configs/train/mnist_mlp_dp_sgd.yaml
python experiments/run_audit_canary.py --config configs/canary/random_canary.yaml
python experiments/run_audit_passive.py --config configs/passive/baseline_passive.yaml
python experiments/aggregate_results.py
```

## What NOT to Change Without Care

- `privacy/empirical.py` — carefully debugged lower-bound estimation logic with multiple fixes for pathological edge cases (direction misalignment, tiny-tail artifacts, score inversion). Changes here can silently produce wrong results.
- `auditing/canary/seeding.py` — seed propagation was a major source of bugs; the current scheme is validated and reproducible.
- `utils/validation.py` — schema checks prevent silent data corruption.

## Open Research Directions

1. **Stronger canary generation** — current pixel-patch canaries are hand-designed; gradient-based adversarial optimization would likely recover more of the privacy bound.
2. **Stronger passive auditors** — beyond the 4 score rules (negative_loss, max_probability, logit_margin, score_fusion); ML-based membership inference classifiers.
3. **Additional datasets** — Purchase-100 and Adult stubs exist in data/datasets.py but need data files.
4. **Systematic hyperparameter sweeps** — noise_multiplier, batch_size, clipping_norm, learning_rate grids.
5. **Gap decomposition validation** — cross-validate accounting_gap vs threat_model_gap vs residual_gap.

## Autoresearch (Karpathy-style Autonomous Experiment Loop)

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). An AI agent autonomously experiments with auditing strategies overnight, keeping changes that improve `tightness_ratio` and discarding the rest.

### Files

```
autoresearch/
  agent_loop.py        THE LOOP — calls OpenRouter LLM, modifies experiment.py, runs, keeps/reverts
  prepare.py           DO NOT MODIFY — evaluation harness (model loading, epsilon estimation, metric output)
  experiment.py        AGENT SANDBOX — scoring functions, attack strategies, query logic
  program.md           LLM context — research knowledge, strategy ideas, progression plan
  results.tsv          Experiment log (flat TSV, auto-created)
  approaches_log.jsonl STRUCTURED MEMORY — every approach tried with code summaries (never truncated)
  notebooks/
    kaggle_autoresearch.ipynb   Ready-to-run Kaggle notebook (imports agent_loop, handles setup/resume/save)
```

### How to run

**One-time setup** (trains and caches the model):
```bash
python autoresearch/experiment.py
```

**Start the autonomous loop** (self-contained, calls OpenRouter API directly):
```bash
set OPENROUTER_API_KEY=sk-or-...
python autoresearch/agent_loop.py
```

Optional flags:
- `--model meta-llama/llama-3.3-70b-instruct` (override model)
- `--max-experiments 50` (stop after N)
- `--timeout 300` (per-experiment timeout seconds)
- `--dry-run` (preview what would be sent to the LLM)

### How it works

1. Agent reads `program.md` and `experiment.py`
2. Agent modifies `experiment.py` (new scoring function, attack strategy, etc.)
3. Agent commits, runs `python autoresearch/experiment.py > run.log 2>&1`
4. Agent extracts `tightness_ratio` from output
5. If improved: keep commit (new baseline). If not: `git reset --hard HEAD~1`
6. Agent logs to `results.tsv` and loops forever

### Metric

`tightness_ratio` — higher is better. Baseline: ~0.12 (passive logit_margin). Target: push toward 0.36+ (match canary) or higher.

### What the agent should try

- Novel scoring functions (entropy, per-class normalization, logit temperature)
- Score fusion / ensembling
- Reference-model attacks (LiRA-style)
- Neighborhood attacks (score perturbed inputs)
- Larger query budgets for tighter confidence intervals
- Calibration (temperature scaling, z-score normalization)
- Class-conditional analysis
