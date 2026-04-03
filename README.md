# DP Auditing Tightness Study

This repository scaffolds an experimental pipeline for a tightness study between:

- `epsilon_upper_theory`: the theoretical upper bound on privacy loss from DP accounting.
- `epsilon_lower_empirical`: the empirical lower bound on privacy loss demonstrated by auditing.

These are intentionally not treated as interchangeable "privacy bounds." Theory provides a guaranteed maximum leakage bound under the assumed DP mechanism and accountant. Auditing provides a demonstrated minimum leakage level under a specified attack regime and auditor.

## Threat Models

The repository keeps two experiment tracks separate in code, configuration, and outputs.

### Track 1: Evaluator-controlled canary stress testing

- The evaluator is allowed to modify the training data by inserting canaries.
- This is a controlled stress-test regime for observable leakage.
- It is useful for probing worst-case leakage signals under designed canary constructions.
- It is not the same as a realistic deployment adversary.

### Track 2: Passive final-model-only auditing

- The auditor cannot alter the training data.
- The auditor only observes the released final model.
- This is the realistic passive audit regime for post hoc evaluation.

## Core Quantities

Each completed audit run records:

- `epsilon_upper_theory`
- `epsilon_lower_empirical`
- `privacy_loss_gap = epsilon_upper_theory - epsilon_lower_empirical`
- `tightness_ratio = epsilon_lower_empirical / epsilon_upper_theory`
- `saturation_detected`

Saturation is a meaningful result. If progressively stronger auditors stop materially increasing `epsilon_lower_empirical`, the pipeline records that as evidence that the implemented auditor family may have saturated under the current setup.

## Project Layout

```text
configs/
  train/
  canary/
  passive/
data/
  raw/
  processed/
experiments/
  run_train.py
  run_audit_canary.py
  run_audit_passive.py
notebooks/
results/
  training/
  audits/
    canary/
    passive/
  summaries/
src/dp_audit_tightness/
  data/
  models/
  training/
  privacy/
  auditing/
    canary/
    passive/
  evaluation/
  reporting/
  utils/
```

## Minimal Workflow

1. Train one DP-SGD model with `experiments/run_train.py`.
2. Compute `epsilon_upper_theory` from the configured accountant.
3. Run one canary or passive auditor variant.
4. Estimate `epsilon_lower_empirical`.
5. Compute `privacy_loss_gap` and `tightness_ratio`.
6. Repeat with stronger auditors.
7. Detect whether improvements saturate.

## Quick Start

```bash
python experiments/run_train.py --config configs/train/mnist_mlp_dp_sgd.yaml
python experiments/run_audit_canary.py --config configs/canary/random_canary.yaml
python experiments/run_audit_passive.py --config configs/passive/baseline_passive.yaml
```

## Current Scope

The scaffold is intentionally simple:

- One benchmark dataset: MNIST.
- One simple architecture: an MLP.
- DP-SGD training through Opacus.
- One fixed `delta` per experiment configuration.
- Several auditor variants and repeated seeds.

Some auditing interfaces currently contain explicit `TODO` placeholders where later paper-specific lower-bound estimation details or stronger attack logic will be inserted.

## Current Validation Status

The project has already been validated at the scaffold level:

- training executes end to end
- artifact flow executes end to end
- bookkeeping and result saving execute end to end
- evaluator-controlled canary stress testing and passive final-model-only auditing remain separated in code and outputs

## What Is Now Implemented

The repository is now transitioning from scaffold validation to substantive first-pass auditing logic:

- canary stress testing now performs evaluator-controlled canary generation, inserts canaries into the training set, retrains a DP-SGD model under the same configuration, and scores inserted canaries against held-out decoy canaries
- passive auditing now loads the released final model, queries model outputs on member and non-member examples, and computes membership scores from those outputs
- empirical lower-bound estimation now uses a threshold sweep over member and non-member score distributions to estimate `epsilon_lower_empirical`
- aggregation now summarizes saved audit runs across seeds
- JSON result artifacts now undergo required-key schema validation before being saved or loaded

## What Remains Provisional

The current code should still be treated as first-pass research infrastructure rather than a final scientific system:

- stronger canary optimization beyond the current hand-designed canary patterns remains future work
- stronger passive auditors beyond the current score rules and calibration hooks remain future work
- current results are engineering validation outputs and should not be treated as scientific conclusions
