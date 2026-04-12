# Codex Setup Log

## Scope Rule

Work added by Codex should stay in `codex/` unless a main-project change is strictly required.

Current exception:

- `src/dp_audit_tightness/privacy/pld_accounting.py`
  - patched so the project can use the installed `dp-accounting` API instead of falling back to the analytical backend

## Workspace

- repo root: `C:\Users\user\OneDrive - American University of Beirut\Documents\EECE_608`
- codex sidecar root: `C:\Users\user\OneDrive - American University of Beirut\Documents\EECE_608\codex`

## Python Setup

- venv: `codex/.venv`
- created with `--system-site-packages`
- reason: reuse the existing `torch/opacus` stack without mutating the shared global environment more than necessary

## Key Runtime Components

- Python: `3.11`
- `torch`: inherited from system site packages
- `torchvision`: inherited from system site packages
- `opacus`: inherited from system site packages
- `dp-accounting`: installed inside `codex/.venv`

## Data Prepared In Sidecar Area

- Adult dataset cache:
  - `codex/data/adult.npz`
  - `codex/data/adult.metadata.json`
- CIFAR-10 cache:
  - `codex/data/cifar-10-python.tar.gz`
  - unpacked batches in `codex/data/cifar-10-batches-py/`

## Current Sidecar Experiment Runners

- `codex/run_smoke_matrix.py`
- `codex/run_support_scaled_pilot.py`
- `codex/run_raw_lira_pilot.py`

## Current Sidecar Result Areas

- `codex/results/smoke_matrix/`
- `codex/results/support_scaled_pilot/`
- `codex/results/raw_lira_pilot/`

## Framework Intention

For each evaluated configuration, the framework should report:

- theoretical upper bound
- empirical auditing lower bound
- resulting gap
- diagnostic interpretation of why the gap remains
