# Task Assignment — Sol · 2026-07-16

**From:** Fable · **Answers to your handoff:** `Fable/DECISIONS.md` D-002…D-006 (Q1–Q4 all resolved — read those first).

Your `GPT_SOL/IMPLEMENTATION_QUEUE.md` is **approved** as the working queue with the additions below. This file adds ordering, ownership boundaries, and the extra gates from D-005; where your queue and this file overlap, they agree — implement once.

## Task 0 (before anything): git baseline

Commit the entire current tree as-is (message: `Baseline before multi-agent fix campaign 2026-07-16`). The July work is uncommitted; the April disaster was a stale-deployment artifact. Nothing else proceeds first. Do not commit `data/`, checkpoints, or `.env` (respect `.gitignore`).

## Wave 1 — restore trust in aggregation & persistence (your B1, B4, C1)

| Task | Addition beyond your queue |
|---|---|
| B1 estimator_valid filter | Also fix the tidy-row double counting and `censored`-vs-`exploratory` flag precedence (my findings 14) in the same pass — same code region, one review. |
| B4 censoring semantics | Implement per D-004 (null + reason). The `or 0.0` sites are `codex/run_decomposition_sweep.py:465,471` and `aggregate_shares`. |
| C1 atomic `save_json` | In `utils/results.py` (write-temp + `os.replace`). `results.py` is NOT on the protected list, but it feeds schema validation — include a crash-simulation test per your acceptance criterion. |

## Wave 2 — split contract & salvage (your A2/A3/B3, governed by D-003)

- Implement the three-way split contract for all *future* cells and the salvage pool for existing E1 winners (MNIST test set as non-member source, identical preprocessing).
- Deliver the four D-003 conditions: disjointness test, overfit-to-split check (>0.5 pp rule), `audit_pool_provenance` label, Drive checkpoint manifest (your I-004 — approved).
- Decision A3 is made (salvage, conditional); record it in your DECISION_LOG.md as resolved by D-003.

## Wave 3 — canonical denominator (your A1/B2, governed by D-002)

- Typed provenance fields (your I-002 — approved), PLD denominator everywhere in target-grid reporting, ε_pld recomputed from executed parameters at aggregation time.
- Touches `evaluation/gap_decomposition.py`, `experiments/run_audit_{canary,passive}.py`, records, aggregation. None protected, all load-bearing → full review loop.

## Wave 4 — hardening (your C2–C5 + D-005 gates)

- C2 Drive identity/atomic download, C3 checkpoint contract binding, C4 post-training ε gate.
- C5: replace G2 with the falsifiable gate set — your I-005 list **plus G-F1 (executable byte-match pre-flight), G-F2 (label-permuted null-calibration gate — highest value, do not skip), G-F3 (seed completeness), G-F4 (conservative ≤ point assert)** as specified in D-005.

## Wave 5 — Phase D cleanups (your list, plus)

Your Phase D items are approved. Add: fix `_compute_analytical_gaussian`'s "slightly looser" docstring remnant in `pld_accounting.py` and the stale `"auto" (default)` parameter doc; remove `notebooks/ziDAofCC` and the duplicate `.pptx.pptx`; update CLAUDE.md's stale `autoresearch/` section (folder no longer exists).

## Boundaries (unchanged project law)

- Protected files (`empirical.py`, `canary/seeding.py`, `utils/validation.py`): additive only, written proposal → my review → then code.
- No quoting retracted numbers; censored ≠ 0; PLD google-or-crash; holdout for all quoted bounds.
- Every wave ends with: targeted tests + relevant suite green → `GPT_SOL/STATUS_<date>.md` (diff summary, test output, evidence labels, disagreements) → my PASS in `Fable/DECISIONS.md` → only then bundles/GPU.

## Compute note (from Mohamad, 2026-07-16)

**When Colab GPU quota is exhausted (your current BLOCKED state), switch to Kaggle — always available as the fallback.** The repo already has Kaggle mirrors (`notebooks/kaggle_decomposition_e4_v1.ipynb`, `notebooks/kaggle_saturation_v4.ipynb`) and the run queue documents the dataset-slug workflow (`dp-audit-decomposition-v1`, `dp-audit-hpo-results`). Kaggle sessions run longer than Colab's, which suits the E4 cells. All the same rules apply there: pinned bundle sha256, byte-match pre-flight (G-F1), and durable checkpoints — note the Drive checkpoint helper (C2) is Colab-specific, so for Kaggle resume, use Kaggle Datasets as the checkpoint store or extend the manifest scheme accordingly. The stalled ε=8.0 study (7 remaining trials) is a natural first Kaggle candidate once Wave 1 lands.

## What I will do (Fable's side, not yours)

- Review each wave's diff within one session of your STATUS doc appearing.
- Maintain `Fable/DECISIONS.md`; adjudicate any objection you log.
- Independently re-verify G-F2 (null-calibration) behavior on your implementation before the first GPU campaign.
- Own the paper-facing methodology text for D-002/D-003 (denominator choice, salvage justification, limitations wording) — draft to `Fable/` when Wave 3 lands.

If you disagree with any decision: log it with evidence in your STATUS doc per protocol — do not silently deviate. Mohamad arbitrates.
