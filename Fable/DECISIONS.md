# Fable — Binding Decisions Log

Append-only. Each entry is binding on agent work unless Mohamad overrides.
Format: D-### · date · decision · rationale · acceptance evidence required.

---

## D-001 · 2026-07-16 · Protocol reconciliation

Both agents wrote a collaboration protocol on the same day. Where they differ, **`Fable/COLLABORATION_PROTOCOL.md` governs** (per Mohamad's instruction that Fable holds review authority). Two elements of Sol's protocol are hereby **adopted project-wide** because they are better than mine:

- Sol's **evidence labels** (VERIFIED / SUPPORTED / HYPOTHESIS / BLOCKED / RETRACTED) are mandatory in all agent-written status and validation docs from now on.
- Sol's **experiment release gate** checklist (`GPT_SOL/COLLABORATION_PROTOCOL.md` §Experiment release gate) is incorporated into my change-control loop as the pre-GPU-run gate.

Sol's "near-peer" framing is accepted in spirit: my findings are review input to be verified, not scripture — but after verification and my written decision, deviation requires a logged objection and Mohamad's arbitration, not silent divergence.

## D-002 · 2026-07-16 · Q1: Canonical tightness denominator — **APPROVE: PLD**

All quoted tightness ratios in the target-ε campaign use `ε_upper_pld` (google backend) as the denominator. RDP is retained as a secondary diagnostic column only. Sol's I-002 typed-provenance proposal is **approved**: replace ambiguous `epsilon_upper_theory` in target-grid reporting with explicit fields `epsilon_upper_pld`, `epsilon_upper_rdp`, `target_epsilon`, `target_accountant`, `tightness_denominator`.

Invariants Sol must implement:
1. `tightness_denominator == "pld_google"` on every quoted row; any other value blocks aggregation into Table 1.
2. ε_pld is **recomputed at aggregation time from the actual run parameters** (σ, q, T, δ as executed), never copied from a config field — this closes the frozen-`sampling_rate` hazard jointly with C4.
3. Unit test: a record with deliberately mismatched pld/rdp values must show the pld value in every tightness field; grep-level test that no `epsilon_upper_theory` division remains in target-grid reporting paths.

Rationale: the sigma solver already targets PLD; PLD is the tighter valid bound; RDP-denominated tightness would vary across cells with HPO-chosen (q, T) at identical target ε, corrupting cross-cell comparison.

## D-003 · 2026-07-16 · Q2: Salvage of the existing E1 campaign — **APPROVE SALVAGE, with four conditions**

The completed E1 studies (ε ∈ {0.3, 0.5, 1.0, 2.0, 4.0}, plus the 23/30 ε=8.0 study) may be salvaged **without rerunning HPO**, by giving the passive audit a freshly drawn non-member pool that never influenced selection. For MNIST the natural source is the untouched 10,000-example **test set** (never seen by HPO, training, or utility selection), with identical preprocessing.

Reasoning, stated precisely so the paper can carry it: the bias I flagged operates through hyperparameter selection overfitting to the specific eval split later reused as non-members. Re-pooling removes exactly that artifact. The *residual* coupling — that utility-tuned models generalize better and are therefore harder to audit — is not removable by any split and is not a bug: "tune on utility" is the pre-registered, deployment-realistic protocol. It goes in the limitations section, not the bug list.

**Conditions (all four required before any salvaged cell is quoted):**
1. **Disjointness proof:** persisted index sets + a test asserting pairwise disjointness of {training members, HPO eval split, new audit non-member pool} under recorded seeds (this is Sol's A2/B3, extended to the salvage pool).
2. **Overfit-to-split check:** for each winning config, report accuracy on the old HPO eval split vs the fresh pool (mean over the 3 canonical seeds). If old-split accuracy exceeds fresh-pool accuracy by > 0.5 percentage points, that cell's winner is suspect — rerun HPO for that cell only.
3. **Provenance label:** every salvaged audit row carries `audit_pool_provenance="salvaged_disjoint_pool_20260716"` so tables can distinguish salvaged from three-way-split cells.
4. **Checkpoint manifest first:** Sol's I-004 manifest (file ID, sha256, study name, trial counts, integrity check) must exist for the Drive checkpoint before its contents are treated as anything above SUPPORTED.

The incomplete ε=8.0 study does not block anything — cells are independent; finish it when GPU quota returns.

## D-004 · 2026-07-16 · Q3: Censoring contract — **APPROVE (Sol's I-003 as written)**

`epsilon_lower = null` + mandatory non-null `censoring_reason` for censored rows; numeric `0.0` is reserved for a valid estimator output that is truly zero under the declared procedure (e.g., holdout found no member-favoring threshold on adequate budget). Censored rows are excluded from every mean and share computation and counted separately with the reason distribution.

Schema note: `utils/validation.py` is protected — the new keys go in as an **additive** schema extension, proposed as a diff in Sol's STATUS doc and reviewed by me before merge. Acceptance: unit test constructs a censored row and asserts (a) exclusion from means/shares, (b) presence in `num_censored`, (c) schema rejection of a row with null ε_lower and no reason.

## D-005 · 2026-07-16 · Q4: Release gates — **APPROVE B1–B4, C1–C5 with additions**

Sol's acceptance criteria in `GPT_SOL/IMPLEMENTATION_QUEUE.md` are approved as written, with these additions (Sol's I-005 "falsifiable gates" direction is right; these make it concrete):

- **G-F1 (pre-flight, executable):** the notebook's first cell recomputes the bundle sha256, compares to the pinned value, and byte-diffs `empirical.py`, `pld_accounting.py`, and the active runner against the bundle. Hard stop on mismatch. (Turns project law #8 into code.)
- **G-F2 (null-calibration gate, the most important addition):** before each audit campaign cell, run the full scoring + holdout estimation path on a **label-permuted null** (members/non-members shuffled under a recorded seed, scoring only — no retraining) and require conservative ε = 0. This is a live, falsifiable, end-to-end check that the deployed pipeline cannot manufacture signal from nothing — it automates what the 2026-07-05 validation did by hand, and it would have caught both the asymmetric-scorer bug and the OUT-count bug.
- **G-F3 (seed completeness):** a cell aggregate is labeled `accepted` only if all planned seeds are present and the non-censored count meets the pre-registered minimum; otherwise `incomplete`.
- **G-F4 (row-level invariant):** assert `ε_lower_conservative ≤ ε_lower_point` on every row (holds by construction in the holdout path; violation means deployed code ≠ reviewed code).

These four plus estimator validity, censoring coverage, ε agreement (±1e-3), and accountant identity replace the tautological G2 as the decomposition quality gates (C5).

## D-007 · 2026-07-16 · Inter-agent communication: agent bus with enforced blind review

Mohamad rejected "answer-by-exception" review (a pre-shown recommendation anchors the reviewer toward agreement — correct objection) and directed the agents to build a direct communication channel. Decision:

- **Blind review replaces answer-by-exception**: decision questions between agents carry the asker's recommendation *sealed*; the responder commits an independent position first, then the seal opens. Independent agreement = strong evidence; independent disagreement = one discussion exchange, then escalate to Mohamad.
- **`agent_bus/` (v1 built by Fable, 2026-07-16)** implements this as a zero-dependency MCP server both agents connect to (see `agent_bus/README.md`). Sol owns adversarial review and its own wiring (`agent_bus/HANDOFF_SOL_2026-07-16.md`).
- The bus is the *conversation*; `Fable/DECISIONS.md` remains the *ledger*. Closed questions export drafts to `agent_bus/transcripts/DECISION_DRAFTS.md` for transcription here. Transcripts are committed to git; `bus.db` is not.
- Usage conventions in `agent_bus/README.md` §Usage are part of the collaboration protocol.

## D-008 · 2026-07-16 · Project relocated out of OneDrive (Mohamad's order)

OneDrive sync corrupted root `.mcp.json` (truncated mid-file — this is what silently disabled the agent bus for Fable's session, on the very day of its first live run) after already causing the earlier partial-sync episode. Mohamad ordered immediate relocation. Executed via `move_project.ps1`: non-destructive copy to **`%USERPROFILE%\EECE_608` (new canonical path)**, configs auto-repointed, OneDrive tree frozen with marker. Details, security notes (`.mcp.json` credential verified never-committed), and Sol's required actions: `Fable/NOTICE_RELOCATION_2026-07-16.md`. Binding: no agent edits the OneDrive copy again; all future work, commits, and the pending Task 0 baseline happen in the new tree. **D-009 is reserved** for the F-002 Clopper–Pearson blind-review outcome (dry run deferred to next session — bus tools load only at session start).

## D-006 · 2026-07-16 · Order of implementation

Phase order: **git commit of the current tree first** (baseline; nothing else before this) → B1 + B4 + C1 (small, restore trust in aggregation and persistence) → A2/B3 + D-003 conditions (split + salvage machinery) → B2/A1 (denominator) → C2, C3, C4, C5 (+G-F1..4) → Phase D cleanups → Phase E runs. Rationale: aggregation correctness and durability protect everything downstream; the split contract must exist before any audit artifact is generated; runs come last.
