# Multi-Agent Collaboration Protocol — DP Audit Tightness Study

**Effective:** 2026-07-16 · **Authored by:** Fable (Claude Fable 5) · **Approved by:** Mohamad Faour (PI)

## Roles and authority

| Who | Folder | Role |
|---|---|---|
| **Mohamad** (PI) | repo root | Final authority on everything. Research decisions, scope, submission. |
| **Fable** (Claude Fable 5) | `Fable/` | Lead reviewer & methodology architect. Sets task priorities, reviews all changes to load-bearing code, decides what is quotable in the paper. Fable's written decisions are binding on agent work unless Mohamad overrides. |
| **Sol** (GPT 5.6) | `GPT_SOL/` | Implementation & experiment execution. Implements assigned fixes, builds/runs experiment bundles, writes status reports. |
| *(legacy)* | `codex/`, `research/` | Work from earlier model sessions. Read-only reference — do not confuse `codex/` with `GPT_SOL/`; new Sol work goes in `GPT_SOL/`. |

Conflict rule: if Sol disagrees with a Fable decision, Sol writes the objection in its status doc with reasoning — it does not silently deviate. Mohamad arbitrates.

## Folder discipline

- Each agent writes its notes, plans, and reports **only in its own folder**. Code changes go in the normal source tree (`src/`, `experiments/`, `tests/`, `configs/`).
- Cross-agent messages are files, named `<TOPIC>_<YYYY-MM-DD>.md`, referenced by exact path. No editing the other agent's files — reply with your own file.
- `Fable/DECISIONS.md` is the running log of binding decisions. Sol reads it at the start of every session.
- Sol's session output goes in `GPT_SOL/STATUS_<YYYY-MM-DD>.md`: what changed (files + line ranges), tests run and their output, anything blocked, anything it disagrees with.

## Change-control loop (mandatory for load-bearing code)

1. Fable assigns work with acceptance criteria (see `Fable/TASKS_FOR_SOL_<date>.md`).
2. Sol implements on the normal tree, runs the relevant tests, writes a STATUS doc.
3. Fable reviews the diff against acceptance criteria and records PASS/CHANGES in `Fable/DECISIONS.md`.
4. Only after PASS: rebuild bundles (byte-match + pin new sha256 in the run queue) and run on GPU.

Load-bearing = anything under `src/dp_audit_tightness/privacy/`, `src/dp_audit_tightness/auditing/`, `experiments/aggregate_results.py`, or any codex/sol experiment runner.

## Inviolable ground rules (both agents; these repeat existing project law plus Fable's 2026-07-16 review)

1. **Protected files** — `privacy/empirical.py`, `auditing/canary/seeding.py`, `utils/validation.py`: additive, opt-in changes only, and never without a written proposal reviewed by Fable first.
2. **Never quote** pre-2026-07-06 ε_lower values, the "93% accounting share", or the 44.3% canary figure as results. They are retracted/quarantined pending regeneration.
3. **PLD = google backend or crash.** The analytical GDP-CLT fallback is never recorded or reported as PLD.
4. **All quoted ε_lower use the holdout estimator** (`threshold_selection="holdout"`). In-sample values are diagnostics only.
5. **Censored ≠ 0.** A conservative zero below the detection floor is reported as censored, and must not enter averages, share arithmetic, or tables as 0.0.
6. **Do not cite the G2 telescoping gate as evidence of run quality** — it is algebraically guaranteed to pass. Quality signals are `validity`, `quality_flag`, `gdp_valid_lower_bound`.
7. Determinism: every source of randomness is seeded; a rerun with the same config must reproduce byte-identical result JSONs.
8. Bundles shipped to Colab/Kaggle must byte-match the repo, with sha256 pinned in the run-queue doc.
9. Commit to git at the end of every working session. (As of 2026-07-16 the entire July tree was uncommitted — this is how the April stale-bundle disaster happened.)
10. Threat models stay strictly separated: canary and passive never share code paths or outputs.

## Direct channel (added 2026-07-16, D-007)

`agent_bus/` provides a direct MCP message bus between the agents with **server-enforced blind review** for decision questions (asker's recommendation sealed until the responder answers independently). Conventions in `agent_bus/README.md` §Usage. The bus is the conversation; this folder's DECIS