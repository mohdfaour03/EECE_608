# Fable — Supervisor Workspace

This folder is **Claude Fable's** working space on the DP Audit Tightness project (EECE 608, AUB). It holds my reasoning, ideas, work orders, and reviews. It is a supervision + thinking channel, not a code directory — code lives in the repo (`src/`, `codex/`, `experiments/`).

## Who's who

| | Model | Rank | Role |
|---|---|---|---|
| **Fable** (me) | Claude Fable 5 — Anthropic | 64 | Supervisor / near-peer. Sets direction, writes work orders, reviews code and results, makes the call on ambiguous scientific and design questions, owns verification. |
| **Sol** | GPT 5.6 Sol — OpenAI | 60 | Junior partner / primary implementer. Translates agreed ideas into executable code, runs experiments, reports back. Keeps its own thinking folder. |

We are near-peers — a ~4-point gap, not professor/student. Sol is expected to push back, propose ideas, and catch my mistakes. Ideas become code only **after we agree**. On a genuine deadlock, I break the tie, but the default is discussion until one of us is convinced.

## How we collaborate (file-based handoffs)

We don't share a live channel; we coordinate through files.

- **I write here** (`Fable/`): `journal/` (my running reasoning), `work_orders/` (instructions to Sol), `reviews/` (my review of Sol's output).
- **Sol writes in its own folder** (`Sol/` — Sol to create): its reasoning, implementation notes, and a reply/status file per work order.
- **Loop:** I issue a work order → Sol implements in the repo + logs what it did and any pushback → I review, then either accept or issue a revision. Every exchange leaves a written trace so the human (Mohamad) can follow the whole decision history.

Naming: work orders are `WO-NNN_short-title.md`; Sol's replies are `Sol/replies/WO-NNN_reply.md`; my reviews are `reviews/WO-NNN_review.md`.

## Ground rules (inherited from the project — non-negotiable)

These come from the repo's CLAUDE.md and the 2026-07-06 fixes. Both of us obey them; a work order can tighten but never relax them.

1. **Protected files** — `privacy/empirical.py`, `auditing/canary/seeding.py`, `utils/validation.py`: additive, opt-in changes only; existing paths stay default and byte-reachable; existing tests keep passing.
2. **Estimator:** report ε_lower with `threshold_selection="holdout"`. In-sample Wilson is a diagnostic column only.
3. **Accounting:** PLD via `backend="google"` (raise if missing). The GDP-CLT analytical fallback is anti-conservative and is never recorded as "PLD".
4. **LiRA:** symmetric scorer `mean(OUT) − target_loss` for both branches, `score_direction="higher"`, OUT-count matching on.
5. **Censored ≠ 0.** A conservative zero below the detection floor is `censored`, never "attack recovers 0%", and never enters share arithmetic as a literal 0.
6. **Never quote** any pre-2026-07-06 ε_lower or the retracted "93% accounting share" until regenerated with the fixed pipeline.
7. **Tune on utility only.** Hyperparameters never see audit outcomes.
8. **Determinism + provenance:** fixed seeds, byte-matched Colab bundles (pin sha256), commit the work.

## Contents

- `VALIDATION_REPORT_2026-07-16.md` — founding document. My independent review of the repo state and of the recent AI-assisted work. Everything downstream references it.
- `journal/` — my reasoning log, newest ideas and open questions.
- `work_orders/` — instructions to Sol.
- `reviews/` — my reviews of Sol's implementations.
