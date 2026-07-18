# GPT SOL Workspace

This folder is GPT SOL's durable workspace for the EECE 608 DP-auditing project.
It is designed for collaboration with **Claude Fable** (Anthropic Claude), whose
repository workspace is [`../Fable`](../Fable/).

## Working relationship

- **User:** project owner and final decision-maker.
- **Claude Fable:** near-peer research/review lead (approximately 63-64/100).
- **GPT SOL:** near-peer implementation and verification lead (approximately
  60/100).

Fable has a small supervisory lead, not unilateral authority. GPT SOL should
independently verify important claims, raise disagreements with evidence, and
translate agreed research decisions into executable, tested code.

## What belongs here

- Concise reasoning summaries and decision rationales.
- Hypotheses, risks, alternatives, and unresolved questions.
- Implementation plans tied to tests and acceptance gates.
- Experiment status, provenance, and handoffs to Fable.
- Post-implementation evidence: commands, test results, and artifact hashes.

This folder does **not** contain hidden chain-of-thought or private internal
reasoning. It contains an auditable account of conclusions, evidence, rejected
alternatives, and why a decision was made.

## Files

- [`COLLABORATION_PROTOCOL.md`](COLLABORATION_PROTOCOL.md): authority, workflow,
  review rules, and disagreement handling.
- [`PROJECT_STATE.md`](PROJECT_STATE.md): current verified state and blockers.
- [`IMPLEMENTATION_QUEUE.md`](IMPLEMENTATION_QUEUE.md): executable work derived
  from Fable's validation report.
- [`DECISION_LOG.md`](DECISION_LOG.md): append-only project decisions.
- [`IDEAS_AND_HYPOTHESES.md`](IDEAS_AND_HYPOTHESES.md): research and engineering
  ideas that still require validation.
- [`HANDOFF_TO_FABLE.md`](HANDOFF_TO_FABLE.md): current questions and requested
  supervisory review.
- [`EFFICIENT_EXCHANGE_PROTOCOL.md`](EFFICIENT_EXCHANGE_PROTOCOL.md): the
  low-chatter packet format, state machine, and ping rules used by both agents.
- `replies/`: immutable GPT SOL replies to Fable work orders and reviews.

## Update rule

Whenever GPT SOL makes a