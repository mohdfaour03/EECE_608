# Efficient GPT SOL / Fable Exchange Protocol

**Effective:** 2026-07-16  
**Governing protocol:** [`../Fable/COLLABORATION_PROTOCOL.md`](../Fable/COLLABORATION_PROTOCOL.md)  
**Purpose:** minimize model-to-model chatter and move from decision to tested
implementation with one assignment packet and one review packet per wave.

This document is an operational addendum. It does not replace Fable's governing
protocol or the user's final authority.

## 1. Communication model

Communication is asynchronous and file-backed:

```text
Fable work order/decision
          |
          v
GPT SOL intake -> implementation + tests -> one consolidated reply
          ^                                      |
          |                                      v
          +---------- Fable PASS/CHANGES --------+
```

The files are the source of truth. A CLI ping or heartbeat only announces that
a new packet exists; it never carries the authoritative decision by itself.

## 2. Where each side writes

- Fable assignments: `Fable/work_orders/WO-NNN_short-title.md`.
- Fable binding decisions: `Fable/DECISIONS.md`.
- Fable reviews: `Fable/reviews/WO-NNN_review.md`.
- GPT SOL replies: `GPT_SOL/replies/WO-NNN_reply.md`.
- GPT SOL session summary: `GPT_SOL/STATUS_YYYY-MM-DD.md`.
- Normal source changes: existing repository source/config/test directories.

Neither agent edits the other agent's files. A correction is a new reply or
review that references the original message ID.

Existing dated task files, such as `Fable/TASKS_FOR_SOL_2026-07-16.md`, are
treated as work orders until Fable adopts the `WO-NNN` naming convention.

## 3. Packet state machine

Every work item has exactly one state:

1. `READY_FOR_SOL`
2. `IN_PROGRESS`
3. `READY_FOR_FABLE_REVIEW`
4. `CHANGES_REQUESTED`
5. `ACCEPTED`
6. `BLOCKED_USER`
7. `BLOCKED_EXTERNAL`

Only Fable sets `READY_FOR_SOL`, `CHANGES_REQUESTED`, or `ACCEPTED`. GPT SOL
sets the remaining states in its own reply/status files.

## 4. Complete Fable work-order contract

To avoid clarification rounds, one Fable work order should include:

- message ID and parent ID, if revising a prior order;
- objective and scientific rationale;
- binding decisions already made;
- exact in-scope and out-of-scope files;
- protected-file restrictions;
- invariants that must remain true;
- acceptance tests and release gates;
- expected artifacts or schema changes;
- default action for every noncritical ambiguity;
- what evidence Fable needs for PASS;
- whether user approval is required before an external side effect.

If a field is absent and the choice is reversible, implementation-local, and
scientifically neutral, GPT SOL chooses the smallest reasonable default and
records it instead of asking.

## 5. Complete GPT SOL reply contract

GPT SOL sends one consolidated reply per wave containing:

- state and message/parent IDs;
- outcome first;
- files changed and the behavior changed in each;
- acceptance-criterion checklist;
- exact tests/commands and summarized output;
- evidence labels: VERIFIED, SUPPORTED, HYPOTHESIS, BLOCKED, RETRACTED;
- assumptions and any deviations;
- residual risks;
- at most one batched blocking-question section;
- a concise review request naming the highest-risk lines or invariants.

Questions that do not block safe progress are recorded under `Assumptions` and
do not delay implementation.

## 6. Question-batching policy

Before asking Fable anything, GPT SOL must:

1. read `Fable/DECISIONS.md`, the current work order, relevant Fable ideas, and
   prior review packets;
2. search repository documentation and source for an existing answer;
3. distinguish scientific decisions from implementation details;
4. propose a recommended answer and consequence for each remaining question;
5. send all blocking questions in one packet.

One work wave gets at most one pre-implementation question packet. If Fable's
response leaves a scientifically material ambiguity, escalate the consolidated
choice to the user instead of starting an indefinite agent loop.

## 7. Autonomy defaults

GPT SOL proceeds without another ping when all are true:

- the work order is `READY_FOR_SOL`;
- the change is within the listed scope;
- no protected-file proposal is missing;
- no new scientific semantics are invented;
- the action is reversible and does not publish, spend money, or overwrite
  external state;
- relevant acceptance tests can be written before or with the implementation.

GPT SOL pauses only for:

- contradictory binding decisions;
- a scientific choice that changes validity or whether results must be rerun;
- protected-file work without the required written proposal/review;
- destructive or externally consequential actions not already authorized;
- dirty-worktree overlap that cannot be safely isolated;
- missing credentials, compute, or another genuine external blocker.

## 8. Review efficiency

- Fable reviews the acceptance contract, not the entire repository again.
- GPT SOL highlights only load-bearing diffs and newly introduced invariants.
- Fable returns one of: `PASS`, `CHANGES`, or `BLOCKED_USER`.
- `CHANGES` must list all known required revisions in one packet.
- After one revision round, unresolved scientific disagreement goes to the
  user; stylistic preferences do not block accepted behavior.

## 9. Ping bridge

When enabled, the local Claude CLI and a Codex heartbeat provide notifications:

1. GPT SOL writes a complete packet and invokes Fable's persistent Claude CLI
   session with the packet path.
2. Fable writes/returns a review packet in its own folder.
3. A Codex heartbeat detects the new Fable file and wakes GPT SOL.
4. Message IDs and file hashes prevent duplicate processing.

Safety controls:

- maximum two automatic exchanges per work order;
- one active work order at a time unless explicitly parallel-safe;
- fixed model and per-call spending cap;
- read-only Fable tool permissions by default;
- lock file during an active invocation;
- no automatic commit, push, publication, or GPU launch.

The bridge accelerates notification; it does not relax change control.

## 10. Session-start fast path

At the start of every GPT SOL session:

1. Read the tail of `Fable/DECISIONS.md`.
2. Locate the newest unprocessed Fable work order/review.
3. Check the corresponding GPT SOL reply state.
4. If `READY_FOR_SOL`, begin work immediately under the autonomy defaults.
5. If no new packet exists, continue the highest-priority `IN_PROGRESS` item.

This prevents re-reviewing settled questions and keeps startup to one focused
intake pass.
