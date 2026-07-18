# Agent Bus — direct Fable ↔ Sol communication over MCP

A zero-dependency MCP 2025-11-25 server (also accepting 2025-06-18 clients) that
lets the two project agents (Fable / Claude and
Sol / GPT 5.6 Codex) message each other directly, with the collaboration protocol's
hardest rule — **blind review** — enforced by the server instead of by discipline.

Built by Fable, 2026-07-16, at Mohamad's direction. Co-development handoff for Sol:
`HANDOFF_SOL_2026-07-16.md`.

## Why a server and not just shared files

Files already work as a mailbox. The server adds the three things files can't enforce:

1. **Blind review (anti-anchoring).** `open_question` requires the asker's own
   recommendation but *seals* it — the peer cannot read it until they commit an
   independent answer via `answer_question`. Two independent agreeing positions are
   strong evidence; pre-shown recommendations just produce agreeable nodding. This
   was Mohamad's objection to "answer-by-exception" review, turned into a mechanism.
2. **Live back-and-forth.** `wait_for_reply` blocks (≤120 s) until the peer posts,
   so when both agents run simultaneously they hold a real conversation with no
   human relaying.
3. **A ledger, not a chat log.** `close_question` appends the full decision record
   (question, both independent positions, outcome) to
   `transcripts/DECISION_DRAFTS.md`; `export_transcript` dumps any thread to
   markdown. The SQLite file is runtime state; the markdown exports are what gets
   committed to git.

## Transport and hardening

The server implements MCP 2025-06-18 over newline-delimited JSON-RPC stdio.
It handles initialize/initialized, ping, shutdown, tools/list, tools/call,
notifications, JSON-RPC errors, and request cancellation. Tool calls run off
the reader loop, so a long wait does not block cancellation or another call.

Inputs are validated at both the protocol and storage layers. SQLite uses WAL,
a busy timeout, transactions, safe migration of the answer-notification flag,
and atomic markdown replacement. Runtime state can live outside a synced folder
while transcripts remain in the repository.

## Architecture

```
Claude (Cowork)  ── stdio MCP ──  server.py --agent FABLE ──┐
                                                            ├── bus.db (SQLite, WAL)
Codex CLI (Sol)  ── stdio MCP ──  server.py --agent SOL  ───┘
                                                 │
                                                 └── transcripts/*.md  (git)
```

- Each agent launches its **own** server process; identity comes from `--agent` in
  the launch config, never from tool arguments — authorship can't be spoofed.
- Both processes share one SQLite file; WAL mode + 10 s busy timeout handle
  concurrent access.
- Pure Python 3.10+ stdlib. Nothing to install.

## Tools

`list_threads` · `post_message` · `read_messages` · `wait_for_reply` ·
`open_question` (sealed recommendation) · `get_question` · `answer_question`
(reveals the seal) · `close_question` (writes decision draft) · `export_transcript`

## Usage conventions (agreed protocol)

- One thread per topic (`wave1-review`, `salvage-conditions`, …), not per day.
- Start every session with `list_threads` — it shows unread counts and questions
  awaiting your answer.
- Decision questions go through `open_question`/`answer_question`, never plain
  messages — the seal only protects you if you use it.
- On independent disagreement: at most one `post_message` exchange to try to
  converge, then escalate to Mohamad. Never argue for multiple rounds.
- Binding outcomes still get transcribed to `Fable/DECISIONS.md`; the bus draft
  file is the staging area.
- Export transcripts and commit them at session end (project ground rule 9).

## Is this novel?

No — and that's fine; it's a tool, not a paper. Agent-to-agent plumbing exists at
every scale in 2026: Google's A2A protocol (150+ orgs, first-class in the major
clouds), IBM's ACP, and several small MCP message buses (AgentChatBus, agent-comms-mcp,
MCP Talk) that let CLI agents chat through a shared queue. The part we haven't seen
elsewhere is the **server-enforced blind review** — anti-anchoring as a protocol
guarantee rather than a prompt instruction. Same epistemic instinct as this project's
own methodology: agreement is only evidence when the answers were independent.

## Changelog

- **v2.0.0 (2026-07-16)** - Reworked the prototype into a complete MCP stdio
  server with lifecycle negotiation, strict tool schemas, structured results,
  cancellation, concurrent request handling, duplicate-request protection,
  graceful shutdown, validation, migr