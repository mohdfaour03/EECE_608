# Agent Bus — Co-development Handoff to Sol

**From:** Fable · 2026-07-16 · Mohamad asked us to build this together. I built v1;
you own the review and the next increment. This is deliberately a small side
project — timebox it, the DP paper comes first.

## What v1 is

Read `README.md` (design + rationale), then `bus_core.py`, `server.py`, `test_bus.py`
(10/10 passing). Core idea: MCP message bus with **server-enforced blind review** —
`open_question` seals the asker's recommendation until the responder commits an
independent answer. This replaces the rejected "answer-by-exception" review model
(anchoring risk — Mohamad's call, and he was right).

## Your tasks

1. **Adversarial review of v1** (VERIFIED-level, per your evidence labels):
   - Can the seal be leaked by any tool path? (I checked `get_question`,
     `read_messages`, `export_transcript` — check again, and check error messages.)
   - Concurrency: two processes, WAL mode — hammer it (e.g., 200 interleaved posts
     from two processes) and confirm no lost writes or `database is locked`.
   - MCP compliance from Codex CLI's side: does the handshake work with your
     client? I hand-rolled the transport (newline-delimited JSON-RPC); if Codex
     requires anything I didn't implement (e.g., `notifications/progress`,
     capability fields), fix `server.py` — it's not protected code.
2. **Wire your side** per `SETUP.md` §1 and run the two-terminal smoke test.
3. **First live exchange:** when Mohamad runs us both, post to thread `handshake`;
   I'll be waiting on `wait_for_reply`. Then we do one blind-review dry run on a
   stakes-free question.
4. **Optional v1.1 (only if cheap):** a `summarize_thread` guard — reject a
   `post_message` that opens a duplicate of an existing open question (repetition
   guard from the protocol discussion). Skip if it takes more than an hour.

## Boundaries

- `agent_bus/` is shared code — normal review loop applies (STATUS doc → my review),
  but it is NOT load-bearing for the paper, so we can iterate fast.
- Don't put bus runtime state in git: `bus.db*` is gitignored; export transcripts
  instead (`export_transcript`) and commit those.
- Usage conventions are in `README.md` §Usage — they're part of the collaboration
  protocol now (see `Fable/DECISIONS.md` D-007).
