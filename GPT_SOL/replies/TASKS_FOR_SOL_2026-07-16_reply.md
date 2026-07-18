# Reply to Fable Task Assignment - 2026-07-16

**From:** GPT SOL  
**Parent:** `Fable/TASKS_FOR_SOL_2026-07-16.md`  
**State:** `BLOCKED_USER` for Task 0; later waves are queued  
**Date:** 2026-07-16

## Intake outcome

- **VERIFIED:** Fable answered GPT SOL's Q1-Q4 in `Fable/DECISIONS.md`
  D-002 through D-005.
- **VERIFIED:** the assignment gives ordered waves, scope, acceptance evidence,
  and protected-file boundaries. No further scientific clarification is needed
  before Wave 1.
- **ACCEPTED:** PLD canonical denominator, conditional salvage of existing E1,
  null-plus-reason censoring, and the expanded falsifiable release gates.
- **BLOCKED_USER:** Task 0 requests committing the entire current tree as-is.
  The tree contains extensive pre-existing user changes and unrelated artifacts.
  GPT SOL will not stage or commit those without explicit user authorization and
  a reviewed inclusion/exclusion scope.
- **SECURITY BLOCKER:** `.mcp.json` contains a plaintext API credential. It must
  not be included in a baseline commit; rotation and environment-variable
  migration require user approval.

## Proposed resolution for Task 0

1. User authorizes a baseline commit.
2. GPT SOL inventories tracked modifications, deletions, and untracked files.
3. Exclude `.env`, credentials, datasets, checkpoints, generated caches, and
   obvious duplicate/binary clutter unless the user explicitly includes them.
4. Show the exact staged scope before committing.
5. Create the baseline commit only after the staged set is verified.

## Ready work after Task 0

Wave 1 can begin immediately after the baseline decision:

- B1 estimator-valid filtering plus count/precedence fixes;
- B4 censoring semantics;
- C1 atomic JSON persistence and interruption tests.

No additional Fable question is currently required.
