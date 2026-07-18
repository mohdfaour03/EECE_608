# NOTICE — Project relocated out of OneDrive · 2026-07-16

**From:** Fable · **Authority:** Mohamad's direct order this session · **State:** binding (recorded as D-008)

## What happened

1. This session's `agent_bus` MCP tools never loaded: root `.mcp.json` was **truncated to 216 bytes** (unterminated JSON) — same OneDrive partial-sync failure mode as the earlier episode. Fable repaired it (brave-search entry salvaged, agent_bus FABLE entry restored per `agent_bus/SETUP.md`).
2. Security check on Sol's finding: the Brave API key in `.mcp.json` was **never committed** (untracked, not in history, `.gitignore`d line 37 — pattern widened to `.mcp.json*` to also cover backups). Rotation still recommended at leisure; exposure was local-only.
3. Mohamad ordered the project **out of OneDrive entirely**. Fable executed `MOVE_PROJECT.cmd` / `move_project.ps1` (repo root): non-destructive robocopy to `%USERPROFILE%\EECE_608`, auto-repoint of destination `.mcp.json` and `%USERPROFILE%\.codex\config.toml` (backups made), freeze marker in the old tree. See `MIGRATION_REPORT.txt`.

## New canonical facts

- **Canonical path: `%USERPROFILE%\EECE_608`.** The OneDrive tree is a frozen backup — never edit it; delete after verification.
- `bus.db` stays at `%LOCALAPPDATA%\agent_bus\bus.db` (both configs already point there; unchanged).
- OneDrive-corruption countermeasures (out-of-tree db rationale in SETUP.md) are now historical; the db location stands anyway.

## Sol — required actions at next session start

1. Restart Codex CLI with cwd in the NEW folder. Verify `~/.codex/config.toml` was repointed (backup: `config.toml.pre_migration.bak`); fix manually if the patcher reported MISSING.
2. `list_threads` on the bus — Fable will initiate the handshake + F-002 blind-review dry run (D-009 reserved) from its next session, which will have working bus tools.
3. **Task 0 remains BLOCKED_USER** — Mohamad has not yet ruled on your scoped-baseline-commit proposal (Fable endorses it; escalated). The baseline commit happens **in the new location** and will capture the migration artifacts.
4. This session ends without a git commit (Fable's mount pointed at the now-frozen OneDrive copy; committing there would fork history). The pending baseline commit covers it — logged deviation from ground rule 9, reason recorded here.

## Evidence labels

- VERIFIED: `.mcp.json` truncation (216 bytes, JSONDecodeError line 10), repair round-trips as valid JSON, key untracked/not-in-history, `.gitignore` coverage.
- VERIFIED-at-write-time: migration script behavior per its own MIGRATION_REPORT.txt (check robocopy exit code there).
