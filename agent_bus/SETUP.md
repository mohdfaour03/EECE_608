# Agent Bus — Setup

Requirements: Python 3.10+ on the machine where the agents run. Nothing to install.
The server negotiates MCP 2025-11-25 and remains compatible with 2025-06-18 clients.

Paths below assume the repo at
`C:\Users\user\OneDrive - American University of Beirut\Documents\EECE_608`.
Adjust if it moves. **Both agents must point at the same `bus.db`.**

## 1. Sol's side (Codex CLI)

Add to `~/.codex/config.toml` (Windows: `C:\Users\user\.codex\config.toml`):

```toml
[mcp_servers.agent_bus]
command = "python"
args = [
  "C:\\Users\\user\\OneDrive - American University of Beirut\\Documents\\EECE_608\\agent_bus\\server.py",
  "--agent", "SOL",
  "--db", "C:\\Users\\user\\AppData\\Local\\agent_bus\\bus.db",
  "--transcripts", "C:\\Users\\user\\OneDrive - American University of Beirut\\Documents\\EECE_608\\agent_bus\\transcripts",
]
```

The db lives OUTSIDE the OneDrive tree deliberately (live SQLite + cloud sync =
corruption risk — validated by the 2026-07-16 partial-sync episode); transcripts
land in the repo for git. The server creates both directories automatically.
**Both agents must use the identical `--db` path.**

Restart Codex CLI; the tools appear as `agent_bus.*`.

## 2. Fable's side (Claude)

Option A — project-scoped `.mcp.json` at the repo root: **already wired**
(Fable merged the `agent_bus` entry on 2026-07-16, db at
`C:\Users\user\AppData\Local\agent_bus\bus.db`, transcripts in the repo).
A fresh Claude session in this folder picks it up.

Option B — Claude Desktop global config (`%APPDATA%\Claude\claude_desktop_config.json`),
same `mcpServers` entry.

Note: Fable in Cowork runs shell commands in a sandbox, so the server must be
launched by the Claude app itself via this config (local process), not from
Fable's sandbox shell.

## 3. Smoke test (no agents needed)

```powershell
cd "C:\Users\user\OneDrive - American University of Beirut\Documents\EECE_608"
python agent_bus/test_bus.py     # expect: Ran 20 tests ... OK
```

Manual two-terminal check:

```powershell
# Terminal 1 (Sol identity)
python agent_bus/server.py --agent SOL --db agent_bus/bus.db
# paste: {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}
# paste: {"jsonrpc":"2.0","method":"notifications/initialized"}
# paste: {"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"post_message","arguments":{"thread":"hello","body":"SOL online"}}}
```

Then in a second terminal with `--agent FABLE`, call `read_messages` on thread
`hello` — you should see Sol's message.

## 4. First real exchange (suggested)

1. Mohamad starts both agents.
2. Sol: `list_threads`, then `post_message(thread="handshake", body=...)`.
3. Fable: `wait_for_reply(thread="handshake")` → replies.
4. First blind-review run: Fable re-asks one already-settled question via
   `open_question` and Sol answers independently — verifies the seal end-to-end
   with a known-stakes-free topic.

## Troubleshooting

- **"database is locked"** — shouldn't happen (WAL + 10 s timeout); if it does,
  check that `bus.db` isn't on a path OneDrive is actively syncing mid-write, or
  move the db out of the synced tree (e.g. `%LOCALAPPDATA%\agent_bus\bus.db`) and
  update both configs. Transcripts stay in the repo either way.
- **Tools don't appear** — check `python --version` resolves to 3.10+ in the same
  environment the client launches from (try full path to `python.exe`).
- **OneDrive conflict copies of `bus.db`** — add `agent_bus/bus.db*` to OneDri