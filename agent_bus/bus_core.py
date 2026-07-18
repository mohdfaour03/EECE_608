"""SQLite-backed core for the two-agent collaboration bus.

The core is intentionally independent of MCP transport details.  It provides
transactional message/question operations, server-enforced blind review,
one-shot answer notifications, safe migrations, and atomic markdown exports.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread TEXT NOT NULL,
    author TEXT NOT NULL,
    body TEXT NOT NULL,
    reply_to INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread, id);

CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread TEXT NOT NULL,
    asker TEXT NOT NULL,
    body TEXT NOT NULL,
    options TEXT NOT NULL,
    evidence TEXT,
    sealed_recommendation TEXT NOT NULL,
    responder TEXT,
    response TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    asker_notified INTEGER NOT NULL DEFAULT 0,
    outcome TEXT,
    created_at TEXT NOT NULL,
    answered_at TEXT,
    closed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_questions_thread_status ON questions(thread, status);

CREATE TABLE IF NOT EXISTS cursors (
    agent TEXT NOT NULL,
    thread TEXT NOT NULL,
    last_read_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (agent, thread)
);

CREATE TABLE IF NOT EXISTS presence (
    agent TEXT PRIMARY KEY,
    last_seen TEXT NOT NULL
);
"""

MAX_AGENT_LENGTH = 100
MAX_THREAD_LENGTH = 200
MAX_BODY_LENGTH = 1_000_000
MAX_OPTION_LENGTH = 2_000
MAX_OPTIONS = 64


class BusCancelled(Exception):
    """Raised when a cancellable bus operation receives a cancellation signal."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_text(value: Any, field: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if len(value) > max_length:
        raise ValueError(f"{field} exceeds the {max_length}-character limit")
    return value


def _require_optional_text(value: Any, field: str, *, max_length: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if len(value) > max_length:
        raise ValueError(f"{field} exceeds the {max_length}-character limit")
    return value


class Bus:
    def __init__(
        self,
        db_path: str | Path,
        agent: str,
        transcript_dir: str | Path | None = None,
    ):
        self.agent = _require_text(agent, "agent name", max_length=MAX_AGENT_LENGTH)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_dir = Path(transcript_dir) if transcript_dir else self.db_path.parent / "transcripts"
        self._conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        columns = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(questions)").fetchall()
        }
        if "asker_notified" not in columns:
            self._conn.execute(
                "ALTER TABLE questions ADD COLUMN asker_notified INTEGER NOT NULL DEFAULT 0"
            )

    def _touch_presence(self) -> None:
        self._conn.execute(
            "INSERT INTO presence(agent, last_seen) VALUES(?, ?) "
            "ON CONFLICT(agent) DO UPDATE SET last_seen=excluded.last_seen",
            (self.agent, _now()),
        )
        self._conn.commit()

    @staticmethod
    def _thread(value: Any) -> str:
        return _require_text(value, "thread", max_length=MAX_THREAD_LENGTH)

    @staticmethod
    def _question_options(options: Any) -> list[str]:
        if not isinstance(options, list) or not 2 <= len(options) <= MAX_OPTIONS:
            raise ValueError(f"options must be a list containing 2-{MAX_OPTIONS} items")
        cleaned: list[str] = []
        for option in options:
            cleaned.append(_require_text(option, "option", max_length=MAX_OPTION_LENGTH))
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("options must be unique")
        return cleaned

    def _question_view(self, row: sqlite3.Row) -> dict[str, Any]:
        question = dict(row)
        try:
            question["options"] = json.loads(question["options"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"question {question.get('id')} has invalid stored options") from exc
        if question["status"] not in ("answered", "closed") and question["asker"] != self.agent:
            question["sealed_recommendation"] = "[SEALED until you answer — call answer_question first]"
        return question

    # -- messages ---------------------------------------------------------

    def post_message(self, thread: str, body: str, reply_to: int | None = None) -> dict[str, Any]:
        self._touch_presence()
        thread = self._thread(thread)
        body = _require_text(body, "body", max_length=MAX_BODY_LENGTH)
        if reply_to is not None:
            if isinstance(reply_to, bool) or not isinstance(reply_to, int) or reply_to <= 0:
                raise ValueError("reply_to must be a positive integer")
            parent = self._conn.execute(
                "SELECT thread FROM messages WHERE id=?", (reply_to,)
            ).fetchone()
            if parent is None:
                raise ValueError(f"no message with id {reply_to}")
            if parent["thread"] != thread:
                raise ValueError("reply_to must reference a message in the same thread")
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO messages(thread, author, body, reply_to, created_at) VALUES(?,?,?,?,?)",
                (thread, self.agent, body, reply_to, _now()),
            )
        return {"message_id": cur.lastrowid, "thread": thread}

    def read_messages(
        self, thread: str, include_own: bool = True, only_unread: bool = True
    ) -> dict[str, Any]:
        self._touch_presence()
        thread = self._thread(thread)
        if not isinstance(include_own, bool) or not isinstance(only_unread, bool):
            raise ValueError("include_own and only_unread must be booleans")
        row = self._conn.execute(
            "SELECT last_read_id FROM cursors WHERE agent=? AND thread=?",
            (self.agent, thread),
        ).fetchone()
        since = row["last_read_id"] if row and only_unread else 0
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE thread=? AND id>? ORDER BY id", (thread, since)
        ).fetchall()
        messages = [dict(item) for item in rows if include_own or item["author"] != self.agent]
        if rows:
            last_id = rows[-1]["id"]
            with self._conn:
                self._conn.execute(
                    "INSERT INTO cursors(agent, thread, last_read_id) VALUES(?,?,?) "
                    "ON CONFLICT(agent, thread) DO UPDATE SET last_read_id=excluded.last_read_id",
                    (self.agent, thread, last_id),
                )
        pending = self._conn.execute(
            "SELECT id, body FROM questions WHERE thread=? AND status='open' AND asker<>? ORDER BY id",
            (thread, self.agent),
        ).fetchall()
        return {
            "messages": messages,
            "open_questions_for_you": [dict(item) for item in pending],
        }

    def list_threads(self) -> dict[str, Any]:
        self._touch_presence()
        rows = self._conn.execute(
            "SELECT thread, COUNT(*) AS n, MAX(created_at) AS last_activity FROM ("
            "SELECT thread, created_at FROM messages UNION ALL "
            "SELECT thread, created_at FROM questions"
            ") GROUP BY thread ORDER BY last_activity DESC, thread"
        ).fetchall()
        threads = []
        for row in rows:
            cursor = self._conn.execute(
                "SELECT last_read_id FROM cursors WHERE agent=? AND thread=?",
                (self.agent, row["thread"]),
            ).fetchone()
            since = cursor["last_read_id"] if cursor else 0
            unread = self._conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE thread=? AND id>? AND author<>?",
                (row["thread"], since, self.agent),
            ).fetchone()["c"]
            pending = self._conn.execute(
                "SELECT COUNT(*) AS c FROM questions WHERE thread=? AND status='open' AND asker<>?",
                (row["thread"], self.agent),
            ).fetchone()["c"]
            answered = self._conn.execute(
                "SELECT COUNT(*) AS c FROM questions WHERE thread=? AND status='answered' "
                "AND asker=? AND asker_notified=0",
                (row["thread"], self.agent),
            ).fetchone()["c"]
            threads.append(
                {
                    **dict(row),
                    "unread_from_peer": unread,
                    "open_questions_for_you": pending,
                    "unread_answer_notifications": answered,
                }
            )
        open_questions = self._conn.execute(
            "SELECT id, thread, asker, status FROM questions WHERE status<>'closed' ORDER BY id"
        ).fetchall()
        presence = self._conn.execute("SELECT * FROM presence ORDER BY agent").fetchall()
        return {
            "you_are": self.agent,
            "threads": threads,
            "open_questions": [dict(item) for item in open_questions],
            "presence": [dict(item) for item in presence],
        }

    def wait_for_reply(
        self,
        thread: str,
        timeout_seconds: int | float = 60,
        poll_seconds: float = 2.0,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        self._touch_presence()
        thread = self._thread(thread)
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be a number")
        if not math.isfinite(float(timeout_seconds)):
            raise ValueError("timeout_seconds must be finite")
        if isinstance(poll_seconds, bool) or not isinstance(poll_seconds, (int, float)):
            raise ValueError("poll_seconds must be a number")
        if not math.isfinite(float(poll_seconds)):
            raise ValueError("poll_seconds must be finite")
        timeout = max(1.0, min(float(timeout_seconds), 120.0))
        poll = max(0.05, min(float(poll_seconds), 5.0))
        cursor = self._conn.execute(
            "SELECT last_read_id FROM cursors WHERE agent=? AND thread=?",
            (self.agent, thread),
        ).fetchone()
        since = cursor["last_read_id"] if cursor else 0
        deadline = time.monotonic() + timeout
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise BusCancelled()
            peer_message = self._conn.execute(
                "SELECT 1 FROM messages WHERE thread=? AND id>? AND author<>? LIMIT 1",
                (thread, since, self.agent),
            ).fetchone()
            fresh_answers = self._conn.execute(
                "SELECT id FROM questions WHERE thread=? AND asker=? AND status='answered' "
                "AND asker_notified=0 ORDER BY id",
                (thread, self.agent),
            ).fetchall()
            if peer_message or fresh_answers:
                result = {"timed_out": False, **self.read_messages(thread)}
                if fresh_answers:
                    ids = [row["id"] for row in fresh_answers]
                    placeholders = ",".join("?" for _ in ids)
                    with self._conn:
                        self._conn.execute(
                            f"UPDATE questions SET asker_notified=1 "
                            f"WHERE id IN ({placeholders}) AND asker_notified=0",
                            ids,
                        )
                    result["answered_question_ids"] = ids
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"timed_out": True, "hint": "Peer has not replied; proceed and check later."}
            if cancel_event is not None:
                cancel_event.wait(min(poll, remaining))
            else:
                time.sleep(min(poll, remaining))

    # -- blind-review questions ------------------------------------------

    def open_question(
        self,
        thread: str,
        body: str,
        options: list[str],
        recommendation: str,
        evidence: str = "",
    ) -> dict[str, Any]:
        self._touch_presence()
        thread = self._thread(thread)
        body = _require_text(body, "question body", max_length=MAX_BODY_LENGTH)
        options = self._question_options(options)
        recommendation = _require_text(
            recommendation, "recommendation", max_length=MAX_BODY_LENGTH
        )
        evidence = _require_optional_text(evidence, "evidence", max_length=MAX_BODY_LENGTH)
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO questions(thread, asker, body, options, evidence, "
                "sealed_recommendation, created_at) VALUES(?,?,?,?,?,?,?)",
                (thread, self.agent, body, json.dumps(options, ensure_ascii=False), evidence, recommendation, _now()),
            )
        return {"question_id": cur.lastrowid, "note": "recommendation sealed from peer"}

    def get_question(self, question_id: int) -> dict[str, Any]:
        self._touch_presence()
        if isinstance(question_id, bool) or not isinstance(question_id, int) or question_id <= 0:
            raise ValueError("question_id must be a positive integer")
        row = self._conn.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
        if row is None:
            raise ValueError(f"no question with id {question_id}")
        if row["asker"] == self.agent and row["status"] == "answered" and not row["asker_notified"]:
            with self._conn:
                self._conn.execute(
                    "UPDATE questions SET asker_notified=1 WHERE id=? AND asker_notified=0",
                    (question_id,),
                )
        return self._question_view(row)

    def answer_question(
        self, question_id: int, position: str, reasoning: str = ""
    ) -> dict[str, Any]:
        self._touch_presence()
        if isinstance(question_id, bool) or not isinstance(question_id, int) or question_id <= 0:
            raise ValueError("question_id must be a positive integer")
        position = _require_text(position, "position", max_length=MAX_BODY_LENGTH)
        reasoning = _require_optional_text(reasoning, "reasoning", max_length=MAX_BODY_LENGTH)
        row = self._conn.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
        if row is None:
            raise ValueError(f"no question with id {question_id}")
        if row["asker"] == self.agent:
            raise ValueError("you cannot answer your own question")
        if row["status"] != "open":
            raise ValueError(f"question {question_id} is already {row['status']}")
        full = position if not reasoning else f"{position}\n\nReasoning: {reasoning}"
        with self._conn:
            cur = self._conn.execute(
                "UPDATE questions SET responder=?, response=?, status='answered', "
                "answered_at=? WHERE id=? AND status='open'",
                (self.agent, full, _now(), question_id),
            )
            if cur.rowcount != 1:
                raise ValueError(f"question {question_id} was answered concurrently")
        return {
            "question_id": question_id,
            "your_position": full,
            "askers_sealed_recommendation_now_revealed": row["sealed_recommendation"],
            "next_step": (
                "Compare the two positions. If they independently agree, the asker "
                "should close_question with the agreed outcome. If they disagree, "
                "discuss via post_message; if still unresolved after one exchange, "
                "escalate to Mohamad."
            ),
        }

    def close_question(self, question_id: int, outcome: str) -> dict[str, Any]:
        self._touch_presence()
        if isinstance(question_id, bool) or not isinstance(question_id, int) or question_id <= 0:
            raise ValueError("question_id must be a positive integer")
        outcome = _require_text(outcome, "outcome", max_length=MAX_BODY_LENGTH)
        row = self._conn.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
        if row is None:
            raise ValueError(f"no question with id {question_id}")
        if row["asker"] != self.agent:
            raise ValueError("only the asker may close a question")
        if row["status"] == "closed":
            raise ValueError("already closed")
        if row["status"] != "answered":
            raise ValueError("question must be answered before it can be closed")
        with self._conn:
            cur = self._conn.execute(
                "UPDATE questions SET status='closed', outcome=?, closed_at=? "
                "WHERE id=? AND status<>'closed' AND asker=?",
                (outcome, _now(), question_id, self.agent),
            )
            if cur.rowcount != 1:
                raise ValueError(f"question {question_id} was closed concurrently")
        draft = self._write_decision_drafts()
        return {"question_id": question_id, "decision_draft": str(draft)}

    # -- export -----------------------------------------------------------

    def _atomic_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temp_name = handle.name
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if temp_name:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass

    def _write_decision_drafts(self) -> Path:
        rows = self._conn.execute(
            "SELECT * FROM questions WHERE status='closed' ORDER BY id"
        ).fetchall()
        lines: list[str] = []
        for row in rows:
            lines.extend(
                [
                    f"\n## Q{row['id']} · {row['thread']} · closed {row['closed_at']} by {self.agent}\n",
                    f"**Question ({row['asker']}):** {row['body']}\n",
                    f"**{row['asker']} (sealed) recommendation:** {row['sealed_recommendation']}\n",
                    f"**{row['responder'] or '—'} independent position:** {row['response'] or '—'}\n",
                    f"**Outcome:** {row['outcome']}\n",
                ]
            )
        path = self.transcript_dir / "DECISION_DRAFTS.md"
        self._atomic_write(path, "\n".join(lines))
        return path

    def export_transcript(self, thread: str) -> dict[str, Any]:
        self._touch_presence()
        thread = self._thread(thread)
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE thread=? ORDER BY id", (thread,)
        ).fetchall()
        questions = self._conn.execute(
            "SELECT * FROM questions WHERE thread=? ORDER BY id", (thread,)
        ).fetchall()
        lines = [f"# Thread: {thread}", ""]
        for row in rows:
            ref = f" (reply to #{row['reply_to']})" if row["reply_to"] else ""
            lines.extend([f"**#{row['id']} {row['author']}** · {row['created_at']}{ref}", "", row["body"], ""])
        for row in questions:
            view = self._question_view(row)
            lines.extend(
                [
                    "---", f"### Question Q{row['id']} [{row['status']}] by {row['asker']}", "",
                    row["body"], "", f"Options: {', '.join(view['options'])}", "",
                    f"Sealed recommendation: {view['sealed_recommendation']}", "",
                    f"Response ({row['responder'] or '—'}): {row['response'] or '—'}", "",
                    f"Outcome: {row['outcome'] or '—'}", "",
                ]
            )
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in thread)
        path = self.transcript_dir / f"{safe}.md"
        self._atomic_write(path, "\n".join(lines))
        self._write_decision_drafts()
        return {"path": str(path), "messages": len(rows), "questions": len(questions)}

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None
