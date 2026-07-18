"""Tests for the agent bus core + a JSON-RPC smoke test of server.py.

Run:  python agent_bus/test_bus.py
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bus_core import Bus  # noqa: E402


class BusCoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "bus.db"
        self.fable = Bus(self.db, "FABLE")
        self.sol = Bus(self.db, "SOL")

    def tearDown(self):
        self.fable.close()
        self.sol.close()
        self.tmp.cleanup()

    def test_message_roundtrip_and_unread_cursor(self):
        self.fable.post_message("t1", "hello sol")
        got = self.sol.read_messages("t1")
        self.assertEqual(len(got["messages"]), 1)
        self.assertEqual(got["messages"][0]["author"], "FABLE")
        # Cursor advanced: nothing unread now.
        again = self.sol.read_messages("t1")
        self.assertEqual(again["messages"], [])

    def test_blind_review_seal(self):
        q = self.fable.open_question(
            "t2", "Which interval?", ["wilson", "clopper_pearson"],
            recommendation="clopper_pearson", evidence="see F-002",
        )
        qid = q["question_id"]
        # Peer cannot see the seal before answering.
        view = self.sol.get_question(qid)
        self.assertIn("SEALED", view["sealed_recommendation"])
        # Asker can see their own seal.
        own = self.fable.get_question(qid)
        self.assertEqual(own["sealed_recommendation"], "clopper_pearson")
        # Answering reveals it.
        ans = self.sol.answer_question(qid, "clopper_pearson", "exact coverage")
        self.assertEqual(ans["askers_sealed_recommendation_now_revealed"], "clopper_pearson")
        # Now visible to everyone.
        after = self.sol.get_question(qid)
        self.assertEqual(after["sealed_recommendation"], "clopper_pearson")

    def test_cannot_answer_own_question(self):
        qid = self.fable.open_question(
            "t3", "q?", ["a", "b"], recommendation="a")["question_id"]
        with self.assertRaises(ValueError):
            self.fable.answer_question(qid, "a")

    def test_cannot_answer_twice(self):
        qid = self.fable.open_question(
            "t4", "q?", ["a", "b"], recommendation="a")["question_id"]
        self.sol.answer_question(qid, "b")
        with self.assertRaises(ValueError):
            self.sol.answer_question(qid, "a")

    def test_recommendation_required(self):
        with self.assertRaises(ValueError):
            self.fable.open_question("t5", "q?", ["a", "b"], recommendation="  ")

    def test_cannot_close_before_peer_answers(self):
        qid = self.fable.open_question(
            "t5b", "q?", ["a", "b"], recommendation="a"
        )["question_id"]
        with self.assertRaises(ValueError):
            self.fable.close_question(qid, "premature")

    def test_close_writes_decision_draft(self):
        qid = self.fable.open_question(
            "t6", "q?", ["a", "b"], recommendation="a")["question_id"]
        self.sol.answer_question(qid, "a")
        out = self.fable.close_question(qid, "Agreed: a")
        draft = Path(out["decision_draft"]).read_text(encoding="utf-8")
        self.assertIn("Q%d" % qid, draft)
        self.assertIn("Agreed: a", draft)

    def test_wait_for_reply_times_out_fast(self):
        res = self.fable.wait_for_reply("empty", timeout_seconds=1, poll_seconds=0.2)
        self.assertTrue(res["timed_out"])

    def test_wait_for_reply_sees_peer_message(self):
        self.sol.post_message("t7", "already here")
        res = self.fable.wait_for_reply("t7", timeout_seconds=5, poll_seconds=0.2)
        self.assertFalse(res["timed_out"])
        self.assertEqual(res["messages"][0]["body"], "already here")

    def test_wait_for_reply_answer_notification_fires_once(self):
        """Regression: Sol's 2026-07-16 finding — an answered-but-unclosed question
        must wake the asker's wait exactly once, not forever."""
        qid = self.fable.open_question(
            "t9", "q?", ["a", "b"], recommendation="a")["question_id"]
        self.sol.answer_question(qid, "b")
        first = self.fable.wait_for_reply("t9", timeout_seconds=5, poll_seconds=0.2)
        self.assertFalse(first["timed_out"])
        self.assertEqual(first["answered_question_ids"], [qid])
        second = self.fable.wait_for_reply("t9", timeout_seconds=1, poll_seconds=0.2)
        self.assertTrue(second["timed_out"])

    def test_get_question_by_asker_consumes_answer_notification(self):
        qid = self.fable.open_question(
            "t10", "q?", ["a", "b"], recommendation="a")["question_id"]
        self.sol.answer_question(qid, "b")
        self.fable.get_question(qid)  # asker reads the answer directly
        res = self.fable.wait_for_reply("t10", timeout_seconds=1, poll_seconds=0.2)
        self.assertTrue(res["timed_out"])

    def test_peer_wait_unaffected_by_answer_notifications(self):
        qid = self.fable.open_question(
            "t11", "q?", ["a", "b"], recommendation="a")["question_id"]
        self.sol.answer_question(qid, "b")
        # SOL is not the asker: no phantom wake-up for it either.
        res = self.sol.wait_for_reply("t11", timeout_seconds=1, poll_seconds=0.2)
        self.assertTrue(res["timed_out"])

    def test_export_transcript_hides_open_seals(self):
        self.fable.open_question("t8", "q?", ["a", "b"], recommendation="a")
        self.sol.post_message("t8", "thinking...")
        out = self.sol.export_transcript("t8")  # exported by the NON-asker
        text = Path(out["path"]).read_text(encoding="utf-8")
        self.assertIn("SEALED", text)
        self.assertNotIn("Sealed recommendation: a\n", text)

    def test_reply_to_must_be_existing_message_in_same_thread(self):
        with self.assertRaises(ValueError):
            self.fable.post_message("reply", "orphan", reply_to=999)
        message_id = self.fable.post_message("reply", "parent")["message_id"]
        with self.assertRaises(ValueError):
            self.fable.post_message("other", "wrong thread", reply_to=message_id)

    def test_question_only_thread_is_listed(self):
        self.fable.open_question("question-only", "q?", ["a", "b"], recommendation="a")
        listed = self.sol.list_threads()
        self.assertEqual([item["thread"] for item in listed["threads"]], ["question-only"])
        self.assertEqual(listed["threads"][0]["open_questions_for_you"], 1)

    def test_external_transcript_directory_and_atomic_cleanup(self):
        with tempfile.TemporaryDirectory() as external:
            bus = Bus(self.db, "FABLE", transcript_dir=Path(external))
            try:
                bus.post_message("safe/name", "hello")
                result = bus.export_transcript("safe/name")
            finally:
                bus.close()
            transcript = Path(result["path"])
            self.assertEqual(transcript.parent, Path(external))
            self.assertTrue(transcript.exists())
            self.assertFalse(any(Path(external).glob(".*.tmp")))

    def test_old_schema_is_migrated(self):
        old_schema = """
        CREATE TABLE questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, thread TEXT NOT NULL,
            asker TEXT NOT NULL, body TEXT NOT NULL, options TEXT NOT NULL,
            evidence TEXT, sealed_recommendation TEXT NOT NULL,
            responder TEXT, response TEXT, status TEXT NOT NULL DEFAULT 'open',
            outcome TEXT, created_at TEXT NOT NULL, answered_at TEXT, closed_at TEXT
        );
        """
        self.fable.close()
        self.sol.close()
        self.db.unlink()
        conn = sqlite3.connect(self.db)
        try:
            conn.executescript(old_schema)
        finally:
            conn.close()
        migrated = Bus(self.db, "FABLE")
        try:
            columns = {row[1] for row in migrated._conn.execute("PRAGMA table_info(questions)")}
            self.assertIn("asker_notified", columns)
        finally:
            migrated.close()
        self.fable = Bus(self.db, "FABLE")
        self.sol = Bus(self.db, "SOL")


class ServerSmokeTest(unittest.TestCase):
    """Speak newline-delimited JSON-RPC to server.py exactly like an MCP client."""

    def test_initialize_list_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "bus.db")
            proc = subprocess.Popen(
                [sys.executable, str(HERE / "server.py"), "--agent", "FABLE", "--db", db],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            )
            def rpc(obj):
                proc.stdin.write(json.dumps(obj) + "\n")
                proc.stdin.flush()
                return json.loads(proc.stdout.readline())

            init = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0"}}})
            self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")

            proc.stdin.write(json.dumps(
                {"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
            proc.stdin.flush()

            tools = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            names = [t["name"] for t in tools["result"]["tools"]]
            self.assertIn("open_question", names)
            self.assertIn("wait_for_reply", names)

            post = rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "post_message",
                                   "arguments": {"thread": "smoke", "body": "hi"}}})
            payload = json.loads(post["result"]["content"][0]["text"])
            self.assertEqual(payload["message_id"], 1)

            bad = rpc({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                       "params": {"name": "answer_question",
                                  "arguments": {"question_id": 99, "position": "x"}}})
            self.assertTrue(bad["result"]["isError"])

            proc.stdin.close()
            proc.wait(timeout=10)
            proc.stdout.close()

    def test_lifecycle_validation_and_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.Popen(
                [sys.executable, str(HERE / "server.py"), "--agent", "FABLE", "--db", str(Path(tmp) / "bus.db")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            )

            def rpc(obj):
                proc.stdin.write(json.dumps(obj) + "\n")
                proc.stdin.flush()
                return json.loads(proc.stdout.readline())

            try:
                before = rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
                self.assertEqual(before["error"]["code"], -32002)
                unsupported = rpc({
                    "jsonrpc": "2.0", "id": 2, "method": "initialize",
                    "params": {"protocolVersion": "1999-01-01"},
                })
                self.assertEqual(unsupported["result"]["protocolVersion"], "2025-11-25")
                proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
                proc.stdin.flush()
                shutdown = rpc({"jsonrpc": "2.0", "id": 2.5, "method": "shutdown"})
                self.assertIsNone(shutdown["result"])

                # A fresh process is used for the normal supported-version path.
                proc.stdin.close()
                proc.wait(timeout=10)
                proc.stdout.close()
                proc = subprocess.Popen(
                    [sys.executable, str(HERE / "server.py"), "--agent", "FABLE", "--db", str(Path(tmp) / "bus2.db")],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                )
                init = rpc({
                    "jsonrpc": "2.0", "id": 3, "method": "initialize",
                    "params": {"protocolVersion": "2025-11-25"},
                })
                self.assertEqual(init["result"]["protocolVersion"], "2025-11-25")
                early = rpc({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
                self.assertEqual(early["error"]["code"], -32002)
                proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
                proc.stdin.flush()
                shutdown = rpc({"jsonrpc": "2.0", "id": 5, "method": "shutdown"})
                self.assertIsNone(shutdown["result"])
            finally:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=10)
                proc.stdout.close()

    def test_wait_for_reply_can_be_cancelled_over_json_rpc(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.Popen(
                [sys.executable, str(HERE / "server.py"), "--agent", "FABLE", "--db", str(Path(tmp) / "bus.db")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            )

            def send(obj):
                proc.stdin.write(json.dumps(obj) + "\n")
                proc.stdin.flush()

            def read_one(output):
                output.append(json.loads(proc.stdout.readline()))

            try:
                send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": "2025-06-18"}})
                self.assertEqual(json.loads(proc.stdout.readline())["id"], 1)
                send({"jsonrpc": "2.0", "method": "notifications/initialized"})
                send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                      "params": {"name": "wait_for_reply",
                                 "arguments": {"thread": "cancel", "timeout_seconds": 120, "poll_seconds": 5}}})
                time.sleep(0.2)
                send({"jsonrpc": "2.0", "method": "notifications/cancelled",
                      "params": {"requestId": 2}})
                output = []
                reader = threading.Thread(target=read_one, args=(output,), daemon=True)
                reader.start()
                reader.join(timeout=5)
                self.assertFalse(reader.is_alive(), "cancelled tool call did not return")
                self.assertEqual(output[0]["error"]["code"], -32800)
            finally:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=10)
                proc.stdout.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
