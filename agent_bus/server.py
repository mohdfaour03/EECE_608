#!/usr/bin/env python3
"""Dependency-free MCP stdio server for the local Fable/SOL agent bus.

The transport implements the MCP 2025-11-25 lifecycle and remains compatible
with MCP 2025-06-18 over newline-delimited UTF-8 JSON-RPC. Tool calls run
independently so a long wait_for_reply call does
not block cancellation or unrelated requests.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bus_core import Bus, BusCancelled  # noqa: E402

SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18")
SERVER_VERSION = "2.0.0"
NOT_INITIALIZED = -32002
REQUEST_CANCELLED = -32800


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOLS = [
    {
        "name": "list_threads",
        "title": "List agent-bus threads",
        "description": "List threads, unread counts, open questions, and agent presence.",
        "inputSchema": _schema({}, []),
    },
    {
        "name": "post_message",
        "title": "Post a bus message",
        "description": "Post a message to a shared thread.",
        "inputSchema": _schema(
            {
                "thread": {"type": "string", "minLength": 1},
                "body": {"type": "string", "minLength": 1},
                "reply_to": {"type": "integer", "minimum": 1},
            },
            ["thread", "body"],
        ),
    },
    {
        "name": "read_messages",
        "title": "Read bus messages",
        "description": "Read unread or all messages in a thread and see questions awaiting your answer.",
        "inputSchema": _schema(
            {
                "thread": {"type": "string", "minLength": 1},
                "include_own": {"type": "boolean", "default": True},
                "only_unread": {"type": "boolean", "default": True},
            },
            ["thread"],
        ),
    },
    {
        "name": "wait_for_reply",
        "title": "Wait for a peer reply",
        "description": "Wait up to 120 seconds for a new peer message or answer notification.",
        "inputSchema": _schema(
            {
                "thread": {"type": "string", "minLength": 1},
                "timeout_seconds": {"type": "number", "minimum": 1, "maximum": 120, "default": 60},
                "poll_seconds": {"type": "number", "minimum": 0.05, "maximum": 5, "default": 2},
            },
            ["thread"],
        ),
    },
    {
        "name": "open_question",
        "title": "Open a blind-review question",
        "description": "Ask a decision question while sealing the asker's recommendation until the peer answers.",
        "inputSchema": _schema(
            {
                "thread": {"type": "string", "minLength": 1},
                "body": {"type": "string", "minLength": 1},
                "options": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 64},
                "recommendation": {"type": "string", "minLength": 1},
                "evidence": {"type": "string", "default": ""},
            },
            ["thread", "body", "options", "recommendation"],
        ),
    },
    {
        "name": "get_question",
        "title": "Get a blind-review question",
        "description": "Fetch a question; the peer's recommendation remains sealed until you answer it.",
        "inputSchema": _schema({"question_id": {"type": "integer", "minimum": 1}}, ["question_id"]),
    },
    {
        "name": "answer_question",
        "title": "Answer a blind-review question",
        "description": "Commit an independent position; only then is the asker's recommendation revealed.",
        "inputSchema": _schema(
            {
                "question_id": {"type": "integer", "minimum": 1},
                "position": {"type": "string", "minLength": 1},
                "reasoning": {"type": "string", "default": ""},
            },
            ["question_id", "position"],
        ),
    },
    {
        "name": "close_question",
        "title": "Close a bus question",
        "description": "Record the asker's final outcome and regenerate the decision-draft ledger atomically.",
        "inputSchema": _schema(
            {"question_id": {"type": "integer", "minimum": 1}, "outcome": {"type": "string", "minLength": 1}},
            ["question_id", "outcome"],
        ),
    },
    {
        "name": "export_transcript",
        "title": "Export a thread transcript",
        "description": "Write an atomic markdown transcript and refresh the decision-draft ledger.",
        "inputSchema": _schema({"thread": {"type": "string", "minLength": 1}}, ["thread"]),
    },
]
TOOL_NAMES = {tool["name"] for tool in TOOLS}
TOOL_KEYS = {
    tool["name"]: set(tool["inputSchema"]["properties"])
    for tool in TOOLS
}
TOOL_REQUIRED = {
    tool["name"]: set(tool["inputSchema"]["required"])
    for tool in TOOLS
}


class ProtocolError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _error_response(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _is_request_id(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool) and value is not None


def _validate_request(request: Any) -> tuple[dict[str, Any], Any, bool]:
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
        raise ProtocolError(-32600, "Invalid Request")
    method = request.get("method")
    if not isinstance(method, str) or not method:
        raise ProtocolError(-32600, "Invalid Request")
    has_id = "id" in request
    request_id = request.get("id")
    if has_id and not _is_request_id(request_id):
        raise ProtocolError(-32600, "Invalid Request")
    if "params" in request and not isinstance(request["params"], (dict, list)):
        raise ProtocolError(-32602, "Invalid params")
    return request, request_id, not has_id


def _tool_args(name: str, args: Any) -> dict[str, Any]:
    if name not in TOOL_NAMES:
        raise ProtocolError(-32602, f"Unknown tool: {name}")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ProtocolError(-32602, "Tool arguments must be an object")
    unknown = set(args) - TOOL_KEYS[name]
    if unknown:
        raise ProtocolError(-32602, f"Unknown argument(s): {', '.join(sorted(unknown))}")
    missing = TOOL_REQUIRED[name] - set(args)
    if missing:
        raise ProtocolError(-32602, f"Missing required argument(s): {', '.join(sorted(missing))}")
    return args


def _validate_tool_types(name: str, args: dict[str, Any]) -> None:
    string_fields = {
        "thread", "body", "recommendation", "evidence", "position", "reasoning", "outcome"
    }
    for field in string_fields & set(args):
        if not isinstance(args[field], str):
            raise ProtocolError(-32602, f"{field} must be a string")
    for field in {"question_id", "reply_to"} & set(args):
        if isinstance(args[field], bool) or not isinstance(args[field], int):
            raise ProtocolError(-32602, f"{field} must be an integer")
    for field in {"include_own", "only_unread"} & set(args):
        if not isinstance(args[field], bool):
            raise ProtocolError(-32602, f"{field} must be a boolean")
    for field in {"timeout_seconds", "poll_seconds"} & set(args):
        if isinstance(args[field], bool) or not isinstance(args[field], (int, float)):
            raise ProtocolError(-32602, f"{field} must be a number")
    if "options" in args and (
        not isinstance(args["options"], list)
        or any(not isinstance(option, str) for option in args["options"])
    ):
        raise ProtocolError(-32602, "options must be an array of strings")


def _dispatch(bus: Bus, name: str, args: dict[str, Any], cancel_event: threading.Event) -> dict[str, Any]:
    args = _tool_args(name, args)
    _validate_tool_types(name, args)
    if name == "list_threads":
        return bus.list_threads()
    if name == "post_message":
        return bus.post_message(args["thread"], args["body"], args.get("reply_to"))
    if name == "read_messages":
        return bus.read_messages(
            args["thread"], include_own=args.get("include_own", True), only_unread=args.get("only_unread", True)
        )
    if name == "wait_for_reply":
        return bus.wait_for_reply(
            args["thread"], args.get("timeout_seconds", 60), args.get("poll_seconds", 2), cancel_event
        )
    if name == "open_question":
        return bus.open_question(
            args["thread"], args["body"], args["options"], args["recommendation"], args.get("evidence", "")
        )
    if name == "get_question":
        return bus.get_question(args["question_id"])
    if name == "answer_question":
        return bus.answer_question(args["question_id"], args["position"], args.get("reasoning", ""))
    if name == "close_question":
        return bus.close_question(args["question_id"], args["outcome"])
    if name == "export_transcript":
        return bus.export_transcript(args["thread"])
    raise ProtocolError(-32602, f"Unknown tool: {name}")


def _result_response(request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "structuredContent": payload,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Bus MCP stdio server")
    parser.add_argument("--agent", required=True, help="Configured agent identity, e.g. FABLE or SOL")
    parser.add_argument("--db", default=str(Path(__file__).resolve().parent / "bus.db"))
    parser.add_argument("--transcripts", default=None, help="Markdown export directory")
    parser.add_argument("--version", action="version", version=SERVER_VERSION)
    opts = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        sys.stderr.reconfigure(encoding="utf-8", newline="\n")
    except AttributeError:
        pass

    send_lock = threading.Lock()
    active_lock = threading.Lock()
    active: dict[Any, threading.Event] = {}
    workers: list[threading.Thread] = []
    # MCP has a small but important lifecycle: initialize, initialized, then
    # normal operation.  Keeping the intermediate state prevents a client
    # from racing a tools/list call ahead of its initialized notification.
    state = {"phase": "new", "stopping": False}

    def send(message: dict[str, Any]) -> None:
        with send_lock:
            sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()

    def run_tool(request_id: Any, name: str, arguments: Any, cancel_event: threading.Event) -> None:
        bus: Bus | None = None
        try:
            bus = Bus(opts.db, opts.agent, transcript_dir=opts.transcripts)
            payload = _dispatch(bus, name, arguments, cancel_event)
            if request_id is not None and not cancel_event.is_set():
                send(_result_response(request_id, payload))
            elif request_id is not None:
                send(_error_response(request_id, REQUEST_CANCELLED, "Request cancelled"))
        except BusCancelled:
            if request_id is not None:
                send(_error_response(request_id, REQUEST_CANCELLED, "Request cancelled"))
        except ProtocolError as exc:
            if request_id is not None:
                send(_error_response(request_id, exc.code, exc.message, exc.data))
        except Exception as exc:  # business/tool failures are tool execution errors
            if request_id is not None:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [{"type": "text", "text": f"Error: {exc}"}],
                            "isError": True,
                        },
                    }
                )
        finally:
            if bus is not None:
                bus.close()
            if request_id is not None:
                with active_lock:
                    active.pop(request_id, None)

    def start_tool(request_id: Any, name: str, arguments: Any) -> None:
        cancel_event = threading.Event()
        if request_id is not None:
            with active_lock:
                if request_id in active:
                    raise ProtocolError(-32600, "Duplicate request id")
                active[request_id] = cancel_event
        worker = threading.Thread(
            target=run_tool, args=(request_id, name, arguments, cancel_event), daemon=True
        )
        workers.append(worker)
        worker.start()

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request, request_id, notification = _validate_request(json.loads(line))
            except json.JSONDecodeError:
                send(_error_response(None, -32700, "Parse error"))
                continue
            except ProtocolError as exc:
                send(_error_response(None, exc.code, exc.message, exc.data))
                continue

            method = request["method"]
            params = request.get("params") if "params" in request else {}
            if state["stopping"]:
                if not notification:
                    send(_error_response(request_id, -32600, "Server is shutting down"))
                break
            if method == "initialize":
                if state["phase"] != "new":
                    if not notification:
                        send(_error_response(request_id, -32600, "Server is already initialized"))
                    continue
                if not isinstance(params, dict):
                    if not notification:
                        send(_error_response(request_id, -32602, "initialize params must be an object"))
                    continue
                requested = params.get("protocolVersion")
                if not isinstance(requested, str):
                    if not notification:
                        send(_error_response(request_id, -32602, "protocolVersion is required"))
                    continue
                # Select the requested revision when available. Otherwise
                # advertise the current revision and let the client decide
                # whether it can continue with that negotiated version.
                negotiated = (
                    requested if requested in SUPPORTED_PROTOCOL_VERSIONS
                    else SUPPORTED_PROTOCOL_VERSIONS[0]
                )
                state["phase"] = "awaiting_initialized"
                if not notification:
                    send(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "protocolVersion": negotiated,
                                "capabilities": {"tools": {}},
                                "serverInfo": {
                                    "name": f"agent-bus ({opts.agent})",
                                    "title": "Agent Bus",
                                    "version": SERVER_VERSION,
                                },
                                "instructions": "Use list_threads at session start. Decision questions must use blind-review tools.",
                            },
                        }
                    )
                continue
            if method == "notifications/initialized":
                if state["phase"] == "awaiting_initialized":
                    state["phase"] = "ready"
                continue
            if method == "notifications/cancelled":
                if isinstance(params, dict):
                    cancelled_id = params.get("requestId")
                    with active_lock:
                        event = active.get(cancelled_id)
                    if event is not None:
                        event.set()
                continue
            if method == "ping":
                if not notification:
                    send({"jsonrpc": "2.0", "id": request_id, "result": {}})
                continue
            if method == "shutdown":
                if state["phase"] == "new":
                    if not notification:
                        send(_error_response(request_id, NOT_INITIALIZED, "Server is not initialized"))
                else:
                    state["stopping"] = True
                    if not notification:
                        send({"jsonrpc": "2.0", "id": request_id, "result": None})
                    break
                continue
            if state["phase"] != "ready":
                if not notification:
                    send(_error_response(request_id, NOT_INITIALIZED, "Server is not initialized"))
                continue
            if method == "tools/list":
                if not isinstance(params, dict):
                    if not notification:
                        send(_error_response(request_id, -32602, "tools/list params must be an object"))
                elif params.get("cursor") not in (None, ""):
                    if not notification:
                        send(_error_response(request_id, -32602, "No tools/list cursor is supported"))
                elif not notification:
                    send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}})
                continue
            if method == "tools/call":
                if not isinstance(params, dict) or not isinstance(params.get("name"), str):
                    if not notification:
                        send(_error_response(request_id, -32602, "tools/call requires a tool name"))
                    continue
                try:
                    arguments = params.get("arguments", {})
                    _tool_args(params["name"], arguments)
                    _validate_tool_types(params["name"], arguments if arguments is not None else {})
                except ProtocolError as exc:
                    if not notification:
                        send(_error_response(request_id, exc.code, exc.message, exc.data))
                    continue
                try:
                    start_tool(None if notification else request_id, params["name"], arguments)
                except ProtocolError as exc:
                    if not notification:
                        send(_error_response(request_id, exc.code, exc.message, exc.data))
                continue
            if not notification:
                send(_error_response(request_id, -32601, f"Method not found: {method}"))
    finally:
        with active_lock:
            for event in active.values():
                event.set()
        for worker in workers:
            worker.join(timeout=2.0)


if __name__ == "__main__":
    main()
