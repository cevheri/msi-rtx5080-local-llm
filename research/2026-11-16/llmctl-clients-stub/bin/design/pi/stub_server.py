#!/usr/bin/env python3
"""Recording stub for an OpenAI-compatible chat completions server (Python standard library only).

Every request is written to <log-root>/<scenario>/<seq>-<method>-<path>.json with its path, headers
(secret values masked: scheme kept, token replaced by its length and a sha256 prefix) and full JSON body.

Endpoints:
  GET  /v1/models, /models          -> one model with the configured id
  POST /v1/chat/completions,
       /chat/completions            -> scripted reply, SSE when body.stream is true, JSON otherwise
  POST /__control                   -> update the stub state (scenario, model_id, mode, reasoning_field, ...)
  GET  /__state                     -> current state

Modes for chat completions:
  tools     no tool results yet -> tool call write(hello.py); one tool result -> tool call read(hello.py);
            two or more -> final text quoting the last tool result
  text      always a final text answer
  error400  HTTP 400 with a llama.cpp style exceed_context_size_error body
  error500  HTTP 500 with an OpenAI style server_error body
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SECRET_HEADERS = {"authorization", "proxy-authorization", "x-api-key", "api-key", "openai-api-key", "cookie"}

LOCK = threading.Lock()
STATE = {
    "scenario": "default",
    "model_id": "local",
    "mode": "tools",
    "reasoning_field": "reasoning_content",  # "reasoning_content", "reasoning", or null for none
    "reasoning_text": "STUB-REASONING",
    "file_path": "hello.py",
    "file_content": "print('hello')\n",
    "final_text": "STUB-FINAL",
}
SEQ = {}
LOG_ROOT = None


def mask_header(name, value):
    if name.lower() not in SECRET_HEADERS:
        return value
    scheme, sep, token = value.partition(" ")
    if not sep:
        scheme, token = "", value
    digest = hashlib.sha256(token.encode()).hexdigest()[:12]
    prefix = f"{scheme} " if scheme else ""
    return f"{prefix}***masked*** (len={len(token)}, sha256[:12]={digest})"


def record(handler, body_raw, body_json, response_note):
    with LOCK:
        scenario = STATE["scenario"]
        SEQ[scenario] = SEQ.get(scenario, 0) + 1
        seq = SEQ[scenario]
    safe_path = re.sub(r"[^A-Za-z0-9._-]+", "_", handler.path.strip("/")) or "root"
    directory = os.path.join(LOG_ROOT, scenario)
    os.makedirs(directory, exist_ok=True)
    entry = {
        "seq": seq,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scenario": scenario,
        "method": handler.command,
        "path": handler.path,
        "headers": {k: mask_header(k, v) for k, v in handler.headers.items()},
        "body": body_json if body_json is not None else (body_raw.decode("utf-8", "replace") if body_raw else None),
        "stub_state": {k: STATE[k] for k in ("model_id", "mode", "reasoning_field")},
        "stub_response": response_note,
    }
    with open(os.path.join(directory, f"{seq:03d}-{handler.command}-{safe_path}.json"), "w") as fh:
        json.dump(entry, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def count_tool_results(messages):
    return sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "tool")


def last_tool_text(messages):
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "tool":
            content = m.get("content")
            if isinstance(content, list):
                return "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return content if isinstance(content, str) else json.dumps(content)
    return ""


def plan_reply(body):
    """Decide the scripted assistant turn: (kind, payload, reasoning)."""
    messages = body.get("messages") or []
    mode = STATE["mode"]
    n_tool = count_tool_results(messages)
    reasoning = None
    if STATE["reasoning_field"]:
        reasoning = f"{STATE['reasoning_text']} turn {n_tool + 1}"
    if mode == "tools" and body.get("tools"):
        if n_tool == 0:
            args = {"path": STATE["file_path"], "content": STATE["file_content"]}
            return "tool", {"id": "call_stub_write_1", "name": "write", "arguments": json.dumps(args)}, reasoning
        if n_tool == 1:
            args = {"path": STATE["file_path"]}
            return "tool", {"id": "call_stub_read_1", "name": "read", "arguments": json.dumps(args)}, reasoning
        text = f"{STATE['final_text']}: {STATE['file_path']} contains: {last_tool_text(messages).strip()}"
        return "text", text, reasoning
    return "text", STATE["final_text"], reasoning


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "pi-research-stub/1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[stub] %s %s\n" % (self.address_string(), fmt % args))

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        parsed = None
        if raw:
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
        return raw, parsed

    def _send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _models(self):
        model_id = STATE["model_id"]
        return {"object": "list", "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "stub"}]}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/__state":
            self._send_json(200, {"state": STATE, "seq": SEQ})
            return
        raw, parsed = b"", None
        if path in ("/v1/models", "/models"):
            reply = self._models()
            record(self, raw, parsed, {"status": 200, "body": reply})
            self._send_json(200, reply)
            return
        if path in ("/health", "/v1/health"):
            record(self, raw, parsed, {"status": 200})
            self._send_json(200, {"status": "ok"})
            return
        record(self, raw, parsed, {"status": 404})
        self._send_json(404, {"error": {"message": f"stub: no route {path}", "type": "not_found"}})

    def do_POST(self):
        path = self.path.split("?")[0]
        raw, parsed = self._read_body()
        if path == "/__control":
            with LOCK:
                for key, value in (parsed or {}).items():
                    if key in STATE:
                        STATE[key] = value
                if (parsed or {}).get("reset_seq"):
                    SEQ.pop(STATE["scenario"], None)
            self._send_json(200, {"state": STATE})
            return
        if path not in ("/v1/chat/completions", "/chat/completions"):
            record(self, raw, parsed, {"status": 404})
            self._send_json(404, {"error": {"message": f"stub: no route {path}", "type": "not_found"}})
            return
        body = parsed or {}
        mode = STATE["mode"]
        if mode == "error400":
            err = {"error": {"code": 400, "message": "stub: the request exceeds the available context size, try increasing it",
                             "type": "exceed_context_size_error", "n_prompt_tokens": 140000, "n_ctx": 131072}}
            record(self, raw, parsed, {"status": 400, "body": err})
            self._send_json(400, err)
            return
        if mode == "error500":
            err = {"error": {"code": 500, "message": "stub: internal server error", "type": "server_error"}}
            record(self, raw, parsed, {"status": 500, "body": err})
            self._send_json(500, err)
            return
        if mode == "error400generic":
            err = {"error": {"code": 400, "message": "stub: unsupported parameter", "type": "invalid_request_error"}}
            record(self, raw, parsed, {"status": 400, "body": err})
            self._send_json(400, err)
            return
        kind, payload, reasoning = plan_reply(body)
        model = body.get("model") or STATE["model_id"]
        note = {"status": 200, "stream": bool(body.get("stream")), "kind": kind,
                "payload": payload, "reasoning_field": STATE["reasoning_field"], "reasoning": reasoning}
        record(self, raw, parsed, note)
        if body.get("stream"):
            self._stream(body, model, kind, payload, reasoning)
        else:
            self._single(model, kind, payload, reasoning)

    def _usage(self):
        return {"prompt_tokens": 1000, "completion_tokens": 20, "total_tokens": 1020}

    def _single(self, model, kind, payload, reasoning):
        message = {"role": "assistant", "content": None}
        if reasoning and STATE["reasoning_field"]:
            message[STATE["reasoning_field"]] = reasoning
        if kind == "tool":
            message["tool_calls"] = [{"id": payload["id"], "type": "function",
                                      "function": {"name": payload["name"], "arguments": payload["arguments"]}}]
            finish = "tool_calls"
        else:
            message["content"] = payload
            finish = "stop"
        reply = {"id": "chatcmpl-stub", "object": "chat.completion", "created": int(time.time()), "model": model,
                 "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": self._usage()}
        self._send_json(200, reply)

    def _chunk(self, data):
        payload = f"data: {data}\n\n".encode()
        self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()

    def _stream(self, body, model, kind, payload, reasoning):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        created = int(time.time())

        def chunk(delta, finish=None):
            obj = {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self._chunk(json.dumps(obj))

        chunk({"role": "assistant", "content": None})
        if reasoning and STATE["reasoning_field"]:
            half = len(reasoning) // 2
            chunk({STATE["reasoning_field"]: reasoning[:half]})
            chunk({STATE["reasoning_field"]: reasoning[half:]})
        if kind == "tool":
            chunk({"tool_calls": [{"index": 0, "id": payload["id"], "type": "function",
                                   "function": {"name": payload["name"], "arguments": ""}}]})
            args = payload["arguments"]
            half = len(args) // 2
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[:half]}}]})
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[half:]}}]})
            chunk({}, "tool_calls")
        else:
            chunk({"content": payload})
            chunk({}, "stop")
        if (body.get("stream_options") or {}).get("include_usage"):
            obj = {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [], "usage": self._usage()}
            self._chunk(json.dumps(obj))
        self._chunk("[DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main():
    global LOG_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--port-file", required=True)
    args = parser.parse_args()
    LOG_ROOT = os.path.abspath(args.log_root)
    os.makedirs(LOG_ROOT, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    port = server.server_address[1]
    with open(args.port_file, "w") as fh:
        fh.write(str(port))
    sys.stderr.write(f"[stub] listening on http://127.0.0.1:{port}, logging to {LOG_ROOT}\n")
    sys.stderr.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()
