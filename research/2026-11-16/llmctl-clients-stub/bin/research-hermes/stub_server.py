#!/usr/bin/env python3
"""OpenAI-compatible stub server that records every request a client sends.

Standard library only. Emulates the endpoints a client probes on a llama.cpp or vLLM server
and answers /v1/chat/completions with a scripted conversation:
  1. a write_file tool call that creates hello.py,
  2. after the write result comes back, a read_file tool call for hello.py,
  3. after the read result comes back, a final text answer quoting the file.
Requests without tools (titles, compression, other side tasks) get a short text answer.
Both stream=true (SSE) and stream=false are supported.

Every request is written to <capture-dir>/<seq>-<method>-<path>.json with the path, query,
headers (credential-like values masked) and the full JSON body, plus a summary of the reply.
"""

import argparse
import json
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

PLACEHOLDER_KEYS = {"no-key-required", "Bearer no-key-required"}
SENSITIVE_HEADER = re.compile(r"auth|key|token|secret|cookie|session", re.I)


def mask_headers(headers):
    out = {}
    for name, value in headers.items():
        if SENSITIVE_HEADER.search(name) and value not in PLACEHOLDER_KEYS:
            scheme = value.split(" ", 1)[0] + " " if value.lower().startswith("bearer ") else ""
            out[name] = f"{scheme}***masked(len={len(value)})***"
        else:
            out[name] = value
    return out


class Stub:
    def __init__(self, args):
        self.args = args
        self.capture = Path(args.capture_dir)
        self.capture.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.seq = 0
        self.reasoning_field = args.reasoning_field or ("reasoning_content" if args.flavor == "llamacpp" else "reasoning")
        self.reject = {v.strip() for v in (args.reject_effort or "").split(",") if v.strip()}

    def next_seq(self):
        with self.lock:
            self.seq += 1
            return self.seq

    def models_payload(self):
        now = int(time.time())
        if self.args.flavor == "llamacpp":
            return {
                "models": [{"name": self.args.model, "model": self.args.model, "type": "model",
                            "capabilities": ["completion"]}],
                "object": "list",
                "data": [{"id": self.args.model, "object": "model", "created": now, "owned_by": "llamacpp",
                          "meta": {"vocab_type": 2, "n_vocab": 151936, "n_ctx_train": 262144,
                                   "n_embd": 2048, "n_params": 34660610688, "size": 21166757888}}],
            }
        return {"object": "list",
                "data": [{"id": self.args.model, "object": "model", "created": now, "owned_by": "vllm",
                          "root": "/models/stub", "parent": None, "max_model_len": self.args.ctx,
                          "permission": []}]}

    def props_payload(self):
        return {"default_generation_settings": {"n_ctx": self.args.ctx, "params": {"n_predict": -1}},
                "total_slots": 1, "model_alias": self.args.model, "model_path": "/models/stub.gguf",
                "build_info": "b12418-stub", "chat_template": "stub-template"}

    # ---- scripted conversation -------------------------------------------------------------
    def plan_reply(self, body):
        """Return (kind, payload): ("tool", {"name", "arguments"}) or ("text", str)."""
        tools = body.get("tools") or []
        names = [t.get("function", {}).get("name") for t in tools if isinstance(t, dict)]
        messages = body.get("messages") or []
        calls = {}
        last_call_name = None
        for m in messages:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    calls[tc.get("id")] = fn.get("name")
                    last_call_name = fn.get("name")
        last = messages[-1] if messages else {}
        users = [m for m in messages if m.get("role") == "user"]
        user_text = ""
        if users:  # only the latest user turn decides whether to start the tool script
            c = users[-1].get("content")
            user_text = c if isinstance(c, str) else json.dumps(c)
        if last.get("role") == "tool":
            answered = calls.get(last.get("tool_call_id")) or last_call_name
            if answered == "write_file" and "read_file" in names:
                return "tool", {"name": "read_file", "arguments": {"path": "hello.py"}}
            content = last.get("content")
            if not isinstance(content, str):
                content = json.dumps(content)
            return "text", "Done. hello.py was written and read back. Read result: " + content[:400]
        if "write_file" in names and "hello.py" in user_text:
            return "tool", {"name": "write_file", "arguments": {"path": "hello.py", "content": 'print("hello")\n'}}
        return "text", "Stub answer."

    def completion(self, body):
        kind, payload = self.plan_reply(body)
        model = body.get("model") or self.args.model
        rid = "chatcmpl-" + uuid.uuid4().hex[:12]
        reasoning = "stub reasoning" if self.args.emit_reasoning else None
        if kind == "tool":
            call_id = "call_" + uuid.uuid4().hex[:10]
            tool_call = {"id": call_id, "type": "function",
                         "function": {"name": payload["name"], "arguments": json.dumps(payload["arguments"])}}
            return rid, model, reasoning, None, [tool_call], "tool_calls"
        return rid, model, reasoning, payload, None, "stop"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "stub/1"
    stub: Stub = None

    def log_message(self, fmt, *a):
        sys.stderr.write("[stub] " + (fmt % a) + "\n")

    def _record(self, body_bytes, extra):
        seq = self.stub.next_seq()
        parts = urlsplit(self.path)
        safe = re.sub(r"[^A-Za-z0-9]+", "_", parts.path).strip("_") or "root"
        try:
            body = json.loads(body_bytes) if body_bytes else None
        except ValueError:
            body = {"_raw_text": body_bytes.decode("utf-8", "replace")}
        rec = {"seq": seq, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "method": self.command,
               "path": parts.path, "query": parse_qs(parts.query), "headers": mask_headers(dict(self.headers)),
               "body": body}
        rec.update(extra)
        (self.stub.capture / f"{seq:03d}-{self.command}-{safe}.json").write_text(json.dumps(rec, indent=2))
        return rec

    def _send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/")
        a = self.stub.args
        if path in ("/v1/models", "/models"):
            code, obj = 200, self.stub.models_payload()
        elif path in ("/v1/props", "/props") and a.flavor == "llamacpp":
            code, obj = 200, self.stub.props_payload()
        elif path == "/version" and a.flavor == "vllm":
            code, obj = 200, {"version": "0.33.0"}
        elif path in ("/health", "/v1/health"):
            code, obj = 200, {"status": "ok"}
        else:
            code, obj = 404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}}
        self._record(b"", {"reply_status": code})
        self._send_json(code, obj)

    def do_POST(self):
        raw = self._read_body()
        path = urlsplit(self.path).path.rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._record(raw, {"reply_status": 404})
            return self._send_json(404, {"error": {"code": 404, "message": "File Not Found"}})
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            self._record(raw, {"reply_status": 400})
            return self._send_json(400, {"error": {"message": "invalid json"}})
        effort = body.get("reasoning_effort")
        if effort is not None and (effort in self.stub.reject or "any" in self.stub.reject):
            msg = f"reasoning_effort {effort!r} is not supported by this stub (scripted rejection)"
            self._record(raw, {"reply_status": 400, "reply": msg})
            return self._send_json(400, {"error": {"message": msg, "type": "BadRequestError", "code": 400}})
        rid, model, reasoning, text, tool_calls, finish = self.stub.completion(body)
        summary = {"reply_status": 200, "reply_stream": bool(body.get("stream")),
                   "reply": {"text": text, "tool_calls": tool_calls, "finish_reason": finish}}
        self._record(raw, summary)
        rf = self.stub.reasoning_field
        usage = {"prompt_tokens": 1200, "completion_tokens": 20, "total_tokens": 1220}
        if not body.get("stream"):
            msg = {"role": "assistant", "content": text}
            if reasoning:
                msg[rf] = reasoning
            if tool_calls:
                msg["tool_calls"] = tool_calls
            return self._send_json(200, {"id": rid, "object": "chat.completion", "created": int(time.time()),
                                         "model": model, "choices": [{"index": 0, "message": msg,
                                                                       "finish_reason": finish}],
                                         "usage": usage})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        created = int(time.time())

        def chunk(delta, finish_reason=None, **extra):
            obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            obj.update(extra)
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        chunk({"role": "assistant", "content": None})
        if reasoning:
            for piece in ("stub ", "reasoning"):
                chunk({rf: piece})
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                args = tc["function"]["arguments"]
                chunk({"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                       "function": {"name": tc["function"]["name"], "arguments": ""}}]})
                half = len(args) // 2
                for piece in (args[:half], args[half:]):
                    chunk({"tool_calls": [{"index": i, "function": {"arguments": piece}}]})
        else:
            words = text.split(" ")
            for i, w in enumerate(words):
                chunk({"content": w + (" " if i < len(words) - 1 else "")})
        chunk({}, finish)
        if (body.get("stream_options") or {}).get("include_usage"):
            obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [], "usage": usage}
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--flavor", choices=("llamacpp", "vllm"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--ctx", type=int, default=131072)
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--reasoning-field", default=None)
    p.add_argument("--emit-reasoning", type=int, default=1)
    p.add_argument("--reject-effort", default="", help="comma list of reasoning_effort values to answer 400 (or 'any')")
    args = p.parse_args()
    Handler.stub = Stub(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    sys.stderr.write(f"[stub] {args.flavor} {args.model} ctx={args.ctx} on {args.host}:{args.port} -> {args.capture_dir}\n")
    sys.stderr.flush()
    print("READY", flush=True)  # the socket is bound; the runner waits for this line
    server.serve_forever()


if __name__ == "__main__":
    main()
