#!/usr/bin/env python3
"""Recording OpenAI chat-completions stub for lane B.

Answers POST /v1/chat/completions (streaming SSE or plain JSON) and GET /v1/models.
Every request is appended to the JSONL file named by the config's "record" key, with
arrival, slot-start and end times, the kind (main, title, compaction), the parsed body
and the action taken. The config file (argv[2]) is re-read on every request, so the
behaviour can change between runs without a restart.

Config keys:
  record: path of the JSONL record file
  single_slot: true serializes request handling like llama.cpp -np 1
  reasoning_field: "reasoning_content" (llama.cpp) or "reasoning" (vLLM)
  default: action for requests no rule matches
  rules: list of {"match": {"kind": k, "nth": n, "min_nth": n}, "action": {...}}
Actions:
  {"type": "text", "text": str, "reasoning": str, "first_delay": s, "chunk_delay": s,
   "usage": {"prompt_tokens": n, "completion_tokens": n}, "finish": "stop"|"length"}
  {"type": "tool", "name": str, "arguments": obj, ...same timing keys}
  {"type": "error", "status": int, "body": obj}
  {"type": "headers_then_hang", "seconds": s}   (200 + SSE headers, then silence)
  {"type": "hang_before_headers", "seconds": s} (silence before any byte)
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
PORT = int(sys.argv[1])
CONFIG = sys.argv[2]

_lock = threading.Lock()
_slot = threading.Lock()
_counts = {"all": 0}


def load_config():
    with open(CONFIG, encoding="utf-8") as handle:
        return json.load(handle)


def classify(body):
    messages = body.get("messages") or []
    system = " ".join(
        m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
        for m in messages if m.get("role") in ("system", "developer"))
    if "Create a concise title" in system:
        return "title"
    last_user = [m for m in messages if m.get("role") == "user"]
    if last_user:
        content = last_user[-1].get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        if "compaction engine" in text:
            return "compaction"
    return "main"


def pick_action(cfg, kind, kind_n):
    for rule in cfg.get("rules", []):
        match = rule.get("match", {})
        if match.get("kind", "any") not in ("any", kind):
            continue
        if "nth" in match and match["nth"] != kind_n:
            continue
        if "min_nth" in match and kind_n < match["min_nth"]:
            continue
        return rule["action"]
    return cfg.get("default", {"type": "text", "text": "OK"})


def append_record(path, record):
    with _lock:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def do_GET(self):
        cfg = load_config()
        if self.path.rstrip("/").endswith("/models"):
            models = cfg.get("models", ["local"])
            self._send_json(200, {"object": "list", "data": [
                {"id": m, "object": "model", "owned_by": "stub"} for m in models]})
            append_record(cfg["record"], {"t_arrive": time.time(), "method": "GET", "path": self.path})
            return
        self._send_json(404, {"error": {"message": "not found", "type": "NotFound", "code": 404}})

    def do_POST(self):
        t_arrive = time.time()
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        cfg = load_config()
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"_unparsed": raw.decode("utf-8", "replace")}
        kind = classify(body) if isinstance(body, dict) else "other"
        with _lock:
            if _counts.get("record") != cfg["record"]:
                _counts.clear()
                _counts.update({"all": 0, "record": cfg["record"]})
            _counts["all"] += 1
            _counts[kind] = _counts.get(kind, 0) + 1
            n, kind_n = _counts["all"], _counts[kind]
        action = pick_action(cfg, kind, kind_n)
        auth = self.headers.get("Authorization", "")
        record = {
            "n": n, "kind": kind, "kind_n": kind_n, "t_arrive": t_arrive,
            "method": "POST", "path": self.path,
            "headers": {"user-agent": self.headers.get("User-Agent"),
                        "authorization_scheme": auth.split(" ")[0] if auth else None,
                        "content-type": self.headers.get("Content-Type")},
            "body": body, "action": action,
        }
        single = cfg.get("single_slot", False)
        if single:
            _slot.acquire()
        record["t_start"] = time.time()
        try:
            self._answer(cfg, body, action, record)
        except (BrokenPipeError, ConnectionResetError) as exc:
            record["client_gone"] = repr(exc)
        finally:
            record["t_end"] = time.time()
            if single:
                _slot.release()
            append_record(cfg["record"], record)

    def _answer(self, cfg, body, action, record):
        kind = action.get("type", "text")
        if kind == "error":
            self._send_json(action.get("status", 400), action.get("body", {}))
            return
        if kind == "hang_before_headers":
            time.sleep(action.get("seconds", 10))
            return
        stream = bool(body.get("stream")) if isinstance(body, dict) else False
        model = body.get("model", "local") if isinstance(body, dict) else "local"
        created = int(time.time())
        cid = f"chatcmpl-stub-{record['n']}"
        field = cfg.get("reasoning_field", "reasoning_content")
        reasoning = action.get("reasoning", "")
        text = action.get("text", "OK")
        usage = action.get("usage", {"prompt_tokens": 1000, "completion_tokens": 10})
        usage = {**usage, "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)}
        finish = "tool_calls" if kind == "tool" else action.get("finish", "stop")
        if not stream:
            message = {"role": "assistant", "content": text if kind != "tool" else None}
            if reasoning:
                message[field] = reasoning
            if kind == "tool":
                message["tool_calls"] = [{"id": "call_stub_1", "type": "function", "function": {
                    "name": action["name"], "arguments": json.dumps(action.get("arguments", {}))}}]
            self._send_json(200, {"id": cid, "object": "chat.completion", "created": created, "model": model,
                                  "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                                  "usage": usage})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.flush()
        if kind == "headers_then_hang":
            time.sleep(action.get("seconds", 10))
            return
        time.sleep(action.get("first_delay", 0))
        delay = action.get("chunk_delay", 0)

        def chunk(delta, finish_reason=None, extra=None):
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            if extra:
                obj.update(extra)
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        chunk({"role": "assistant", "content": ""})
        for piece in split(reasoning, action.get("chunks", 3)):
            time.sleep(delay)
            chunk({field: piece})
        if kind == "tool":
            time.sleep(delay)
            chunk({"tool_calls": [{"index": 0, "id": "call_stub_1", "type": "function",
                                   "function": {"name": action["name"], "arguments": ""}}]})
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": json.dumps(action.get("arguments", {}))}}]})
        else:
            for piece in split(text, action.get("chunks", 3)):
                time.sleep(delay)
                chunk({"content": piece})
        chunk({}, finish)
        if isinstance(body, dict) and (body.get("stream_options") or {}).get("include_usage"):
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [], "usage": usage}
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def split(text, parts):
    if not text:
        return []
    size = max(1, len(text) // max(1, parts))
    return [text[i:i + size] for i in range(0, len(text), size)]


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"stub listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
