"""Recording OpenAI chat-completions stub for the Kimi experiments (adapted from the dsh lane B stub).

usage: python3 -I stub.py <port> <config.json>

Answers POST */chat/completions (SSE when the body asks for stream, else JSON) and GET */models.
Appends one reduced shape per request to the config's "record" file: body keys and the listed values, tool names,
message roles and per-message key names and sizes, never message text, so no system prompt is kept.

Config keys:
  record: JSONL file for the shapes
  rules: list of {"nth": n, "action": {...}}; the first rule whose nth equals the request number wins
  default: the action when no rule matches
Actions:
  {"type": "text", "text": str, "reasoning": str, "field": "reasoning_content"|"reasoning"}
  {"type": "tool", "name": str, "arguments": obj, "reasoning": str, "field": ...}
  {"type": "error", "status": int, "body": obj}
  {"type": "error_if_tools", "status": int, "body": obj, "otherwise": action}
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])
CONFIG = sys.argv[2]
LOCK = threading.Lock()
COUNT = [0]
REASONING_KEYS = ("reasoning_content", "reasoning", "reasoning_details", "thinking", "think")
KEEP = ("model", "max_tokens", "max_completion_tokens", "stream", "stream_options", "temperature", "top_p",
        "tool_choice", "parallel_tool_calls", "reasoning_effort", "reasoning", "thinking", "chat_template_kwargs",
        "enable_thinking", "n", "stop", "store", "user", "metadata", "response_format")


def load():
    with open(CONFIG, encoding="utf-8") as handle:
        return json.load(handle)


def size(value):
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value))


def reduce_message(message, canned):
    out = {"role": message.get("role")}
    content = message.get("content")
    if isinstance(content, str):
        out["content_chars"] = len(content)
    elif isinstance(content, list):
        out["content_parts"] = [part.get("type") for part in content if isinstance(part, dict)]
    elif content is None and "content" in message:
        out["content"] = None
    other = sorted(k for k in message if k not in ("role", "content"))
    if other:
        out["keys"] = other
    for key in REASONING_KEYS:
        if key in message:
            value = message[key]
            out[key] = {"chars": size(value), "equals_stub_reasoning": value == canned if canned else None}
    if "tool_calls" in message:
        out["tool_calls"] = [{"name": (call.get("function") or {}).get("name"),
                              "argument_keys": sorted(json.loads((call.get("function") or {}).get("arguments") or "{}"))}
                             for call in message["tool_calls"]]
    return out


def reduce_body(body, canned):
    shape = {"keys": sorted(body)}
    for key in KEEP:
        if key in body:
            shape[key] = body[key]
    for key in body:
        if key not in shape and any(w in key.lower() for w in ("reason", "think", "effort", "template")):
            shape[key] = body[key]
    if "prompt_cache_key" in body:
        value = body["prompt_cache_key"]
        shape["prompt_cache_key"] = {"present": True, "prefix": str(value).split("_")[0] + "_" if "_" in str(value) else "?"}
    tools = body.get("tools")
    shape["tools_present"] = "tools" in body
    shape["tools_count"] = len(tools) if isinstance(tools, list) else None
    shape["tool_names"] = [((t.get("function") or {}).get("name") or t.get("name")) for t in (tools or [])]
    messages = body.get("messages") or []
    shape["roles"] = [m.get("role") for m in messages]
    shape["messages"] = [reduce_message(m, canned) for m in messages]
    return shape


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def _record(self, cfg, rec):
        with LOCK:
            with open(cfg["record"], "a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def do_GET(self):
        cfg = load()
        self._record(cfg, {"t": round(time.time(), 3), "method": "GET", "path": self.path,
                           "user_agent": self.headers.get("User-Agent")})
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in cfg.get("models", [])]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        t0 = time.time()
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        cfg = load()
        with LOCK:
            COUNT[0] += 1
            n = COUNT[0]
        action = cfg.get("default", {"type": "text", "text": "OK"})
        for rule in cfg.get("rules", []):
            if rule.get("nth") == n:
                action = rule["action"]
                break
        canned = cfg.get("canned_reasoning", "")
        try:
            body = json.loads(raw)
            shape = reduce_body(body, canned)
        except ValueError:
            body, shape = {}, {"unparsed_bytes": len(raw)}
        auth = self.headers.get("Authorization", "")
        rec = {"n": n, "t": round(t0, 3), "method": "POST", "path": self.path,
               "user_agent": self.headers.get("User-Agent"),
               "authorization": (auth.split(" ")[0] + " <" + ("no-key-required" if auth.endswith("no-key-required") else "other") + ">") if auth else None,
               "x_msh_headers": sorted(k for k in self.headers.keys() if k.lower().startswith("x-msh")),
               "request": shape, "reply": action.get("type")}
        try:
            self._answer(body, action, n)
        except (BrokenPipeError, ConnectionResetError) as exc:
            rec["client_gone"] = repr(exc)
        rec["seconds"] = round(time.time() - t0, 3)
        self._record(cfg, rec)

    def _answer(self, body, action, n):
        kind = action.get("type", "text")
        if kind == "error_if_tools":  # vLLM without a tool parser: tools in the body give HTTP 400
            if "tools" in body:
                self._json(action.get("status", 400), action.get("body", {}))
                return
            action, kind = action.get("otherwise", {"type": "text", "text": "OK"}), "text"
        if kind == "error":
            self._json(action.get("status", 400), action.get("body", {}))
            return
        model = body.get("model", "local")
        created = int(time.time())
        cid = f"chatcmpl-stub-{n}"
        field = action.get("field", "reasoning_content")
        reasoning = action.get("reasoning", "")
        text = action.get("text", "OK")
        usage = {"prompt_tokens": 12000, "completion_tokens": 20, "total_tokens": 12020}
        finish = "tool_calls" if kind == "tool" else "stop"
        if not body.get("stream"):
            message = {"role": "assistant", "content": None if kind == "tool" else text}
            if reasoning:
                message[field] = reasoning
            if kind == "tool":
                message["tool_calls"] = [{"id": f"call_stub_{n}", "type": "function", "function": {
                    "name": action["name"], "arguments": json.dumps(action.get("arguments", {}))}}]
            self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": model,
                             "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def chunk(delta, finish_reason=None):
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        chunk({"role": "assistant", "content": ""})
        for piece in split(reasoning):
            chunk({field: piece})
        if kind == "tool":
            chunk({"tool_calls": [{"index": 0, "id": f"call_stub_{n}", "type": "function",
                                   "function": {"name": action["name"], "arguments": ""}}]})
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": json.dumps(action.get("arguments", {}))}}]})
        else:
            for piece in split(text):
                chunk({"content": piece})
        chunk({}, finish)
        if (body.get("stream_options") or {}).get("include_usage"):
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [], "usage": usage}
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def split(text, parts=3):
    if not text:
        return []
    step = max(1, len(text) // parts)
    return [text[i:i + step] for i in range(0, len(text), step)]


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    print(f"stub on 127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
