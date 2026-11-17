"""Recording proxy: 127.0.0.1:<port> -> upstream llama.cpp. Stdlib only.
Saves a reduced request body and the response status per request into the shapes dir."""
import http.client, http.server, json, sys, threading, time, os

PORT, UP_HOST, UP_PORT, OUT = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
LOCK = threading.Lock()
COUNTER = [0]
KEEP = ("model", "max_tokens", "max_completion_tokens", "stream", "stream_options", "temperature",
        "top_p", "top_k", "reasoning_effort", "reasoning", "thinking", "chat_template_kwargs",
        "enable_thinking", "prompt_cache_key", "tool_choice", "parallel_tool_calls", "n", "stop")

def reduce(body):
    try:
        d = json.loads(body)
    except Exception:
        return {"unparsed_bytes": len(body)}
    out = {"keys": sorted(d.keys())}
    for k in KEEP:
        if k in d:
            out[k] = d[k]
    for k in d:
        if any(w in k.lower() for w in ("reason", "think", "effort")) and k not in out:
            out[k] = d[k]
    tools = d.get("tools") or []
    out["tool_count"] = len(tools)
    out["tool_names"] = [((t.get("function") or {}).get("name") or t.get("name")) for t in tools]
    msgs = []
    for m in d.get("messages") or []:
        e = {"role": m.get("role")}
        c = m.get("content")
        if isinstance(c, str):
            e["content"] = c if len(c) <= 200 else "<%d chars dropped>" % len(c)
        elif isinstance(c, list):
            e["content_parts"] = [p.get("type") for p in c]
        for k in m:
            if k not in ("role", "content"):
                v = m[k]
                if k == "tool_calls":
                    e[k] = [{"name": (tc.get("function") or {}).get("name"),
                             "arguments": ((tc.get("function") or {}).get("arguments") or "")[:200]} for tc in v]
                elif isinstance(v, str) and len(v) > 200:
                    e[k] = "<%d chars dropped>" % len(v)
                else:
                    e[k] = v
        msgs.append(e)
    out["messages"] = msgs
    return out

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _go(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        with LOCK:
            COUNTER[0] += 1
            idx = COUNTER[0]
        rec = {"idx": idx, "time": time.strftime("%H:%M:%S"), "method": self.command, "path": self.path,
               "headers": {k: ("<masked>" if k.lower() == "authorization" else v) for k, v in self.headers.items()}}
        if body:
            rec["request"] = reduce(body)
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection", "accept-encoding")}
        t0 = time.time()
        conn = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=900)
        try:
            conn.request(self.command, self.path, body=body or None, headers=hdrs)
            r = conn.getresponse()
        except Exception as e:
            rec["error"] = repr(e)
            self.send_response(502); self.send_header("Content-Length", "0"); self.send_header("Connection", "close"); self.end_headers()
            self._save(idx, rec); return
        rec["status"] = r.status
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        seen = {"reasoning_content": False, "tool_calls": False, "content": False}
        finish, usage, tail = None, None, b""
        errbody = b""
        while True:
            chunk = r.read1(65536)
            if not chunk:
                break
            self.wfile.write(chunk); self.wfile.flush()
            buf = tail + chunk
            lines = buf.split(b"\n"); tail = lines.pop()
            for ln in lines:
                if r.status >= 400 and len(errbody) < 2000:
                    errbody += ln + b"\n"
                if not ln.startswith(b"data: ") or ln.strip() == b"data: [DONE]":
                    continue
                try:
                    j = json.loads(ln[6:])
                except Exception:
                    continue
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices") or []:
                    dl = ch.get("delta") or ch.get("message") or {}
                    for k in seen:
                        if dl.get(k):
                            seen[k] = True
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        if r.status >= 400:
            rec["error_body"] = (errbody + tail)[:2000].decode("utf-8", "replace")
        rec["response"] = {"seen": seen, "finish_reason": finish, "usage": usage, "seconds": round(time.time() - t0, 2)}
        self.close_connection = True
        self._save(idx, rec)
    def _save(self, idx, rec):
        tag = os.environ.get("PROXY_TAG", "")
        try:
            tag = open(os.path.join(OUT, ".tag")).read().strip()
        except Exception:
            pass
        with open(os.path.join(OUT, "%s-%03d.json" % (tag or "req", idx)), "w") as f:
            json.dump(rec, f, indent=1)
    do_POST = do_GET = _go

http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
