#!/usr/bin/env python3
"""Recording pass-through proxy: 127.0.0.1:<port> -> 127.0.0.1:<upstream_port>.

proxy.py <port> <upstream_port> <record.jsonl-path-file>
The third argument names a file whose content is the current record path, so the record
target can change between runs without a restart. Each request is forwarded unchanged; the
response is streamed back unchanged. The record keeps the request body, the upstream status,
arrival/first-byte/end times, and a summary of the SSE stream (content and reasoning characters
per field, finish_reason, usage, timings) or the error body.
"""
import http.client
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, UPSTREAM, RECORD_PTR = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
_lock = threading.Lock()


def record_path():
    with open(RECORD_PTR, encoding="utf-8") as handle:
        return handle.read().strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        conn = http.client.HTTPConnection("127.0.0.1", UPSTREAM, timeout=30)
        conn.request("GET", self.path)
        resp = conn.getresponse()
        data = resp.read()
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        t_arrive = time.time()
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        rec = {"t_arrive": t_arrive, "path": self.path, "body": body}
        headers = {"Content-Type": "application/json"}
        if self.headers.get("Authorization"):
            headers["Authorization"] = self.headers["Authorization"]
        conn = http.client.HTTPConnection("127.0.0.1", UPSTREAM, timeout=1800)
        summary = {"fields": {}, "finish": None, "usage": None, "timings": None, "chunks": 0}
        try:
            conn.request("POST", self.path, body=raw, headers=headers)
            resp = conn.getresponse()
            rec["status"] = resp.status
            rec["t_headers"] = time.time()
            self.send_response(resp.status)
            ctype = resp.getheader("Content-Type", "application/json")
            self.send_header("Content-Type", ctype)
            self.send_header("Connection", "close")
            self.end_headers()
            if "text/event-stream" not in ctype:
                data = resp.read()
                self.wfile.write(data)
                rec["error_body"] = data.decode("utf-8", "replace")[:2000]
            else:
                first = None
                buf = b""
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
                    buf = line.strip()
                    if not buf.startswith(b"data: ") or buf == b"data: [DONE]":
                        continue
                    try:
                        obj = json.loads(buf[6:])
                    except ValueError:
                        continue
                    summary["chunks"] += 1
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        for key, value in delta.items():
                            if isinstance(value, str) and value:
                                if first is None:
                                    first = time.time()
                                summary["fields"][key] = summary["fields"].get(key, 0) + len(value)
                            elif key == "tool_calls" and value:
                                summary["fields"]["tool_calls"] = summary["fields"].get("tool_calls", 0) + 1
                        if choice.get("finish_reason"):
                            summary["finish"] = choice["finish_reason"]
                    if obj.get("usage"):
                        summary["usage"] = obj["usage"]
                    if obj.get("timings"):
                        summary["timings"] = obj["timings"]
                rec["t_first_token"] = first
            rec["summary"] = summary
        except (BrokenPipeError, ConnectionResetError) as exc:
            rec["client_gone"] = repr(exc)
        except OSError as exc:
            rec["upstream_error"] = repr(exc)
            try:
                self.send_response(502)
                self.end_headers()
            except OSError:
                pass
        finally:
            rec["t_end"] = time.time()
            conn.close()
            with _lock:
                with open(record_path(), "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    print(f"proxy 127.0.0.1:{PORT} -> 127.0.0.1:{UPSTREAM}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
